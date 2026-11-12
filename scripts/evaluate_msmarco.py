"""Evaluate a saved MiniLM checkpoint on the released full-corpus MS MARCO split."""

import argparse
import hashlib
import json
import math
from pathlib import Path
import tempfile

ROOT = Path(__file__).resolve().parents[1]
COUNT = 8_841_823
METRICS = ("p1", "mrr10", "ndcg10", "recall100")


def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def test_data(data):
    protocol = json.loads((ROOT / "results/experiment_manifest.json").read_text())[
        "msmarco"
    ]
    frozen_ids = ROOT / "results/msmarco_test_ids.json"
    ids = json.loads(frozen_ids.read_text())
    expected = protocol["test_file_sha256"]
    if sha(frozen_ids) != expected["query_ids.json"]:
        raise ValueError("Test ID checksum mismatch")
    selected = set(ids)
    with (data / "queries.train.tsv").open(encoding="utf-8") as stream:
        queries = {
            q: t.rstrip("\n")
            for q, t in (line.split("\t", 1) for line in stream)
            if q in selected
        }
    qrels = {q: {} for q in ids}
    with (data / "qrels.train.tsv").open(encoding="utf-8") as stream:
        for line in stream:
            query, _, passage, grade = line.split()
            if query in selected and int(grade) > 0:
                qrels[query][passage] = int(grade)
    ordered_queries = {q: queries[q] for q in ids}
    for name, value in [("queries.json", ordered_queries), ("qrels.json", qrels)]:
        if (
            hashlib.sha256(json.dumps(value, indent=2).encode()).hexdigest()
            != expected[name]
        ):
            raise ValueError(f"Test data checksum mismatch: {name}")
    collection = data / "collection.tsv"
    if sha(collection) != protocol["collection_tsv_sha256"]:
        raise ValueError("Passage collection checksum mismatch")
    return collection, ids, ordered_queries, qrels


def unit(vectors):
    import numpy as np

    vectors = np.asarray(vectors, np.float32)
    return vectors / np.maximum(np.linalg.norm(vectors, axis=1, keepdims=True), 1e-12)


def encode(model, texts):
    return unit(
        model.encode(
            texts, batch_size=128, convert_to_numpy=True, show_progress_bar=False
        )
    )


def encode_passages(model, collection, cache):
    import numpy as np

    embeddings = np.lib.format.open_memmap(
        cache, mode="w+", dtype=np.float32, shape=(COUNT, 384)
    )
    with collection.open("rb") as stream:
        for lower, upper in ((0, 4_420_912), (4_420_912, COUNT)):
            for start in range(lower, upper, 4096):
                end = min(start + 4096, upper)
                texts = []
                for index in range(start, end):
                    passage, body = (
                        stream.readline().decode("utf-8").rstrip("\n").split("\t", 1)
                    )
                    if int(passage) != index:
                        raise ValueError(f"Unexpected passage ID at row {index}")
                    texts.append(body)
                embeddings[start:end] = encode(model, texts)
                if start % 1_048_576 == 0:
                    print(f"Encoded {end:,}/{COUNT:,} passages", flush=True)
    embeddings.flush()
    return embeddings


def rank(query_vectors, passages):
    import torch

    queries = torch.tensor(query_vectors, device="cuda")
    k = min(100, len(passages))
    values = torch.full((len(queries), k), -torch.inf, device="cuda")
    positions = torch.zeros((len(queries), k), dtype=torch.long, device="cuda")
    for start in range(0, len(passages), 200_000):
        documents = torch.tensor(unit(passages[start : start + 200_000]), device="cuda")
        for q in range(0, len(queries), 128):
            section = slice(q, q + 128)
            scores, ids = (queries[section] @ documents.T).topk(
                min(k, len(documents)), dim=1
            )
            combined_values = torch.cat((values[section], scores), dim=1)
            combined_ids = torch.cat((positions[section], ids + start), dim=1)
            values[section], selected = combined_values.topk(k, dim=1)
            positions[section] = combined_ids.gather(1, selected)
    return positions.cpu().numpy()


def score(ranked, relevant):
    grades = [relevant.get(str(index), 0) for index in ranked]
    positive = [i + 1 for i, grade in enumerate(grades) if grade > 0]
    dcg = math.fsum(grade / math.log2(i + 2) for i, grade in enumerate(grades[:10]))
    ideal = math.fsum(
        grade / math.log2(i + 2)
        for i, grade in enumerate(sorted(relevant.values(), reverse=True)[:10])
    )
    return {
        "p1": float(grades[0] > 0),
        "mrr10": 1 / positive[0] if positive and positive[0] <= 10 else 0.0,
        "ndcg10": dcg / ideal,
        "recall100": len(positive) / len(relevant),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data",
        type=Path,
        required=True,
        help="Extracted collectionandqueries.tar.gz directory",
    )
    parser.add_argument("--model", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--validate-inputs", action="store_true")
    args = parser.parse_args()
    collection, ids, queries, qrels = test_data(args.data)
    if args.validate_inputs:
        print(f"Verified test data for {len(ids):,} queries and {COUNT:,} passages")
        return
    if args.model is None or args.output is None:
        parser.error("--model and --output are required for evaluation")
    if args.output.exists():
        raise FileExistsError(args.output)

    import torch
    from sentence_transformers import SentenceTransformer

    torch.set_num_threads(8)
    torch.backends.cuda.matmul.allow_tf32 = False
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=args.output.parent) as temporary:
        model = SentenceTransformer(
            str(args.model), device="cuda", model_kwargs={"torch_dtype": torch.float16}
        )
        model.max_seq_length = 256
        model.eval()
        passages = encode_passages(model, collection, Path(temporary) / "corpus.npy")
        query_vectors = encode(model, [queries[q] for q in ids])
        del model
        rankings = rank(query_vectors, passages)
        del passages
    rows = {query: score(ranked, qrels[query]) for query, ranked in zip(ids, rankings)}
    summary = {
        metric: math.fsum(row[metric] for row in rows.values()) / len(rows)
        for metric in METRICS
    }
    summary.update(
        queries=len(ids),
        corpus=COUNT,
        checkpoint_sha256=sha(args.model / "model.safetensors"),
    )
    args.output.mkdir()
    (args.output / "full.json").write_text(json.dumps(summary, indent=2) + "\n")
    (args.output / "full.perquery.json").write_text(json.dumps(rows) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
