"""Full-corpus BEIR transfer, custodian-only after all editor selections freeze."""

import os, sys, json, time, hashlib, csv, gc
from pathlib import Path
import numpy as np, torch, pytrec_eval
from sentence_transformers import SentenceTransformer
import argparse

parser = argparse.ArgumentParser(
    description="Evaluate one MiniLM checkpoint on a full BEIR corpus"
)
parser.add_argument("--model", type=Path, required=True)
parser.add_argument("--dataset", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()
name, task = args.model.name, args.dataset.name
out = args.output
out.parent.mkdir(parents=True, exist_ok=True)
if out.exists():
    raise FileExistsError(out)
torch.set_num_threads(8)
starttime = time.time()
data = args.dataset
corpus = [json.loads(x) for x in (data / "corpus.jsonl").open()]
ids = [str(x["_id"]) for x in corpus]
texts = [(x.get("title", "") + " " + x["text"]).strip() for x in corpus]
queries = {
    str(x["_id"]): x["text"] for x in map(json.loads, (data / "queries.jsonl").open())
}
qrels = {}
with (data / "qrels/test.tsv").open() as f:
    for row in csv.DictReader(f, delimiter="\t"):
        q, d, v = row["query-id"], row["corpus-id"], int(row["score"])
        if v > 0:
            qrels.setdefault(q, {})[d] = v
idset = set(ids)
qids = sorted(qrels)
assert all(q in queries for q in qids)
missing_qrels = [
    {"query_id": q, "document_id": d, "relevance": v}
    for q, rel in qrels.items()
    for d, v in rel.items()
    if d not in idset
]


# Official BEIR loader retains qrels even when judged documents are absent from corpus.
# Keep every query and its denominator; missing judged documents cannot be retrieved.
def unit(x):
    x = np.asarray(x, np.float32)
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)


m = SentenceTransformer(
    str(args.model), device="cuda", model_kwargs={"torch_dtype": torch.float16}
)
m.max_seq_length = 256
m.eval()
encoding_started = time.time()
de = unit(
    m.encode(texts, batch_size=128, convert_to_numpy=True, show_progress_bar=False)
)
qe = unit(
    m.encode(
        [queries[q] for q in qids],
        batch_size=128,
        convert_to_numpy=True,
        show_progress_bar=False,
    )
)
encoding_seconds = time.time() - encoding_started
del m
gc.collect()
torch.cuda.empty_cache()
dt = torch.tensor(de, device="cuda")
index = {d: i for i, d in enumerate(ids)}
rows = {}
rankings = {}
for start in range(0, len(qids), 64):
    batch = qids[start : start + 64]
    sim = torch.tensor(qe[start : start + 64], device="cuda") @ dt.T
    for i, q in enumerate(batch):
        if q in index:
            sim[i, index[q]] = -torch.inf
    scores, positions = sim.topk(min(1000, len(ids)), dim=1)
    positions = positions.cpu().numpy()
    scores = scores.cpu().numpy()
    for i, q in enumerate(batch):
        pairs = sorted(
            [(ids[j], float(v)) for j, v in zip(positions[i], scores[i])],
            key=lambda x: (x[1], x[0]),
            reverse=True,
        )
        ranked = [d for d, v in pairs]
        rel = qrels[q]
        ranks = [j + 1 for j, d in enumerate(ranked) if d in rel]
        dcg = sum(rel.get(d, 0) / np.log2(j + 2) for j, d in enumerate(ranked[:10]))
        ideal = sum(
            v / np.log2(j + 2)
            for j, v in enumerate(sorted(rel.values(), reverse=True)[:10])
        )
        rows[q] = {
            "p1": float(ranked[0] in rel),
            "mrr10": 1 / min(ranks) if ranks and min(ranks) <= 10 else 0.0,
            "ndcg10": dcg / ideal,
            "recall100": sum(r <= 100 for r in ranks) / len(rel),
        }
        rankings[q] = {"ids": ranked, "scores": [v for d, v in pairs]}
official = pytrec_eval.RelevanceEvaluator(
    qrels, {"P_1", "ndcg_cut_10", "recall_100"}
).evaluate({q: dict(zip(v["ids"], v["scores"])) for q, v in rankings.items()})
for q, v in official.items():
    for own, standard in [
        ("p1", "P_1"),
        ("ndcg10", "ndcg_cut_10"),
        ("recall100", "recall_100"),
    ]:
        assert abs(rows[q][own] - v[standard]) < 1e-9, (
            q,
            own,
            rows[q][own],
            v[standard],
        )
summary = {
    k: float(np.mean([x[k] for x in rows.values()]))
    for k in ["p1", "mrr10", "ndcg10", "recall100"]
}
summary.update(
    {
        "arm": name,
        "task": task,
        "queries": len(qids),
        "corpus": len(ids),
        "encoding_seconds": encoding_seconds,
        "wall_seconds": time.time() - starttime,
        "checkpoint_sha256": hashlib.sha256(
            (args.model / "model.safetensors").read_bytes()
        ).hexdigest(),
        "missing_judged_documents": len(missing_qrels),
        "queries_with_no_retrievable_judged_document": sum(
            not any(d in idset for d in rel) for rel in qrels.values()
        ),
        "qrels_policy": "Retain official qrels and all queries, including absent corpus IDs, matching BEIR loader/evaluator.",
        "self_id_excluded": True,
        "corpus_input": "title + space + text, strip",
        "max_tokens": 256,
    }
)
for suffix, value in [
    (".perquery.json", rows),
    (".rankings.json", rankings),
    (".json", summary),
]:
    p = out.with_name(out.stem + suffix)
    temp = p.with_suffix(".tmp")
    temp.write_text(json.dumps(value))
    temp.replace(p)
print(json.dumps(summary), flush=True)
