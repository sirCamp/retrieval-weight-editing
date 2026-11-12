"""Build TRAIN embeddings and hard competitors for a recorded fitting configuration."""

from engine import R, corpus, encode, loadsets, model, score, write
import numpy as np


def main():
    sets, queries, qrels = loadsets()
    ids, texts = corpus()
    encoder = model("base")
    halves = np.array_split(np.arange(len(texts)), 2)
    documents = np.concatenate(
        [encode(encoder, [texts[i] for i in half]) for half in halves]
    )
    folder = R / "cache_embeddings/base"
    folder.mkdir(parents=True, exist_ok=True)
    np.save(folder / "corpus.npy", documents)
    query_ids = sets["fit_expansion"] + sets["feedback"]
    vectors = encode(encoder, [queries[q] for q in query_ids])
    _, rows = score(
        query_ids,
        vectors,
        ids,
        documents,
        qrels,
        queries,
        R / "splits/base_hard_competitors.json",
    )
    write(R / "results/base.fit_pool.perquery.json", rows)
    for split in ("feedback", "diagnostic"):
        positions = {qid: i for i, qid in enumerate(query_ids)}
        subset = sets[split]
        summary, scores = score(
            subset,
            vectors[[positions[q] for q in subset]],
            ids,
            documents,
            qrels,
            queries,
            R / f"results/base.{split}.top.json",
        )
        write(R / f"results/base.{split}.json", summary)
        write(R / f"results/base.{split}.perquery.json", scores)


if __name__ == "__main__":
    main()
