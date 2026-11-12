import os, time, json, gc, hashlib
from pathlib import Path
import numpy as np
import torch
from sentence_transformers import SentenceTransformer

R = Path(os.environ["MINILM_EXPERIMENT_ROOT"]).resolve()
torch.set_num_threads(8)


def write(p, x):
    p = Path(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    t = p.with_suffix(p.suffix + f".{os.getpid()}.tmp")
    t.write_text(json.dumps(x, indent=2))
    t.replace(p)


def status(stage, **kw):
    x = {
        "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "stage": stage,
        **kw,
    }
    write(R / "status.json", x)
    print(json.dumps(x), flush=True)


def unit(a):
    a = np.asarray(a, dtype=np.float32)
    return a / np.maximum(np.linalg.norm(a, axis=1, keepdims=True), 1e-12)


def model(name):
    m = SentenceTransformer(
        str(R / "models" / name),
        device="cuda",
        model_kwargs={"torch_dtype": torch.float16},
    )
    m.max_seq_length = 256
    m.eval()
    return m


def corpus():
    ids = []
    texts = []
    for line in (R / "data/corpus_search.jsonl").open():
        x = json.loads(line)
        ids.append(x["id"])
        texts.append(x["text"])
    return ids, texts


def encode(m, texts):
    return unit(
        m.encode(
            texts,
            batch_size=128,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=False,
        )
    )


def loadsets():
    return [
        json.loads((R / "splits" / n).read_text())
        for n in ["query_ids.json", "queries.json", "qrels.json"]
    ]


def stratum(text):
    n = len(text.split())
    return "short" if n <= 5 else ("medium" if n <= 9 else "long")


def score(qids, qe, dids, de, rels, queries, topfile=None):
    # FP32 normalized cosine and exact exhaustive search within fixed shared corpus.
    dt = torch.tensor(unit(de), device="cuda")
    dmap = {d: i for i, d in enumerate(dids)}
    rows = {}
    tops = {}
    t = time.time()
    for start in range(0, len(qids), 64):
        ids = qids[start : start + 64]
        qt = torch.tensor(unit(qe[start : start + 64]), device="cuda")
        sim = qt @ dt.T
        vals, inds = sim.topk(100, dim=1)
        va = vals.cpu().numpy()
        ix = inds.cpu().numpy()
        for i, q in enumerate(ids):
            relevant = rels[q]
            ranked = [dids[j] for j in ix[i]]
            rr = [j + 1 for j, d in enumerate(ranked) if d in relevant]
            mrr = 1 / min(rr) if rr and min(rr) <= 10 else 0.0
            dcg = sum(
                relevant[d] / np.log2(j + 2)
                for j, d in enumerate(ranked[:10])
                if d in relevant
            )
            ideal = sum(
                v / np.log2(j + 2)
                for j, v in enumerate(sorted(relevant.values(), reverse=True)[:10])
            )
            pp = [dmap[d] for d in relevant if d in dmap]
            ps, pind = sim[i, pp].max(0)
            positive = pp[int(pind)]
            rank = int((sim[i] > ps).sum()) + 1
            competitors = [j for j, d in enumerate(ranked) if d not in relevant]
            ci = competitors[0] if competitors else 99
            rows[q] = {
                "p1": float(mrr == 1.0),
                "mrr10": mrr,
                "ndcg10": dcg / ideal,
                "recall100": len(rr) / len(relevant),
                "best_positive_rank": rank,
                "positive_cosine": float(ps),
                "margin": float(ps) - float(va[i, ci]),
                "stratum": stratum(queries[q]),
                "best_positive": dids[positive],
            }
            tops[q] = {"ids": ranked[:10], "scores": va[i, :10].tolist()}
    summary = {
        k: float(np.mean([x[k] for x in rows.values()]))
        for k in ["p1", "mrr10", "ndcg10", "recall100", "best_positive_rank", "margin"]
    }
    summary["strata"] = {
        s: float(np.mean([x["mrr10"] for x in rows.values() if x["stratum"] == s]))
        for s in ["short", "medium", "long"]
    }
    summary["metrics_by_stratum"] = {
        s: {
            k: float(np.mean([x[k] for x in rows.values() if x["stratum"] == s]))
            for k in ["p1", "mrr10", "ndcg10", "recall100"]
        }
        for s in ["short", "medium", "long"]
    }
    summary["queries"] = len(rows)
    summary["seconds"] = time.time() - t
    if topfile:
        write(topfile, tops)
    del dt
    torch.cuda.empty_cache()
    return summary, rows


def evaluate(name, split="feedback"):
    sets, queries, rels = loadsets()
    ids, texts = corpus()
    folder = R / "cache_embeddings" / name
    folder.mkdir(parents=True, exist_ok=True)
    m = model(name)
    path = folder / "corpus.npy"
    start = time.time()
    if not path.exists():
        a = encode(m, texts)
        np.save(path, a)
    else:
        a = np.load(path)
    qids = sets[split]
    qe = encode(m, [queries[q] for q in qids])
    summary, rows = score(
        qids, qe, ids, a, rels, queries, R / "results" / f"{name}.{split}.top.json"
    )
    summary["wall_seconds"] = time.time() - start
    write(R / "results" / f"{name}.{split}.json", summary)
    write(R / "results" / f"{name}.{split}.perquery.json", rows)
    del m
    gc.collect()
    torch.cuda.empty_cache()
    return summary
