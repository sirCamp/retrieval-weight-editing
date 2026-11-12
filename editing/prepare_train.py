import hashlib, json, unicodedata, time, os
from pathlib import Path
import numpy as np

R = Path(os.environ["MINILM_EXPERIMENT_ROOT"]).resolve()
D = R / "data"
seed = 20261105
ARCHIVE_SHA256 = "decac356eb8cc5b9cea2e30b8738dc6f367e4147aaefe8fc7526ddda382fd2fc"
SEARCH_CORPUS_SHA256 = (
    "4334bde48e97a730731b7eace49b117c82ef6aac4221788fb47991ab026fae14"
)


def file_sha256(path):
    with path.open("rb") as stream:
        digest = hashlib.sha256()
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_search_corpus():
    actual = file_sha256(D / "corpus_search.jsonl")
    if actual != SEARCH_CORPUS_SHA256:
        raise ValueError(f"Search corpus checksum mismatch: {actual}")


def write(p, x):
    p = Path(p)
    t = p.with_suffix(p.suffix + ".tmp")
    t.write_text(json.dumps(x, indent=2))
    t.replace(p)


def norm(t):
    return " ".join(unicodedata.normalize("NFKC", t).lower().split())


def order(ids):
    return sorted(ids, key=lambda x: hashlib.sha256(f"{seed}:{x}".encode()).digest())


def find(name):
    a = list(D.rglob(name))
    assert len(a) == 1, (name, a)
    return a[0]


def queries(p):
    return {q: t.rstrip("\n") for q, t in (line.split("\t", 1) for line in p.open())}


def qrels(p):
    r = {}
    for line in p.open():
        q, _, d, v = line.split()
        if int(v) > 0:
            r.setdefault(q, {})[d] = int(v)
    return r


(R / "splits").mkdir(parents=True, exist_ok=True)
if (D / "archive.sha256").read_text().split()[0] != ARCHIVE_SHA256:
    raise ValueError("MS MARCO source archive checksum mismatch")
prepared = R / "prepared.json"
if prepared.exists():
    metadata = json.loads(prepared.read_text())
    verify_search_corpus()
    if metadata["source_archive_sha256"] != ARCHIVE_SHA256:
        raise ValueError("Prepared data has a different source archive")
    required = {
        "query_ids.json",
        "queries.json",
        "qrels.json",
        "corpus_search.npy",
        "corpus_1m.npy",
    }
    hashes = metadata["split_hashes"]
    if not required <= hashes.keys():
        raise ValueError("Prepared split manifest is incomplete")
    for name in required:
        if file_sha256(R / "splits" / name) != hashes[name]:
            raise ValueError(f"Prepared split checksum mismatch: {name}")
    print("Verified existing TRAIN preparation", flush=True)
    raise SystemExit(0)
train = queries(find("queries.train.tsv"))
dev = queries(find("queries.dev.small.tsv"))
tr = qrels(find("qrels.train.tsv"))
dr = qrels(find("qrels.dev.small.tsv"))
devseen = set()
dev_ids = []
for q in order(set(dev) & set(dr)):
    n = norm(dev[q])
    if n not in devseen:
        devseen.add(n)
        dev_ids.append(q)
seen = set(devseen)
train_ids = []
for q in order(set(train) & set(tr)):
    n = norm(train[q])
    if n not in seen:
        seen.add(n)
        train_ids.append(q)
assert len(train_ids) > 52000 and len(dev_ids) > 5000
sets = {
    "fit": train_ids[:10000],
    "fit_expansion": train_ids[:50000],
    "feedback": train_ids[50000:52000],
    "diagnostic": train_ids[:512],
    "confirmation": dev_ids[:1000],
    "sealed": dev_ids[1000:],
}
write(R / "splits/query_ids.json", sets)
allq = {q: train[q] for q in train_ids[:52000]}
allq.update({q: dev[q] for q in dev_ids})
rels = {q: tr[q] for q in train_ids[:52000]}
rels.update({q: dr[q] for q in dev_ids})
write(R / "splits/queries.json", allq)
write(R / "splits/qrels.json", rels)
collection = find("collection.tsv")
expected = 8841823
offset = np.zeros(expected, dtype=np.int64)
count = 0
with collection.open("rb") as f:
    while True:
        pos = f.tell()
        line = f.readline()
        if not line:
            break
        pid = int(line.split(b"\t", 1)[0])
        assert 0 <= pid < expected
        offset[pid] = pos
        count += 1
assert count == expected, ("unexpected corpus size", count)
np.save(D / "offsets.npy", offset)
write(D / "collection_location.json", {"path": str(collection), "count": count})
rng = np.random.default_rng(seed)
random1m = rng.choice(expected, size=1000000, replace=False)
positive = {int(d) for q in allq for d in rels[q]}
small = sorted(positive | set(map(int, random1m[:100000])))
large = sorted(positive | set(map(int, random1m)))
np.save(R / "splits/corpus_search.npy", np.asarray(small, np.int64))
np.save(R / "splits/corpus_1m.npy", np.asarray(large, np.int64))
# Materialize only shared reduced corpus; corpus indexing remains available for larger confirmation.
with collection.open("rb") as f, (D / "corpus_search.jsonl").open("w") as out:
    for pid in small:
        f.seek(int(offset[pid]))
        line = f.readline().decode("utf-8").rstrip("\n")
        d, t = line.split("\t", 1)
        assert int(d) == pid
        out.write(json.dumps({"id": d, "text": t}, ensure_ascii=False) + "\n")
verify_search_corpus()
write(
    R / "prepared.json",
    {
        "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "corpus_count": count,
        "search_count": len(small),
        "confirmation_corpus_count": len(large),
        "queries": {k: len(v) for k, v in sets.items()},
        "query_overlap": 0,
        "source_archive_sha256": ARCHIVE_SHA256,
        "search_corpus_sha256": SEARCH_CORPUS_SHA256,
        "split_hashes": {
            p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in (R / "splits").glob("*")
        },
    },
)
print((R / "prepared.json").read_text(), flush=True)
