"""Verify direct-edit aggregates against saved per-query metrics (no GPU required)."""
from pathlib import Path
import gzip
import hashlib
import json
import math

ROOT = Path(__file__).resolve().parents[1]
METRICS = ('p1', 'mrr10', 'ndcg10', 'recall100')
ARMS = {'base', 'astra_rejected', 'astra_rejected_random', 'astra_best_random',
        'fable_rejected', 'fable_rejected_random', 'fable_best_random'}
DATASETS = {'nfcorpus': (323, 3633), 'fiqa': (648, 57638),
            'scifact': (300, 5183), 'arguana': (1406, 8674),
            'scidocs': (1000, 25657), 'trec-covid': (50, 171332)}

def close(a, b):
    assert math.isclose(a, b, rel_tol=0, abs_tol=1e-12), (a, b)

def panel(arms, summary, count):
    assert set(arms) == set(summary) == ARMS
    ids = set(arms['base']['perquery'])
    assert len(ids) == count
    for name, arm in arms.items():
        assert set(arm['perquery']) == ids
        for row in arm['perquery'].values():
            assert set(row) == set(METRICS)
            assert all(math.isfinite(v) and 0 <= v <= 1+1e-12 for v in row.values())
        for m in METRICS:
            actual = math.fsum(row[m] for row in arm['perquery'].values()) / count
            close(actual, arm['scores'][m])
            close(actual, summary[name][m])
    return ids

def main():
    for line in (ROOT/'SHA256SUMS').read_text().splitlines():
        digest, name = line.split('  ', 1)
        assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest() == digest, name
    folder = ROOT/'results/direct_editing'
    with gzip.open(folder/'per_query.json.gz', 'rt') as f:
        data = json.load(f)
    summary = json.loads((folder/'summary.json').read_text())
    manifest = json.loads((folder/'manifest.json').read_text())
    assert manifest['checkpoint_sha256'] == summary['checkpoint_sha256']
    assert manifest['beir_sizes'] == data['datasets']
    seen = set()
    assert set(data['msmarco']) == {'fresh_msmarco', 'historical_msmarco'}
    for split, arms in data['msmarco'].items():
        assert all(a['queries'] == 5000 and a['corpus'] == 8841823 for a in arms.values())
        ids = panel(arms, summary['msmarco'][split], 5000)
        idfile = folder/(split+'_query_ids.json')
        assert set(json.loads(idfile.read_text())) == ids
        assert hashlib.sha256(idfile.read_bytes()).hexdigest() == manifest['test_splits'][split]['file_sha256']['query_ids.json']
        assert not seen.intersection(ids)
        seen.update(ids)
    assert set(data['beir']) == set(DATASETS)
    for task, (queries, corpus) in DATASETS.items():
        assert data['datasets'][task] == {'queries': queries, 'corpus': corpus}
        panel(data['beir'][task], summary['beir'][task], queries)
    for arm in ARMS:
        for m in METRICS:
            close(sum(summary['beir'][t][arm][m] for t in DATASETS)/6,
                  summary['beir_macro'][arm][m])
    for editor in ('astra', 'fable'):
        target = editor+'_rejected'
        for control in ('base', target+'_random'):
            for m in METRICS:
                close(summary['beir_macro'][target][m]-summary['beir_macro'][control][m],
                      summary['paired_beir'][target+'_minus_'+control][m]['delta'])
    assert summary['checkpoint_sha256'] == data['checkpoint_sha256']
    assert set(summary['checkpoint_sha256']) == ARMS
    assert all(len(v) == 64 and all(c in '0123456789abcdef' for c in v)
               for v in summary['checkpoint_sha256'].values())
    print('Verified 56 direct-edit evaluations, query alignment, macro means, contrast point estimates and release hashes.')
    print('Bootstrap intervals and original model inference are not recomputed by this check.')

if __name__ == '__main__':
    main()
