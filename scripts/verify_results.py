"""Verify the score release from saved query-level metrics."""

from pathlib import Path
import gzip
import hashlib
import json
import math

ROOT = Path(__file__).resolve().parents[1]
METRICS = ("p1", "mrr10", "ndcg10", "recall100")
SEEDS = (701, 702, 703)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def close(actual, expected, label):
    require(
        math.isclose(actual, expected, rel_tol=0.0, abs_tol=1e-12),
        f"{label}: computed {actual}, recorded {expected}",
    )


def verify_scores(rows, expected, query_ids, label):
    require(set(rows) == query_ids, f"{label}: query IDs differ")
    for values in rows.values():
        require(len(values) == len(METRICS), f"{label}: wrong metric count")
        require(
            all(math.isfinite(v) and -1e-12 <= v <= 1 + 1e-12 for v in values),
            f"{label}: score outside the valid range",
        )
    for j, metric in enumerate(METRICS):
        mean = math.fsum(values[j] for values in rows.values()) / len(rows)
        close(mean, expected[metric], f"{label}/{metric}")


def main():
    for line in (ROOT / "SHA256SUMS").read_text().splitlines():
        expected, filename = line.split("  ", 1)
        actual = hashlib.sha256((ROOT / filename).read_bytes()).hexdigest()
        require(actual == expected, f"File hash mismatch: {filename}")

    summary = json.loads((ROOT / "results/summary.json").read_text())
    with gzip.open(
        ROOT / "results/per_query.json.gz", "rt", encoding="utf-8"
    ) as stream:
        per_query = json.load(stream)
    require(
        tuple(summary["metric_order"]) == tuple(per_query["metric_order"]) == METRICS,
        "Metric order differs",
    )

    query_ids = set(per_query["msmarco_original"]["base"])
    require(len(query_ids) == 5000, "Expected 5,000 MS MARCO test queries")
    for panel in ("msmarco_original", "msmarco_matched_budget"):
        recorded = summary[panel]
        require(recorded["corpus"] == 8841823, "MS MARCO corpus size differs")
        require(
            set(per_query[panel]) == set(recorded["scores"]),
            f"{panel}: arm set differs",
        )
        for arm, scores in recorded["scores"].items():
            verify_scores(per_query[panel][arm], scores, query_ids, f"{panel}/{arm}")

    beir = summary["beir_matched_budget"]
    tasks = sorted(beir["scores"]["base"])
    require(
        len(tasks) == 6 and len(beir["scores"]) == 7, "Expected 42 BEIR evaluations"
    )
    for arm, scores in beir["scores"].items():
        require(
            set(scores) == set(per_query["beir_matched_budget"][arm]) == set(tasks),
            f"{arm}: BEIR dataset set differs",
        )
        for task in tasks:
            rows = per_query["beir_matched_budget"][arm][task]
            ids = set(per_query["beir_matched_budget"]["base"][task])
            require(
                len(rows) == scores[task]["queries"],
                f"{arm}/{task}: query count differs",
            )
            verify_scores(rows, scores[task], ids, f"BEIR/{arm}/{task}")
        for metric in METRICS:
            macro = math.fsum(scores[t][metric] for t in tasks) / len(tasks)
            close(macro, beir["macro"][arm][metric], f"BEIR/{arm}/macro/{metric}")

    for reference in ("base", "random"):
        for metric in METRICS:
            deltas = []
            for seed in SEEDS:
                target = f"seed{seed}_target"
                control = "base" if reference == "base" else f"seed{seed}_random"
                delta = beir["macro"][target][metric] - beir["macro"][control][metric]
                recorded = beir["paired_intervals"][f"{target}_minus_{reference}"][
                    "macro"
                ][metric]["delta"]
                close(delta, recorded, f"BEIR/{target}/{reference}/{metric}")
                deltas.append(delta)
            recorded_mean = beir["paired_intervals"][f"mean_target_minus_{reference}"][
                "macro"
            ][metric]["delta"]
            close(
                math.fsum(deltas) / len(SEEDS),
                recorded_mean,
                f"BEIR/mean/{reference}/{metric}",
            )

    matched = summary["msmarco_matched_budget"]
    for reference in ("base", "random"):
        deltas = []
        for seed in SEEDS:
            target = f"seed{seed}_target"
            control = "base" if reference == "base" else f"seed{seed}_random"
            delta = (
                matched["scores"][target]["mrr10"] - matched["scores"][control]["mrr10"]
            )
            close(
                delta,
                matched["paired_mrr_intervals"][f"{target}_minus_{reference}"]["delta"],
                f"MS MARCO/{target}/{reference}/MRR@10",
            )
            deltas.append(delta)
        close(
            math.fsum(deltas) / len(SEEDS),
            matched["paired_mrr_intervals"][f"mean_target_minus_{reference}"]["delta"],
            f"MS MARCO/mean/{reference}/MRR@10",
        )

    print(
        "Verified release hashes, all 56 evaluations, query alignment, macro scores, and paired point estimates."
    )


if __name__ == "__main__":
    main()
