"""Astra-requested supervised paired-document rows; no gradient computation."""

import numpy as np


def paired_document_rows(X, Y, weights, owners, pairs, competitor_count=1):
    K = competitor_count
    assert K in [1, 3]
    width = K + 2
    n = len(pairs)
    assert len(X) == len(Y) == len(weights) == len(owners) and len(X) >= width * n
    xx = []
    yy = []
    ww = []
    oo = []
    flags = []
    replaced = 0
    for i, p in enumerate(pairs):
        j = width * i
        assert owners[j : j + width] == [p["q"]] * width
        if not p["solved_base"]:
            assert abs(weights[j] - 1.0) < 1e-12
            for c in range(K):
                k = j + 1 + c
                assert abs(weights[k] - 1.0 / K) < 1e-12
                xx.append(X[j] - X[k])
                yy.append(Y[j] - Y[k])
                ww.append(2.0 / K)
                oo.append(p["q"])
                flags.append(True)
            replaced += K + 1
            keep = [j + K + 1]
        else:
            keep = list(range(j, j + width))
        for k in keep:
            xx.append(X[k])
            yy.append(Y[k])
            ww.append(weights[k])
            oo.append(owners[k])
            flags.append(False)
    for k in range(width * n, len(X)):
        xx.append(X[k])
        yy.append(Y[k])
        ww.append(weights[k])
        oo.append(owners[k])
        flags.append(False)
    xx, yy, ww, flags = (
        np.asarray(xx),
        np.asarray(yy),
        np.asarray(ww),
        np.asarray(flags, dtype=bool),
    )
    assert abs(float(ww.sum()) - float(np.asarray(weights).sum())) < 1e-7
    return (
        xx,
        yy,
        ww,
        oo,
        flags,
        {
            "original_rows": len(X),
            "competitors": K,
            "paired_rows": int(flags.sum()),
            "replaced_document_rows": replaced,
            "result_rows": len(xx),
            "weight_sum_before": float(np.asarray(weights).sum()),
            "weight_sum_after": float(ww.sum()),
            "paired_input_norm_quantiles": dict(
                zip(
                    ["min", "p10", "p25", "p50", "p75", "p90", "max"],
                    np.quantile(
                        np.linalg.norm(xx[flags], axis=1),
                        [0, 0.1, 0.25, 0.5, 0.75, 0.9, 1],
                    ).tolist(),
                )
            ),
            "paired_input_mean_norm": float(np.linalg.norm(xx[flags], axis=1).mean()),
            "paired_target_mean_norm": float(np.linalg.norm(yy[flags], axis=1).mean()),
            "construction": "K rows (Xpositive-Xcompetitor_j,Ypositive-Ycompetitor_j), each weight2/K, after existing independent LN-compatible target conversion. No additional repulsion scaling. All solved rows, unsolved query anchors, background anchors preserved.",
        },
    )
