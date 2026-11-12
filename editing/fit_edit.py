"""Analytic supervised pooled-activation ridge edits; no optimizer/backprop."""

from engine import *
import torch.nn.functional as F
from paired_objective import paired_document_rows


def collect_targets(m, texts, directions, weights, layer):
    block = m[0].auto_model.encoder.layer[layer]
    capt = {}
    handles = [
        block.output.dense.register_forward_pre_hook(
            lambda mod, args: capt.update(x=args[0].detach())
        ),
        block.output.LayerNorm.register_forward_pre_hook(
            lambda mod, args: capt.update(z=args[0].detach())
        ),
    ]
    g = block.output.LayerNorm.weight.detach().float()
    b = block.output.LayerNorm.bias.detach().float()
    eps = block.output.LayerNorm.eps
    xs = []
    ys = []
    forward_error = 0.0
    try:
        for start in range(0, len(texts), 64):
            features = m.tokenize(texts[start : start + 64])
            features = {k: v.to("cuda") for k, v in features.items()}
            mask = features["attention_mask"].float().unsqueeze(-1)
            with torch.inference_mode():
                m(features)
                x = capt["x"].float()
                z = capt["z"].float()
                h = F.layer_norm(z, (384,), g, b, eps)
                mu = z.mean(-1, keepdim=True)
                var = z.var(-1, unbiased=False, keepdim=True)
                null = g.abs() < 1e-6
                safe = torch.where(null, torch.ones_like(g), g)
                off = torch.tensor(
                    directions[start : start + 64], device="cuda"
                ).unsqueeze(1)
                v = torch.where(
                    null[None, None, :],
                    (z - mu) / torch.sqrt(var + 1e-12),
                    (h + off - b) / safe,
                )
                v -= v.mean(-1, keepdim=True)
                v /= torch.sqrt(v.var(-1, unbiased=False, keepdim=True) + 1e-12)
                target = mu + var.sqrt() * v
                expected = g * v * torch.sqrt(var / (var + eps)) + b
                actual = F.layer_norm(target, (384,), g, b, eps)
                err = float((actual - expected).abs().max())
                assert err < 1e-4, err
                forward_error = max(forward_error, err)
                y = target - z
                zero = torch.tensor(
                    np.linalg.norm(directions[start : start + 64], axis=1) == 0,
                    device="cuda",
                )
                y[zero] = 0
                den = mask.sum(1)
                xs.append(((x * mask).sum(1) / den).cpu().numpy())
                ys.append(((y * mask).sum(1) / den).cpu().numpy())
    finally:
        for h in handles:
            h.remove()
    return np.concatenate(xs), np.concatenate(ys), forward_error


def fit_candidate(name, source, config, random_source, seed):
    config = dict(config)
    config.setdefault("side", "document")
    config.setdefault("target_centering", "none")
    config.setdefault("holdout_gate", False)
    config.setdefault("holdout_partition", "legacy")
    config.setdefault("row_objective", "independent")
    assert config["row_objective"] in ["independent", "paired_document"]
    if config["row_objective"] == "paired_document":
        assert (
            config["side"] == "document"
            and config["competitors"] in [1, 3]
            and config["target_centering"] == "none"
        )
    assert config["layer"] in [4, 5] and config["rank"] in [4, 8, 16, 32]
    assert 0.001 <= config["eta"] <= 0.2 and 0.0001 <= config["ridge"] <= 0.1
    assert 0 <= config["competitor_push"] <= 0.5
    assert config["competitors"] in [1, 3]
    assert config["fit_queries"] in [10000, 50000]
    assert config["mechanism"] in [
        "contrastive_margin",
        "positive_attraction",
        "competitor_repulsion",
    ]
    assert config["weighting"] in ["all", "near_miss"]
    assert 1 <= config["query_anchor"] <= 16 and 1 <= config["solved_anchor"] <= 32
    assert config["side"] in ["document", "query", "both"]
    assert config["target_centering"] in ["none", "mean", "top1"]
    assert isinstance(config["holdout_gate"], bool)
    sets, queries, rels = loadsets()
    dids, texts = corpus()
    dmap = {d: i for i, d in enumerate(dids)}
    de = np.load(R / "cache_embeddings" / source / "corpus.npy")
    qids = sets["fit_expansion"][: config["fit_queries"]]
    m = model(source)
    qe = encode(m, [queries[q] for q in qids])
    hard = json.loads((R / "splits/base_hard_competitors.json").read_text())
    base_rows = json.loads((R / "results/base.fit_pool.perquery.json").read_text())
    pairs = []
    fittexts = []
    directions = []
    weights = []
    owners = []
    permutation = np.random.default_rng(20261105 + 88).permutation(len(qids))
    heldout = (
        {qids[i] for i in permutation[int(0.8 * len(qids)) :]}
        if config["holdout_gate"]
        else set()
    )
    common_order = np.random.default_rng(20261105 + 88).permutation(
        min(10000, len(qids))
    )
    common_heldout = {qids[i] for i in common_order[int(0.8 * len(common_order)) :]}
    if (
        config["holdout_gate"]
        and config["holdout_partition"] == "nested10k"
        and len(qids) > 10000
    ):
        extra = np.random.default_rng(20261105 + 88).permutation(len(qids) - 10000)
        heldout = common_heldout | {
            qids[10000 + i] for i in extra[int(0.8 * len(extra)) :]
        }

    def add(text, direction, weight, owner):
        fittexts.append(text)
        directions.append(direction)
        weights.append(weight)
        owners.append(owner)

    for i, q in enumerate(qids):
        positives = [d for d in rels[q] if d in dmap]
        p = max(positives, key=lambda d: qe[i] @ de[dmap[d]])
        cs = [d for d in hard[q]["ids"] if d not in rels[q]][: config["competitors"]]
        assert cs
        margin = float(qe[i] @ de[dmap[p]] - max(qe[i] @ de[dmap[c]] for c in cs))
        solved = base_rows[q]["mrr10"] == 1.0
        w = (
            0.0
            if solved
            else (
                float(np.exp(-max(-margin, 0) / 0.05))
                if config["weighting"] == "near_miss"
                else 1.0
            )
        )
        pres = config["solved_anchor"] if solved else 1.0
        pairs.append(
            {
                "q": q,
                "p": p,
                "competitors": cs,
                "before_margin": margin,
                "solved_base": solved,
                "fit_holdout": q in heldout,
                "common_10k_holdout": q in common_heldout,
            }
        )
        pu = de[dmap[p]]
        docactive = config["side"] in ["document", "both"]
        qactive = config["side"] in ["query", "both"]
        direction = qe[i] - np.dot(qe[i], pu) * pu
        add(
            texts[dmap[p]],
            config["eta"]
            * w
            * direction
            * (docactive and config["mechanism"] != "competitor_repulsion"),
            pres if docactive else pres * config["query_anchor"],
            q,
        )
        qdirection = (pu - np.dot(pu, qe[i]) * qe[i]) * (
            config["mechanism"] != "competitor_repulsion"
        )
        for c in cs:
            cu = de[dmap[c]]
            v = qe[i] - np.dot(qe[i], cu) * cu
            add(
                texts[dmap[c]],
                -config["eta"]
                * w
                * config["competitor_push"]
                / len(cs)
                * v
                * (docactive and config["mechanism"] != "positive_attraction"),
                (
                    pres / len(cs)
                    if docactive
                    else pres * config["query_anchor"] / len(cs)
                ),
                q,
            )
            if config["mechanism"] != "positive_attraction":
                qdirection -= (
                    config["competitor_push"]
                    / len(cs)
                    * (cu - np.dot(cu, qe[i]) * qe[i])
                )
        add(
            queries[q],
            config["eta"] * w * qdirection if qactive else np.zeros(384),
            pres if qactive else config["query_anchor"],
            q,
        )
    rng = np.random.default_rng(int(os.environ["MINILM_SEARCH_SEED"]))
    anchors = rng.choice(len(texts), min(2000, len(texts)), replace=False)
    for i in anchors:
        add(texts[i], np.zeros(384), 1.0, None)
    X, Y, err = collect_targets(
        m, fittexts, np.asarray(directions, np.float32), weights, config["layer"]
    )
    pairmask = np.zeros(len(X), dtype=bool)
    pair_audit = {"row_objective": "independent"}
    if config["row_objective"] == "paired_document":
        X, Y, weights, owners, pairmask, pair_audit = paired_document_rows(
            X, Y, weights, owners, pairs, config["competitors"]
        )
    w = np.asarray(weights, np.float64)
    X = X.astype(np.float64)
    Y = Y.astype(np.float64)
    fitmask = np.array([q not in heldout for q in owners])
    targeted = np.linalg.norm(Y, axis=1) > 1e-12
    fit_target = fitmask & targeted
    raw = Y[fit_target]
    mean = raw.mean(0)
    _, target_sv, target_vt = np.linalg.svd(raw, full_matrices=False)
    energy = float(np.square(raw).sum())
    target_spectrum = {
        "common_mode_energy_fraction": float(len(raw) * np.square(mean).sum() / energy),
        "top1_energy_fraction": float(target_sv[0] ** 2 / energy),
        "top4_energy_fraction": float(np.square(target_sv[:4]).sum() / energy),
        "centering_fitted_on": "nonzero target rows of80%training partition only; zero anchors preserved",
    }
    if config["target_centering"] == "mean":
        Y[targeted] -= mean
    elif config["target_centering"] == "top1":
        v = target_vt[0]
        Y[targeted] -= (Y[targeted] @ v)[:, None] * v[None, :]
    wf = w * fitmask
    C = (X.T * wf) @ X / wf.sum()
    ridge = config["ridge"] * np.trace(C) / len(C)
    C += ridge * np.eye(len(C))
    ev, V = np.linalg.eigh(C)
    inv = (V * (1 / np.sqrt(ev))) @ V.T
    A = ((Y * wf[:, None]).T @ X / wf.sum()) @ inv
    U, S, Vt = np.linalg.svd(A, full_matrices=False)
    rank = config["rank"]
    delta = (U[:, :rank] * S[:rank]) @ Vt[:rank] @ inv
    module = m[0].auto_model.encoder.layer[config["layer"]].output.dense
    old = module.weight.detach().float()
    source_norm = float(old.norm())
    cap = 0.02 * source_norm
    precap_norm = float(np.linalg.norm(delta))
    assert precap_norm > 0
    cap_factor = min(1, cap / precap_norm)
    delta *= cap_factor
    prediction = X @ delta.T
    fit_quality = {}
    objective_constant = float((np.square(Y) * wf[:, None]).sum() / wf.sum())
    retained_energy = float(np.square(S[:rank]).sum())
    weighted_objective = {
        "definition": "weighted mean squared residual summed across output coordinates plus ridge Frobenius penalty",
        "zero_update": objective_constant,
        "full_rank_optimum": objective_constant - float(np.square(S).sum()),
        "rank_truncated_before_cap": objective_constant - retained_energy,
        "rank_truncated_after_cap": objective_constant
        + (cap_factor**2 - 2 * cap_factor) * retained_energy,
        "truncation": "SVD of B C_ridge^(-1/2), then right unwhitening; optimal rank truncation for activation-covariance-weighted ridge objective before norm cap, not Euclidean weight SVD",
    }
    common_audit_mask = np.array([q in common_heldout for q in owners])
    for label, mask in [
        ("heldout_common_targeted", ~fitmask & targeted & common_audit_mask),
        ("heldout_extra_targeted", ~fitmask & targeted & ~common_audit_mask),
        ("fit_targeted", fitmask & targeted),
        ("fit_anchors", fitmask & ~targeted),
        ("heldout_targeted", ~fitmask & targeted),
        ("heldout_anchors", ~fitmask & ~targeted),
        ("fit_paired", fitmask & pairmask),
        ("heldout_paired", ~fitmask & pairmask),
    ]:
        if not mask.any():
            continue
        yy = Y[mask]
        pp = prediction[mask]
        den = float(np.square(yy - yy.mean(0)).sum())
        fit_quality[label] = {
            "rows": int(mask.sum()),
            "weight_sum": float(w[mask].sum()),
            "mse": float(np.square(pp - yy).mean()),
            "target_variance": float(np.var(yy, axis=0).mean()),
            "achieved_target_projection_ratio": (
                float((pp * yy).sum() / np.square(yy).sum())
                if np.square(yy).sum() > 1e-15
                else None
            ),
            "achieved_target_norm_ratio": (
                float(np.linalg.norm(pp) / np.linalg.norm(yy))
                if np.linalg.norm(yy) > 1e-15
                else None
            ),
            "r2": 1 - float(np.square(pp - yy).sum()) / den if den > 1e-12 else None,
        }
    requested = torch.tensor(delta, device="cuda", dtype=torch.float32)
    with torch.no_grad():
        module.weight.copy_(old + requested)
    real = module.weight.detach().float() - old
    m.save_pretrained(str(R / "models" / name))
    audit = {
        "name": name,
        "source": source,
        "config": config,
        "source_weight_norm": source_norm,
        "precap_norm": precap_norm,
        "precap_relative_norm": precap_norm / source_norm,
        "cap_applied": cap_factor < 1,
        "cap_scale_factor": cap_factor,
        "fit_quality": fit_quality,
        "paired_row_audit": pair_audit,
        "weighted_fitting_objective": weighted_objective,
        "target_spectrum": target_spectrum,
        "requested_rank": rank,
        "requested_norm": float(np.linalg.norm(delta)),
        "realized_norm": float(real.norm()),
        "realized_rank": int(np.linalg.matrix_rank(real.cpu().numpy())),
        "rounding_relative_residual": float(
            (real - requested).norm() / requested.norm()
        ),
        "LN_forward_max_error": err,
        "query_pool": len(qids),
        "fitted_queries": len(qids) - len(heldout),
        "heldout_queries": len(heldout),
        "fit_rows": int(fitmask.sum()),
        "ridge": ridge,
        "method": "Supervised pooled-activation ridge with optional query/document targets and target centering. Zero anchors remainzero; centering directions fit on training target rows only. Internal80/20 holdout is diagnostic and can inform adaptive search, not a final test.",
    }
    write(R / "edits" / f"{name}.json", audit)
    write(R / "edits" / f"{name}.pairs.json", pairs)
    np.save(R / "edits" / f"{name}.delta.npy", delta.astype(np.float32))
    del m, X, Y, C, A, old, real, requested, prediction
    gc.collect()
    torch.cuda.empty_cache()
    rng = np.random.default_rng(seed)
    ur = np.linalg.qr(rng.normal(size=(384, rank)))[0]
    vr = np.linalg.qr(rng.normal(size=(1536, rank)))[0]
    sv = np.linalg.svd(delta, compute_uv=False)[:rank]
    rnd = (ur * sv) @ vr.T
    rn = name.replace("target", "random")
    np.save(R / "edits" / f"{rn}.delta.npy", rnd.astype(np.float32))
    m = model(random_source)
    mod = m[0].auto_model.encoder.layer[config["layer"]].output.dense
    old = mod.weight.detach().float()
    req = torch.tensor(rnd, device="cuda", dtype=torch.float32)
    with torch.no_grad():
        mod.weight.copy_(old + req)
    real = mod.weight.detach().float() - old
    m.save_pretrained(str(R / "models" / rn))
    write(
        R / "edits" / f"{rn}.json",
        {
            "source": random_source,
            "paired_target": name,
            "config": config,
            "requested_rank": rank,
            "requested_norm": float(np.linalg.norm(rnd)),
            "realized_norm": float(real.norm()),
            "realized_rank": int(np.linalg.matrix_rank(real.cpu().numpy())),
            "rounding_relative_residual": float((real - req).norm() / req.norm()),
        },
    )
    del m
    gc.collect()
    torch.cuda.empty_cache()
    return rn


def fixed_margin_audit(name, targetname):
    pairs = [
        p
        for p in json.loads((R / "edits" / f"{targetname}.pairs.json").read_text())
        if not p.get("fit_holdout", False)
    ][:512]
    sets, qs, rels = loadsets()
    dids, texts = corpus()
    dm = {d: i for i, d in enumerate(dids)}
    de = np.load(R / "cache_embeddings" / name / "corpus.npy")
    m = model(name)
    qe = encode(m, [qs[p["q"]] for p in pairs])
    margins = []
    for i, p in enumerate(pairs):
        margins.append(
            float(
                qe[i] @ de[dm[p["p"]]]
                - max(qe[i] @ de[dm[c]] for c in p["competitors"])
            )
        )
    del m
    gc.collect()
    torch.cuda.empty_cache()
    return np.asarray(margins)
