# Replay a recorded MiniLM candidate

The direct tensor-edit follow-up is documented separately in [its results and protocol](../results/direct_editing/README.md); this folder does not implement that notebook harness.

These files contain the fitting code used in the full-budget searches. The example starts from the base, fits the recorded configuration selected for seed 701, and builds its matched random counterpart. The released checkpoint patches are the route to exact evaluated bytes. Re-fitting on another runtime can change floating-point results.

Use a CUDA-capable PyTorch build and the search-side packages in [requirements-edit.txt](../requirements-edit.txt). Run the commands from the repository root. Download the official MS MARCO passage archive and verify its SHA-256 before extraction.

```bash
python3 -m pip install -r requirements-edit.txt
export MINILM_EXPERIMENT_ROOT="$PWD/runs/replay701"
mkdir -p "$MINILM_EXPERIMENT_ROOT/data"

curl -fL https://msmarco.z22.web.core.windows.net/msmarcoranking/collectionandqueries.tar.gz -o collectionandqueries.tar.gz
printf '%s  %s\n' decac356eb8cc5b9cea2e30b8738dc6f367e4147aaefe8fc7526ddda382fd2fc collectionandqueries.tar.gz | sha256sum -c -
sha256sum collectionandqueries.tar.gz > "$MINILM_EXPERIMENT_ROOT/data/archive.sha256"
tar -xzf collectionandqueries.tar.gz -C "$MINILM_EXPERIMENT_ROOT/data"

python3 scripts/restore_checkpoint.py base --output "$MINILM_EXPERIMENT_ROOT/models/base"
python3 editing/prepare_train.py
python3 editing/build_baseline.py
MINILM_SEARCH_SEED=701 python3 editing/fit_worker.py replay_701_target base base examples/candidate_701.json 7010702
```

The training preparation step checks the original 161,594-passage search corpus hash. The baseline step builds embeddings, hard-competitor lists and ranking feedback for the TRAIN query pools. The fitter saves `replay_701_target` and `replay_701_random` under `models/`, with fit settings and requested updates under `edits/`.

The example configuration is the saved decision for the third candidate of round 7 in seed 701. It uses paired document rows, a rank-32 update to the FFN output in zero-based layer 5, and the original 10,000-query fit pool. The search kept the fit, feedback and final test data separate. Use the benchmark evaluators after choosing and freezing a checkpoint.
