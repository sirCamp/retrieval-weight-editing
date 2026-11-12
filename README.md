# Retrieval Weight Editing

## Goal

Can a language model guide small weight edits that improve retrieval beyond a matched random search? We test that question on a trained MiniLM retriever, first on held-out MS MARCO queries and then on six BEIR datasets.

## Task

Given a text query, rank passages by the cosine similarity of their MiniLM embeddings. The editor sees relevance-labeled training feedback and proposes where and how to edit the retriever. For the solver-assisted searches, we compare the selected edit with the original checkpoint and an independently selected random control. A later direct-edit follow-up gives Astra and Fable a persistent numerical notebook and adaptive forward tools; its candidate-matched controls and supplemental selection status are reported separately.

## How the solver-assisted edits were made

We started from `sentence-transformers/all-MiniLM-L6-v2` at revision `1110a243`. It encodes queries and passages into 384 dimensions. [Pinned model revision](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/tree/1110a243fdf4706b3f48f1d95db1a4f5529b4d41)

1. **Propose.** Astra chose a layer, rank, update size, ridge strength, fit pool and margin objective from the available TRAIN feedback. The search could start a candidate from the base or compose it with the current incumbent.
2. **Fit.** For judged positives and retrieved unjudged competitors, the solver constructed desired embedding movements and used pooled feed-forward activations to fit a low-rank additive update to an FFN output matrix. Preservation anchors limited movement on already-solved queries. The toolkit supports independent targets and a paired document objective. Fitting used relevance labels and a closed-form ridge solve without backpropagation.
3. **Control.** Each targeted candidate had a random-direction counterpart at the same tensor, with the same requested rank and singular spectrum. Random and targeted branches each maintained an incumbent. FP16 rounding and different cumulative choices can produce different realized update norms and final edit paths.
4. **Select.** Choose the eligible checkpoint with highest MRR@10 on the 2,000 adaptive feedback queries, retaining the incumbent on ties. Both branches limited MRR loss in each query-length stratum to 0.01 on the 0–1 scale. The targeted branch also required a realized mean fit-margin gain above 1e-5 and limited solved-diagnostic MRR loss to 0.01 on that scale. The base remained eligible.

The original winner composed edits to the output projection in layer 4 (zero-based). The search repeated here used the final toolkit from the start. The original trial introduced some of those options during the search.

The [fitter](editing/fit_edit.py) implements the activation-based update and the matched random control. [Candidate replay](editing/README.md) shows how to prepare the training state and run one recorded configuration. The [checkpoint patches](patches/manifest.json) rebuild the original and all six selected weights used in the later MS MARCO and BEIR comparisons.

## Evaluation protocol

| Stage | Data and use |
|---|---|
| Fit | 10,000 MS MARCO TRAIN queries, with an allowed expansion to the fixed 50,000-query pool. Optional inner holdouts served as fit diagnostics. |
| Adaptive feedback | 2,000 separate TRAIN queries and a shared 161,594-passage corpus. This stage selected checkpoints. The reduced search corpus was constructed once, with positive passages from the original TRAIN and dev reserves. Fitting and selection use the TRAIN query judgments. |
| Mechanism diagnostics | 512 fixed queries, some drawn from the fit pool. These diagnostics informed the search. |
| MS MARCO test | The same 5,000 custom-held-out queries from MS MARCO TRAIN in every full-budget comparison, with exhaustive ranking of 8,841,823 passages. The new editors received TRAIN feedback only and all three selections were frozen before final evaluation. |
| BEIR transfer | Six fixed test datasets. Encode title and text together, exclude a document whose ID equals the query ID, and preserve the official qrels. ArguAna retains its five absent judged document IDs and all 1,406 queries. |

These are custom held-out TRAIN queries, not the official MS MARCO dev/test leaderboard split. The separately reserved query IDs are hidden from the editor, but the underlying base-model training sources may overlap.

Encoders ran in FP16 with a 256-token limit. Ranking used normalized embeddings and FP32 cosine scores. MRR@10 was the primary MS MARCO metric and macro nDCG@10 the primary BEIR metric. P@1 measures whether the first result has positive relevance, MRR@10 uses the reciprocal rank of the first relevant result within ten, and nDCG@10 uses the qrel grades.

The later search used seven rounds with candidate counts **4 / 4 / 4 / 3 / 3 / 3 / 3** per branch. The query partitions and corpus were fixed across seeds. Editor sessions and the anchor and random-direction seeds varied. Astra ran in its original agent environment for the first round of seed 701 and via AWS Bedrock afterward.

A host outage interrupted seed 702. Completed rounds and their selected weights were restored from checksummed archives, and one incomplete candidate pair was replayed from its saved configuration. The search continued on an RTX 5090 after replaying the dev baseline and checking the original corpus and test hashes. Every arm in the subsequent MS MARCO test ran on that RTX 5090. All BEIR arms ran on a single RTX 4090 with the same evaluator and package versions as the earlier BEIR check.

## Try a released checkpoint

The example uses the selected edit from seed 703. Reconstruction downloads the pinned public MiniLM weights, casts the unchanged parameters to the evaluated FP16 format, applies the exact replacement bytes, and verifies the resulting checkpoint SHA-256.

```bash
python3 -m pip install -r requirements-eval.txt
python3 scripts/restore_checkpoint.py seed703_target --output checkpoints/seed703_target
python3 examples/rank_passages.py --model checkpoints/seed703_target
```

Use `--base /path/to/model.safetensors` with the reconstruction command to reuse a local copy of the pinned public or evaluated base weights. The example ranks three illustrative passages on CPU.

### Run the benchmark evaluators

Install `requirements-eval.txt` in the BEIR environment. Extract the official BEIR archive for the dataset to evaluate, then run the evaluator on a CUDA GPU.

```bash
python3 scripts/evaluate_beir.py \
  --model checkpoints/seed703_target \
  --dataset data/beir/nfcorpus \
  --output runs/seed703_target/nfcorpus.json
```

For MS MARCO, use a separate environment with `requirements-edit.txt` and the extracted official `collectionandqueries.tar.gz`. The evaluator checks the collection, query IDs, query text and judgments against the experiment hashes. It uses about 13.6 GB of temporary disk space for one model's passage embeddings.

```bash
python3 scripts/evaluate_msmarco.py \
  --data data/msmarco \
  --model checkpoints/seed703_target \
  --output runs/seed703_target/msmarco
```

Restore `base` or any of the six named edit/control arms to repeat the same command. Use `--validate-inputs` with the MS MARCO evaluator to verify the test data on CPU. The [experiment manifest](results/experiment_manifest.json) lists the GPU and package versions used for the reported scores.

## Results

The original Astra run improved full-corpus MS MARCO MRR@10 by **0.1065 points** over random. Three later searches with the same candidate budget averaged **+0.0245 points** on the same test, with an interval that included zero. On BEIR, those three edits averaged **+0.0590 points in macro nDCG@10** over their controls.

Scores and deltas in the results tables are on a 0–100 scale. The machine-readable records use 0–1. **Bold** marks the highest displayed score in each metric column within a table, including ties; it does not indicate statistical significance.

### MS MARCO passage ranking

All rows below use the same **5,000 reserved query IDs** and rank all **8,841,823 passages**. The original run and the three subsequent searches were evaluated on different GPU stacks. Compare edit and random within each table.

**Original Astra search · 24 targeted candidates and 24 matched random candidates · Tesla V100**

| Model | P@1 | MRR@10 | nDCG@10 | Recall@100 |
|---|---:|---:|---:|---:|
| Original | 18.3800 | 29.2384 | 35.1553 | **86.2680** |
| Matched random | 18.4000 | 29.2565 | 35.1675 | 86.2547 |
| Astra edit | **18.6000** | **29.3631** | **35.2650** | 86.2647 |

The paired 95% interval for edit minus random MRR@10 was **[+0.0220, +0.1930]**. The corresponding gain over the base was **+0.1247 [+0.0448, +0.2114]**. On the smaller corpus used for adaptive feedback, the selected random had higher MRR than the selected edit. The ordering reversed on the full-corpus test.

**Three searches with the same candidate budget · 24 + 24 per seed · RTX 5090 final evaluation**

| Model | P@1 |      MRR@10 |     nDCG@10 | Recall@100 |
|---|---:|------------:|------------:|---:|
| Original | 18.3600 |     29.2239 |     35.1403 | 86.2480 |
| 701 random | 18.3200 |     29.2073 |     35.1435 | **86.2880** |
| 701 edit | 18.3400 | **29.2389** | **35.1750** | 86.2747 |
| 702 random | **18.3800** |     29.2345 |     35.1519 | 86.2480 |
| 702 edit | **18.3800** |     29.2339 |     35.1512 | **86.2880** |
| 703 random | 18.3400 |     29.2190 |     35.1375 | 86.2480 |
| 703 edit | **18.3800** | **29.2616** | **35.1998** | 86.2747 |

MRR@10 improved over the base in all three runs. Edit minus random was **+0.0316**, **−0.0006**, and **+0.0426** for seeds 701, 702, and 703. The mean was **+0.0245 [−0.0037, +0.0531]**. The corresponding mean over the base was **+0.0209 [−0.0106, +0.0541]**.

The original comparison also evaluated Opus and a joint editor. Opus was essentially flat on this test and the joint editor retained the base. Their actual search budgets differed. A separate pilot with four edit/control pairs per seed averaged **−0.0011 MRR points** versus random on a different 3,000-query test. The [score records](results/summary.json) include these runs.

### BEIR transfer

We evaluated the **same six selected checkpoints** from the three subsequent searches and the unchanged base. Each dataset uses its full corpus and official test qrels. The table gives equal weight to NFCorpus, FiQA, SciFact, ArguAna, SciDocs, and TREC-COVID.

| Model |         P@1 |      MRR@10 |     nDCG@10 |  Recall@100 |
|---|------------:|------------:|------------:|------------:|
| Original |     39.6910 |     50.8203 |     41.9970 |     58.5944 |
| 701 random |     39.7492 |     50.8225 |     41.9485 | **58.6287** |
| 701 edit | **39.8219** | **50.8776** | **42.0569** |     58.5949 |
| 702 random |     39.6910 |     50.8192 |     41.9922 |     58.6062 |
| 702 edit | **39.7029** | **50.8204** | **42.0024** | **58.6241** |
| 703 random |     39.8163 |     50.7855 |     42.0166 |     58.5896 |
| 703 edit | **40.1719** | **51.0550** | **42.0748** | **58.6317** |

| Mean edit minus control | Δ P@1 [95% CI] | Δ MRR@10 [95% CI] | Δ nDCG@10 [95% CI] |
|---|---:|---:|---:|
| Matched random | +0.1467 [-0.0234, +0.3975] | +0.1086 [+0.0029, +0.2452] | +0.0590 [+0.0156, +0.1088] |
| Original | +0.2079 [+0.0230, +0.4635] | +0.0974 [+0.0039, +0.2267] | +0.0477 [+0.0080, +0.1017] |

The nDCG@10 point estimate favored the edit over its random control in all three searches. The largest nDCG gain relative to random was **+0.1084** for seed 701. Seed 703 had the largest P@1 and MRR gains. For the solver-assisted three-seed comparison, TREC-COVID, with 50 queries, contributed about 60% of the mean nDCG gain over random under equal dataset weighting. Per-dataset scores and intervals are in the [score records](results/summary.json).

The original Astra checkpoint had a smaller BEIR point estimate against the base, **+0.0144 nDCG@10** with a paired interval of **[−0.0356, +0.0647]**. Its matched-random contrast was **+0.0383 [−0.0456, +0.1240]**. These earlier scores are recorded separately from the three new runs. [Historical BEIR scores](results/summary.json)

## Direct tensor editing with Astra and Fable 5.2

We next let Astra and Fable compute tensor replacements directly, using a retrieval adaptation of MIT's diagnose/localize/intervene/confirm workflow with a persistent CPU notebook and 50 forward calls per editor. One trajectory per editor explored 10 Astra and 19 Fable edited snapshots. Both editors ultimately submitted the base; the candidates below were preserved and evaluated as **supplemental analyses**, chosen before their final scores were observed.

On six BEIR tasks, the Astra candidate increased macro **P@1 by 0.4622 points** (39.6910 → 40.1532), MRR@10 by 0.1883 and nDCG@10 by 0.0453 versus the original. The paired 95% query-bootstrap interval for the P@1 gain is **[−0.7310, +1.6838] points**. These are observed gains with substantial uncertainty; per-dataset variation and all metric intervals are reported in the detailed record. For this direct candidate, about 72% of the P@1 macro gain comes from TREC-COVID: one additional correct top-ranked result out of 50 queries. P@1 improves on three of the six datasets. The same edit lost MRR/nDCG on both 5,000-query full-corpus MS MARCO endpoints. Fable's positive internal confirmation did not transfer to full-corpus MRR. Both edits improved P@1 on the second MS MARCO endpoint only; there is no across-the-board improvement claim.

| Checkpoint | P@1 | MRR@10 | nDCG@10 | Recall@100 |
|---|---:|---:|---:|---:|
| Original | 39.6910 | 50.8203 | 41.9970 | 58.5944 |
| Astra edit (supplemental) | **40.1532** | **51.0086** | 42.0423 | 58.6092 |
| Astra matched random | 39.7711 | 50.8290 | 41.9681 | 58.5936 |
| Fable edit (supplemental) | 39.6946 | 50.7283 | 41.8909 | 58.4052 |
| Fable matched random | 39.8074 | 50.8191 | **42.0881** | **58.6751** |

The table is the equal-weight mean of NFCorpus, FiQA, SciFact, ArguAna, SciDocs and TREC-COVID, using their full corpora and official test qrels. Values are ×100. See the [direct-editing record](results/direct_editing/README.md) for every MS MARCO metric, dataset/query counts, per-dataset BEIR tables, exact editor budgets, internal selection rules, paired intervals and limitations. Candidate-matched randoms appear in the main tables; the independently selected **best-development randoms remain fully reported in its appendix**.

The follow-up reused previously evaluated final benchmarks, hidden from the editors. It is exploratory and does not provide new untouched research-level confirmation. The existing released weight patches and replay commands cover the solver experiments; the direct follow-up currently releases scores and protocol documentation.

## Interpreting the results

- The MS MARCO test queries were reserved from editor feedback. The later comparison intentionally reused the original test after its results were known. The base model card lists MS MARCO among its training sources, so this measures same-domain adaptation. [Pinned model card](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/blob/1110a243fdf4706b3f48f1d95db1a4f5529b4d41/README.md)
- The paired 95% intervals resample the same queries for both arms. BEIR resamples within each dataset before taking the equal-dataset mean. Intervals for seed-averaged effects condition on the three fitted searches and the fixed test panel. They quantify query sampling variation and are unadjusted for the additional contrasts reported here.
- The experiments compare selected checkpoints under a specific search budget. The solver-assisted results support a small transfer effect on this BEIR panel, with more uncertainty about the mean MS MARCO advantage over random. Different serving environments and the evolving toolkit in the original trial limit comparisons between editor runs.

## Relationship to MIT's Weight-Space Programming experiment

In Experiment 8, Takehi et al. let Astra probe a math model on practice problems drawn from training sets and then evaluated its submitted edits on held-out benchmarks, including MATH-500 and GSM8K. That experiment already separated the editor's feedback from the final test. The accompanying logs describe the practice pool, forward-pass budget and post-edit scoring. [Authors' write-up, Experiment 8](https://rikiyat.github.io/blog/weight-space-programming/#make-a-language-model-better-at-math) · [Experiment logs and protocol](https://github.com/RikiyaT/lm_understand-logs/tree/4d09ba6d6a7e891dd8063974dfe81b3d78e3494f/exp8_math)

| Design choice | MIT Experiment 8 | MiniLM solver-assisted retrieval |
|---|---|---|
| What the editor does | Inspects weights and probe outputs, then writes tensor edits directly. | Chooses candidate settings from retrieval feedback. A supervised closed-form solver computes each update. |
| Search budget | A budget of 10, 25 or 50 forward passes on practice inputs. | 24 targeted and 24 matched random candidates per full-budget search. |
| Final evaluation | Held-out math and other language benchmarks, including instruction following. | Full-corpus ranking on 5,000 MS MARCO queries reserved from the editor, followed by transfer to six BEIR datasets. |

The later [direct-edit follow-up](results/direct_editing/README.md) instead offers persistent weight analysis, adaptive per-example probes and interventions across attention/FFN matrices in all six layers. Its retrieval task, editor runtime/reasoning settings, forward-tool work units, editable parameters, precision, resource caps, controls, supplemental selection and reused test benchmarks differ from MIT. The [explicit comparison table](results/direct_editing/README.md#differences-from-mit-experiment-8) lists these differences; this is an adaptation, not an exact replication.

The MiniLM comparison tests whether a selected edit carries beyond the TRAIN query feedback used to fit and select it. The shared MS MARCO passage corpus is available at retrieval time. The BEIR tests add an evaluation across different retrieval corpora and tasks.

## Audit the scores

From the repository root, with Python 3.9 or newer:

```bash
python3 scripts/verify_results.py
```

For the direct-edit follow-up, also run `python3 scripts/verify_direct_results.py`.

The original verifier checks the release file hashes, query alignment, and aggregate values for the original MS MARCO run, the three subsequent searches, and all 42 BEIR evaluations. It recomputes scores from saved query-level metrics. Use the GPU runners above to recompute rankings from the released checkpoints and benchmark data.

- [Score records](results/summary.json) contain unrounded scores, paired intervals and per-dataset BEIR results.
- [Query-level scores](results/per_query.json.gz) contain query IDs and metric values for the main comparisons.
- [Experiment manifest](results/experiment_manifest.json) records selected checkpoint hashes, test split hashes, dataset hashes and runtime versions.

Dataset documentation and evaluation references. [MS MARCO Passage Ranking](https://github.com/microsoft/MSMARCO-Passage-Ranking) and [BEIR](https://github.com/beir-cellar/beir).

## Release scope and licensing

The solver-assisted checkpoint patches, fitter replay and benchmark runners are packaged. Direct-edit results include score records and provenance, but the direct harness, transcripts and direct checkpoint patches are not yet packaged for independent replay. The two earlier restricted direct pilots are also disclosed in the [experiment history](results/direct_editing/README.md#earlier-direct-edit-pilots).

The original project code is licensed under the [Apache License 2.0](LICENSE). The checkpoint patches and model configuration files derive from [sentence-transformers/all-MiniLM-L6-v2 at the pinned revision](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/tree/1110a243fdf4706b3f48f1d95db1a4f5529b4d41), whose upstream license is [Apache 2.0](licenses/Apache-2.0.txt). Their upstream attribution and applicable terms remain in effect. Dataset terms remain those of their respective upstream releases; this code license does not relicense them.

## Citing

For these MiniLM experiments and artifacts:

```bibtex
@misc{retrieval_weight_editing_2026,
  title = {Retrieval Weight Editing: {MiniLM} Experiments and Benchmark Records},
  year = {2026},
  author = {Stefano Campese},
  note = {Code, checkpoint patches, and benchmark records}
}
```
