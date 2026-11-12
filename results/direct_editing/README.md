# Direct tensor editing with Astra and Fable 5.2

## Scope and selection

This follow-up adapts the diagnostic workflow of MIT Experiment 8 to `sentence-transformers/all-MiniLM-L6-v2`: diagnose retrieval errors, inspect internal representations, localize a mechanism, intervene on weights, and confirm on separate practice examples. It is not an exact replication of the generative math experiment. It is separate from the solver-assisted searches in the main README.

There was **one search trajectory per editor**. Both editors submitted the unchanged base model under their own confirmation rules. The edited checkpoints below are **supplemental candidates**, requested for final evaluation after internal rejection and before their full-corpus/BEIR scores were observed. Their results do not replace the recorded original submissions. All candidates were preserved; small positive effects are reported alongside uncertainty and regressions.

## Differences from MIT Experiment 8

We cite Takehi, Teoh, Zhang and Khattab, *Sparks of Weight-Space Programming: LLMs can natively read & write neural network weights* (2 November 2026). The referenced publication is the authors' technical article and released experiment logs; no journal venue, DOI or peer-review status is asserted. [Article](https://rikiyat.github.io/blog/weight-space-programming/#make-a-language-model-better-at-math) · [Pinned experiment-8 protocol/logs](https://github.com/RikiyaT/lm_understand-logs/tree/4d09ba6d6a7e891dd8063974dfe81b3d78e3494f/exp8_math) · [BibTeX](../../CITATIONS.bib)

| Aspect | MIT Experiment 8 | This direct-edit follow-up |
|---|---|---|
| Task/model | Generative math, Qwen3.5-0.8B | Dense retrieval, MiniLM-L6-v2; query and passage encoder share weights |
| Editor | Astra via Codex CLI with maximum reasoning effort | Astra and Fable via Bedrock, provider-default reasoning; one trajectory each |
| Practice | GSM8K/MATH train and AIME practice | MS MARCO TRAIN-derived practice and authored diagnostic variants |
| Forward tools | Generation or teacher-forced internal probes | Exhaustive development retrieval or representation/attention inspection |
| Budget | 10/25/50 forward passes; up to 300 Python runs | 50 forward calls per editor; up to 300 Python calls; a call is a different unit of work |
| Kernel limits | Persistent CPU Python, up to 900 seconds and 96 GB per execution | Persistent CPU NumPy/SciPy, 180 seconds and 12 GB per execution |
| Edits | Direct tensor writes; excludes embedding/output layer, final norm and whole-tensor scaling | Direct attention/FFN matrix writes; additionally fixes every norm and bias and caps cumulative per-tensor changes at 25% |
| Selection | Editor chooses final weights after practice | Same principle; both chose base, then rejected candidates received explicitly supplemental tests |
| Controls | The reported math comparisons use the unedited model | Adds candidate-matched and best-development random controls |
| Final evaluation | Hidden math and other language benchmarks | Two custom MS MARCO TRAIN-derived holdouts and six BEIR tasks; previously evaluated in earlier project runs |

The solver-assisted experiments in the main README differ more fundamentally: the editor chooses configuration parameters and a supervised ridge/SVD fitter computes the update. Absence of backpropagation does not make that method fitting-free. MIT already separated editor practice from final testing; this project does not claim to introduce that separation. Numerical write precision also differs (MIT's evaluated checkpoint uses BF16; this project's encoders use FP16).

## Editors and tools

| Item | Astra | Fable 5.2 |
|---|---|---|
| Bedrock model ID | `us.openai.gpt-6-astra` | `us.anthropic.claude-fable-5-2` |
| Successful forward calls | 50 | 50 |
| Python calls | 37 | 116 |
| Distinct edited snapshots observed through forward tools | 10 | 19 |
| Internal candidate confirmation | 1,000 reserved practice queries | 2,000 reserved practice queries |
| Submitted checkpoint | Original | Original |

Both received a persistent CPU Python notebook with NumPy/SciPy and the same tools and caps: 50 successful forward calls, up to 300 Python calls, 400 conversation turns, 12,000 maximum response tokens per API call, and provider-default reasoning settings. These are equal limits, not equal realized compute. Retrieval calls ranked up to 512 selected practice queries over the complete 161,594-passage development corpus. Inspection calls exposed per-example or token-level FFN inputs, attention output-projection inputs, block outputs, normalized embeddings and optionally attention probabilities. The latter used eager attention, which can differ slightly numerically from default FP16 inference.

The editor could modify attention and FFN matrices across all six layers, including multiple matrices. Embeddings, LayerNorm, biases, architecture, tokenizer, pooling and inference settings stayed fixed. Each tensor had a cumulative relative Frobenius-change cap of 25%; whole-tensor scalar scaling was excluded. No gradients, optimizers, supervised regression or iterative loss fitting were permitted. Syntax guards are not a proof of semantic compliance; full transcripts need audit before claiming independently certified absence of training.

The kernel had no network or GPU access, ran as a non-root user with capabilities dropped and read-only model/practice mounts, and could not access final data, credentials, or the other editor's run. GPU forward calls ran in a separate trusted process. A parser failure interrupted Astra after 19 forward calls; its CPU state was reconstructed from recorded Python actions, every historical edited forward snapshot was matched, and the current patch bytes were identical. No GPU forward calls were replayed and budgets were preserved. The parser correction also applied to Fable.

## Data and separation

| Stage | Data | Role |
|---|---|---|
| Practice pool | MS MARCO TRAIN; a fixed 10,000-query pool with an allowed 50,000-query expansion pool | Available for diagnosis; the 10,000-query pool is a subset of the 50,000-query pool, not 60,000 independent examples |
| Development feedback | 2,000 separate TRAIN queries | Available to the editors; also used after editor freeze to select the best random control |
| Diagnostics | 512 fixed queries, all contained in the 10,000-query fit pool | Available for mechanistic investigation |
| Development corpus | 161,594 fixed passages | Exhaustive practice ranking; originally constructed using positive passages from TRAIN and dev reserves, not independent unseen text |
| Historical MS MARCO endpoint | 5,000 reserved query IDs | Same historical test as prior solver experiments |
| Second MS MARCO endpoint (`fresh_msmarco` in files) | A separate 5,000 reserved query IDs | The name is historical: it had already been evaluated in earlier direct-edit pilots |
| Final passage corpus | All 8,841,823 MS MARCO v1 passages | Same full corpus for both endpoints and all checkpoints |
| BEIR | Six fixed official test datasets, below | Transfer evaluation; no BEIR examples or feedback supplied to editors |

The practice query dictionary contains 52,000 unique queries: 50,000 in the expansion pool and 2,000 disjoint feedback queries. Editors also authored diagnostic text variants, including sentence reorderings, to test mechanisms; these were not additional external training corpora.

Both 5,000-query endpoints are custom holdouts drawn from the official **TRAIN** query/qrel files. They are not the official MS MARCO dev/test leaderboard splits, so the scores should not be compared to leaderboard scores as if the evaluation queries were identical.

The harness checked query-ID and normalized-text separation between practice and both final endpoints, and between the final endpoints. Internal confirmation sets were chosen by each editor from practice, with their construction and prior use recorded in its logs. Both final MS MARCO endpoints and this BEIR panel had been evaluated in earlier experiments and informed this follow-up. They were hidden from these editors but are **not untouched research-level confirmation sets**. Base-model pretraining exposure is not ruled out. No external training corpus was added.

| BEIR dataset | Test queries | Corpus documents |
|---|---:|---:|
| nfcorpus | 323 | 3,633 |
| fiqa | 648 | 57,638 |
| scifact | 300 | 5,183 |
| arguana | 1,406 | 8,674 |
| scidocs | 1,000 | 25,657 |
| trec-covid | 50 | 171,332 |

BEIR uses each complete corpus and official test qrels, concatenates title and text, excludes self-ID matches and retains all official query denominators. ArguAna includes five judged document IDs absent from its corpus and all 1,406 test queries. Encoders use FP16 and at most 256 tokens; ranking uses normalized embeddings and FP32 cosine similarity. MS MARCO runs on RTX 5090 and BEIR on RTX 4090. Metrics are P@1, MRR@10, nDCG@10 and Recall@100. Macro BEIR means give each dataset equal weight, not each query; the 50-query TREC-COVID task therefore has the same weight as the 1,406-query ArguAna task.

Dataset references: [MS MARCO Passage Ranking](https://github.com/microsoft/MSMARCO-Passage-Ranking), [BEIR](https://github.com/beir-cellar/beir).

## Candidate mechanisms and internal confirmation

- **Astra:** rank-one attenuation of a CLS-versus-content direction in the final-layer attention key matrix, at dose 0.30 and about 2.14% relative Frobenius change. Its 1,000-query confirmation yielded MRR@10 77.3898 → 77.3288 (delta −0.0610 points, paired SE 0.2385). Its predeclared nonpositive-MRR rule restored the base.
- **Fable:** removal of a position-related direction from the final-layer key matrix, approximately `W(I − d dᵀ)`, with about 8.2% relative Frobenius change. On four reserved groups totaling 2,000 queries it gained +0.1190 MRR points, +0.3500 P@1, +0.0488 nDCG and lost 0.0625 Recall@100. MRR improved in three of four groups, but failed its predeclared +2.5-SE and nondecreasing-recall requirements. It restored the base. Local activation effects and ranking improvements are different claims.

## Controls

**Matched random** applies independent signed row/column permutations to each cumulative candidate delta from the original, on the same tensors. This preserves its singular spectrum, Frobenius norm and nonzero-entry count before FP16 rounding, while changing direction/support locations. It is not Haar sampling or a complete mechanistic null. One counterpart is reported for each supplemental candidate.

**Best-development random** is the highest-MRR random checkpoint on the fixed 2,000-query feedback set, including base fallback, across 10 Astra or 19 Fable snapshots. Its selection rule differs from the editor's final decision. It is an additional random-search baseline, not the matched control for the chosen candidate. It is selected without test scores and is fully reported in the appendix rather than omitted.

## Final results

All values below are ×100; JSON values use 0–1. Tables report point estimates, not declarations of statistical significance. **Bold** marks the highest displayed score in each metric column within a table, including ties.

### MS MARCO: second endpoint, 5,000 queries

| Checkpoint | P@1 | MRR@10 | nDCG@10 | Recall@100 |
|---|---:|---:|---:|---:|
| Original | 17.4000 | 28.7556 | 34.9104 | **85.9467** |
| Astra edit (supplemental) | **17.5200** | 28.6420 | 34.7475 | 85.7800 |
| Astra matched random | 17.5000 | 28.7693 | 34.8937 | 85.8867 |
| Fable edit (supplemental) | 17.4400 | 28.6140 | 34.7066 | 85.5600 |
| Fable matched random | 17.4600 | **28.7937** | **34.9477** | 85.9200 |

### MS MARCO: historical endpoint, 5,000 queries

| Checkpoint | P@1 | MRR@10 | nDCG@10 | Recall@100 |
|---|---:|---:|---:|---:|
| Original | 18.3600 | 29.2239 | 35.1403 | 86.2480 |
| Astra edit (supplemental) | 18.1800 | 28.9928 | 34.9506 | 85.9780 |
| Astra matched random | 18.3600 | 29.2083 | 35.1315 | 86.2847 |
| Fable edit (supplemental) | 17.9600 | 28.9361 | 34.9508 | 85.9880 |
| Fable matched random | **18.4800** | **29.2558** | **35.1731** | **86.2980** |

### BEIR: equal-dataset macro mean

| Checkpoint | P@1 | MRR@10 | nDCG@10 | Recall@100 |
|---|---:|---:|---:|---:|
| Original | 39.6910 | 50.8203 | 41.9970 | 58.5944 |
| Astra edit (supplemental) | **40.1532** | **51.0086** | 42.0423 | 58.6092 |
| Astra matched random | 39.7711 | 50.8290 | 41.9681 | 58.5936 |
| Fable edit (supplemental) | 39.6946 | 50.7283 | 41.8909 | 58.4052 |
| Fable matched random | 39.8074 | 50.8191 | **42.0881** | **58.6751** |

Astra improves all four BEIR macro point estimates over the base: +0.4622 P@1, +0.1883 MRR, +0.0453 nDCG and +0.0148 Recall@100. TREC-COVID contributes +0.3333 of the +0.4622 macro P@1 points (about 72.1%): its P@1 rises from 62% to 64%, one additional correct top-ranked result among 50 queries. P@1 improves on three of six datasets, is unchanged on one and falls on two. These gains coexist with MRR/nDCG losses on both full-corpus MS MARCO endpoints. Both edits improve P@1 on the second MS MARCO endpoint, but neither improves all metrics or both endpoints. Fable also loses macro MRR/nDCG/recall on BEIR.

### BEIR paired uncertainty

| Contrast | Δ P@1 [95% CI] | Δ MRR@10 [95% CI] | Δ nDCG@10 [95% CI] | Δ Recall@100 [95% CI] |
|---|---:|---:|---:|---:|
| Astra edit (supplemental) minus Original | +0.4622 [-0.7310, +1.6838] | +0.1883 [-0.4779, +0.8416] | +0.0453 [-0.2111, +0.3017] | +0.0148 [-0.1768, +0.2087] |
| Astra edit (supplemental) minus Astra matched random | +0.3821 [-0.8054, +1.6309] | +0.1795 [-0.4530, +0.8606] | +0.0743 [-0.1831, +0.3200] | +0.0155 [-0.1736, +0.2242] |
| Fable edit (supplemental) minus Original | +0.0036 [-1.0922, +1.1022] | -0.0920 [-0.6815, +0.4714] | -0.1061 [-0.3632, +0.1470] | -0.1891 [-0.3802, -0.0195] |
| Fable edit (supplemental) minus Fable matched random | -0.1128 [-1.0803, +0.8453] | -0.0909 [-0.6394, +0.4515] | -0.1972 [-0.4483, +0.0657] | -0.2699 [-0.4692, -0.0928] |

Intervals use 2,000 paired query-bootstrap draws within each dataset before the equal-dataset mean. They condition on the fixed six tasks and one realized trajectory per editor, are unadjusted for multiple contrasts, and do not estimate editor-run variability. A positive point estimate remains a measured gain even when its interval includes zero; that interval limits how confidently the effect can be generalized.

## Per-dataset results

### nfcorpus

| Checkpoint | P@1 | MRR@10 | nDCG@10 | Recall@100 |
|---|---:|---:|---:|---:|
| Original | 41.4861 | 50.4042 | 31.5950 | 31.1517 |
| Astra edit (supplemental) | 41.1765 | 50.3802 | 31.4662 | 31.1628 |
| Astra matched random | **41.7957** | **50.6253** | **31.6176** | 31.1998 |
| Fable edit (supplemental) | 40.5573 | 50.0720 | 31.3604 | 30.7085 |
| Fable matched random | 40.8669 | 50.1145 | 31.4733 | **31.2560** |

### fiqa

| Checkpoint | P@1 | MRR@10 | nDCG@10 | Recall@100 |
|---|---:|---:|---:|---:|
| Original | 34.7222 | 44.5122 | 36.8652 | 70.6571 |
| Astra edit (supplemental) | **35.4938** | **44.7058** | **37.0833** | **71.1023** |
| Astra matched random | 34.7222 | 44.4187 | 36.8225 | 70.9658 |
| Fable edit (supplemental) | 34.2593 | 44.1115 | 36.5651 | 70.9219 |
| Fable matched random | 35.0309 | 44.6198 | 36.9937 | 70.9878 |

### scifact

| Checkpoint | P@1 | MRR@10 | nDCG@10 | Recall@100 |
|---|---:|---:|---:|---:|
| Original | 50.3333 | 60.4725 | 64.5082 | **92.5000** |
| Astra edit (supplemental) | 51.0000 | 61.0427 | **65.2750** | 92.3333 |
| Astra matched random | 50.3333 | 60.4733 | 64.5124 | 92.1667 |
| Fable edit (supplemental) | **51.3333** | **61.1446** | 65.0583 | 92.1667 |
| Fable matched random | 51.0000 | 60.6275 | 64.7869 | **92.5000** |

### arguana

| Checkpoint | P@1 | MRR@10 | nDCG@10 | Recall@100 |
|---|---:|---:|---:|---:|
| Original | 25.6046 | 41.1527 | 50.1417 | 97.7240 |
| Astra edit (supplemental) | 25.2489 | 41.0109 | 50.1127 | **97.9374** |
| Astra matched random | 25.6757 | 41.1253 | 50.1146 | 97.7240 |
| Fable edit (supplemental) | **25.8179** | **41.4626** | **50.4487** | 97.8663 |
| Fable matched random | 25.7468 | 41.2471 | 50.2445 | 97.7240 |

### scidocs

| Checkpoint | P@1 | MRR@10 | nDCG@10 | Recall@100 |
|---|---:|---:|---:|---:|
| Original | 24.0000 | 35.9420 | **21.6449** | 50.9917 |
| Astra edit (supplemental) | 24.0000 | 35.9896 | 21.5790 | 50.6067 |
| Astra matched random | 24.1000 | 35.8935 | 21.6187 | 50.9517 |
| Fable edit (supplemental) | **24.2000** | 35.7045 | 21.4013 | 50.5267 |
| Fable matched random | **24.2000** | **36.0123** | 21.6307 | **51.0767** |

### trec-covid

| Checkpoint | P@1 | MRR@10 | nDCG@10 | Recall@100 |
|---|---:|---:|---:|---:|
| Original | 62.0000 | 72.4381 | 47.2271 | 8.5417 |
| Astra edit (supplemental) | **64.0000** | **72.9222** | 46.7376 | 8.5125 |
| Astra matched random | 62.0000 | 72.4381 | 47.1225 | **8.5539** |
| Fable edit (supplemental) | 62.0000 | 71.8746 | 46.5119 | 8.2414 |
| Fable matched random | 62.0000 | 72.2937 | **47.3995** | 8.5063 |

## Appendix: best-development random controls

These additional controls remain part of the record even when they outperform a targeted edit. Their search budgets and selection rule are described above. Main tables use the candidate-matched controls for clarity.

### MS MARCO second endpoint

| Checkpoint | P@1 | MRR@10 | nDCG@10 | Recall@100 |
|---|---:|---:|---:|---:|
| Original | 17.4000 | 28.7556 | 34.9104 | **85.9467** |
| Astra edit (supplemental) | 17.5200 | 28.6420 | 34.7475 | 85.7800 |
| Astra matched random | 17.5000 | 28.7693 | 34.8937 | 85.8867 |
| Astra best-development random | **17.5600** | **28.8164** | **34.9479** | 85.7467 |
| Fable edit (supplemental) | 17.4400 | 28.6140 | 34.7066 | 85.5600 |
| Fable matched random | 17.4600 | 28.7937 | 34.9477 | 85.9200 |
| Fable best-development random | 17.5200 | 28.7659 | 34.8597 | 85.7467 |

### MS MARCO historical endpoint

| Checkpoint | P@1 | MRR@10 | nDCG@10 | Recall@100 |
|---|---:|---:|---:|---:|
| Original | 18.3600 | 29.2239 | 35.1403 | 86.2480 |
| Astra edit (supplemental) | 18.1800 | 28.9928 | 34.9506 | 85.9780 |
| Astra matched random | 18.3600 | 29.2083 | 35.1315 | 86.2847 |
| Astra best-development random | 18.3200 | 29.2598 | **35.2433** | **86.2980** |
| Fable edit (supplemental) | 17.9600 | 28.9361 | 34.9508 | 85.9880 |
| Fable matched random | 18.4800 | 29.2558 | 35.1731 | **86.2980** |
| Fable best-development random | **18.6200** | **29.3026** | 35.2158 | 85.9147 |

### BEIR macro

| Checkpoint | P@1 | MRR@10 | nDCG@10 | Recall@100 |
|---|---:|---:|---:|---:|
| Original | 39.6910 | 50.8203 | 41.9970 | 58.5944 |
| Astra edit (supplemental) | **40.1532** | **51.0086** | 42.0423 | 58.6092 |
| Astra matched random | 39.7711 | 50.8290 | 41.9681 | 58.5936 |
| Astra best-development random | 39.8571 | 50.8565 | 42.0669 | 58.6019 |
| Fable edit (supplemental) | 39.6946 | 50.7283 | 41.8909 | 58.4052 |
| Fable matched random | 39.8074 | 50.8191 | **42.0881** | **58.6751** |
| Fable best-development random | 40.1133 | 50.9887 | 41.8595 | 58.6256 |

Per-dataset scores for all seven arms, including best-development randoms, are in [summary.json](summary.json); query-level scores are in [per_query.json.gz](per_query.json.gz). Checkpoint hashes are in the summary. The [manifest](manifest.json) records final query/qrel/corpus hashes, custom split provenance and primary versus supplemental selections. The original ordered query-ID files for [the historical endpoint](historical_msmarco_query_ids.json) and [the second endpoint](fresh_msmarco_query_ids.json) are included and checked against their frozen hashes. Run `python3 scripts/verify_direct_results.py` from the repository root to verify every recorded aggregate and macro mean. The verifier does not rerun model inference or recompute bootstrap intervals.

## Earlier direct-edit pilots

Two restricted direct-edit pilots preceded this mechanistic follow-up and informed its design. Both used 24 targeted and 24 random candidates, aggregate activation probes, a 12-assignment numerical language and only layer-4/5 FFN output matrices. The second changed the generalization prompt; the first also included a solver branch. Neither established a full-corpus direct-edit improvement. These are different protocols, not additional replicates of the mechanistic run. Their [complete aggregate score records](prior_pilots.json) remain available, including controls and the solver comparison; their per-query data are not included in this release's verifier.

| Pilot/direct candidate | MRR second endpoint | MRR historical endpoint | BEIR macro nDCG@10 |
|---|---:|---:|---:|
| restricted_direct_v1 | 28.3154 | 28.8325 | 41.4688 |
| generalization_prompt_v2 | 28.4173 | 29.0206 | 41.7622 |

Values ×100. These pilots reused the same endpoints later used here, which is another reason to treat the new experiments as exploratory.

## Release scope

This addition releases the direct-edit score records and protocol description. The existing checkpoint restoration command and `editing/` fitter reproduce the earlier solver-assisted experiments; they do not launch this new mechanistic harness or restore these supplemental direct edits. Direct-search orchestration, raw editor transcripts and exact direct checkpoint patches are retained in the experiment archive and are not yet packaged as portable release artifacts. No such reproducibility claim is made here.
