# BENCHMARKS

> Fill these tables from **your own runs**. Do not copy course reference numbers.

## Lab 1 — Tokenizer audit
| Tokenizer | AR Fertility | EN Fertility | AR p95 | EN p95 |
|---|---:|---:|---:|---:|
| mBERT | 2.15 | 1.51 | 27 | 25 |
| XLM-R | 1.67 | 1.43 | 21 | 23 |
| CAMeLBERT | 1.41 | 2.70 | 20 | 38 |
| DistilBERT | 4.53 | 1.30 | 47 | 21 |

- Golden preprocessing: 25/ 25 passed
- PII masking recall: 60 / 60 = 100%


## Lab 2 — Step 5: Attention diagnostics
| Metric | Result |
|---|---:|
| Candidate adjacency layer | <YOUR_LAYER> |
| Candidate adjacency head | <YOUR_HEAD> |
| Candidate adjacency score | <YOUR_SCORE> |
| Candidate [SEP]-sink layer | <YOUR_LAYER> |
| Candidate [SEP]-sink head | <YOUR_HEAD> |
| Candidate [SEP]-sink score | <YOUR_SCORE> |
| PAD attention mass WITH mask | <YOUR_VALUE> |
| PAD attention mass WITHOUT mask | <YOUR_VALUE> |

PAD leak regression check: PASSED

## Lab 3 — Models
| Model | Metric | Validation | Frozen test | Train time |
|---|---|---:|---:|---:|
| TF-IDF + LinearSVC | macro-F1 | | | |
| Topic classifier | macro-F1 | | | |
| NER | entity-F1 | | | |
| QA | span/null smoke | | | |

## Lab 4 — Arabic model bake-off
| Checkpoint | macro-F1 all | Gulf | MSA | AR fertility |
|---|---:|---:|---:|---:|
| multilingual incumbent | | | | |
| Arabic dialect-aware | | | | |
| optional third model | | | | |

## Lab 5 — Search
| Configuration | recall@10 | MRR@10 | p50 latency/query |
|---|---:|---:|---:|
| bi-encoder only | | | |
| + cross-encoder rerank | | | |
| cross-lingual slice | | | |

- no-answer empty-correct: ___ / 20
- cross-lingual gap: ___

## Lab 6 — Evaluation
| Model | Aggregate macro-F1 [CI] | Gulf [CI] | Invariance pass | MFT pass |
|---|---|---|---:|---:|
| topic classifier | | | | |
| dialect-aware | | | | |

- paired comparison verdict:
- error taxonomy top categories:
- top-3 prioritised fixes:

## Lab 7 — Optimisation ladder
| Rung | p50 | p99 | quality metric / paired Δ | Artefact size |
|---|---:|---:|---|---:|
| fp32 torch @512 padded | | | | |
| fp32 torch @128 dynamic | | | | |
| ONNX fp32 @128 | | | | |
| ONNX INT8 @128 | | | | |

- HTTP p99, 16 concurrent:
- classifier quantisation decision:
- NER quantisation decision:

## Lab 3A — TF-IDF Baseline

| Model | Metric | Validation | Frozen Test |
|---|---|---:|---:|
| TF-IDF + LinearSVC | macro-F1 | 1.0000 | 1.0000 |
