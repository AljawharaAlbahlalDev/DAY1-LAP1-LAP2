# Decision Records

## tokenizer
For Arabic-focused Bayan tasks, CAMeLBERT is the preferred checkpoint.
Evidence:
- Arabic fertility: 1.41, the lowest among the four candidates
- Arabic p95 sequence length: 20 tokens
- This means less subword fragmentation and shorter Arabic sequences

For bilingual workloads, XLM-R is a strong balanced option:
- Arabic fertility: 1.67
- English fertility: 1.43
- p95: 21 AR / 23 EN

Decision:
- CAMeLBERT for Arabic-centric tasks
- XLM-R for balanced bilingual tasks

## arabic-model
- Incumbent:
- Candidate:
- All/Gulf/MSA evidence:
- CI-backed verdict:
- Segmentation contract:

## search-min-score
- Threshold:
- No-answer evidence:
- False-positive / false-negative trade-off:

## quantisation-split
- Topic artefact:
- NER artefact:
- Latency evidence:
- Paired quality-tax evidence:
- Rollback artefact retained:

## architecture
- Encoder/decoder rationale by task:
- Multilingual vs Arabic-centric rationale:
- Evidence used:
