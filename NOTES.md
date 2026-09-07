# Lab Notes

## Lab 1 — Defect Safari
Inspect `data/raw/bayan_raw_sample.csv` and document at least six defect classes.
For each one record: example, why it matters, and clean/preserve/task-dependent.

1. Unicode variations
2. Tatweel
3. Arabic-English code-switching
4. PII: phone numbers and national IDs
5. Emoji
6. HTML remnants 


## Lab 2 — Transformer Anatomy

### Attention
- Custom scaled dot-product attention matches PyTorch at atol=1e-6.
- 4-token toy sequence produces a 4x4 attention matrix.
- Attention weights from random Q/K/V are not linguistically meaningful.

### Multi-Head Attention
- d_model = 768
- n_heads = 12
- d_k = 64
- MHA parameters = 2,362,368
- Input shape  = [1, 4, 768]
- Output shape = [1, 4, 768]

### Parameter Audit
- mBERT total = 177.85M
- mBERT embeddings = 92.21M (51.8%)
- CAMeLBERT total = 109.08M
- CAMeLBERT embeddings = 23.43M (21.5%)
- Attention and FFN sizes are nearly identical across both models.
- Main size difference comes from vocabulary / embedding table:
  multilingual tax.

### Causal Mask
- Lower-triangular attention mask passed.
- Each token can attend only to itself and previous positions.
- This is decoder-style causal attention.

### Attention Diagnostics
- Candidate adjacency heads were observed across the three Arabic sentences.
- Candidate [SEP]-sink heads were observed in deeper layers.
- Attention maps are diagnostics, not causal explanations.

### PAD Leak
- PAD attention mass WITH mask: 0.000000
- PAD attention mass WITHOUT mask: 0.253806
- Regression check passed: PAD mass with mask < 0.01.
- Missing attention masks can create silent numerical errors without crashing.

### Interesting tokenizer observation
- "بطيئًا" produced [UNK] in one example.
- Tokenisation problems happen before attention and cannot be repaired downstream.

## Lab 4 — Dialect audit
- Distribution:
- One-sentence implication for MSA-only evaluation:
