# Bayan | بيان
## SDA-AIE-211 — Natural Language Processing with Transformers

> **From raw bilingual text to a working NLP service — one lab at a time.**
>
> خلال 4 أيام، هذا الـrepo بيتطور معكم من تنظيف النصوص واختيار الـTokenizer إلى Fine-tuning، Semantic Search، Evaluation، Optimisation، وأخيرًا خدمة NLP متكاملة باسم **Bayan**.

---

## 🚀 What are we building?

**Bayan (بيان)** is a bilingual citizen-feedback intelligence service for Arabic and English text.

By the end of the course, your repository should evolve into a service that can:

- 🧹 preprocess Arabic + English text consistently
- 🔐 mask PII before model use
- 🧠 classify feedback topics and sentiment
- 🏷️ extract entities with NER
- ❓ handle extractive QA with honest no-answer behaviour
- 🔎 retrieve similar historical cases using semantic search
- 📊 evaluate models with slices, confidence intervals, and behavioural tests
- ⚡ optimise inference with ONNX + INT8
- 🌐 serve the final pipeline through FastAPI

The final goal is **not seven disconnected labs**.

It is one evolving engineering project:

```text
Raw Citizen Feedback
        ↓
Versioned Preprocessing
        ↓
┌───────────────────────────────────────┐
│ Topic / Sentiment Classification      │
│ NER Entity Extraction                 │
│ Semantic Search → FAISS → Re-ranking  │
└───────────────────────────────────────┘
        ↓
Evaluation + Model Cards + Benchmarks
        ↓
Optimised FastAPI NLP Service
```

---

# 🗺️ Your 4-Day Journey

| Day | Labs | What you build | Main evidence |
|---|---|---|---|
| **Day 1** | Lab 1–2 | Preprocessing, tokenizer decision, attention understanding | Golden tests, PII recall, fertility audit, attention checks |
| **Day 2** | Lab 3 | Topic classifier, NER, QA | Baseline comparison, alignment tests, model metrics |
| **Day 3** | Lab 4–6 | Arabic-aware pipeline, semantic search, honest evaluation | Slice results, retrieval metrics, CIs, behavioural tests |
| **Day 4** | Lab 7 + Capstone | Optimised serving + full integration | p99 latency, canaries, final service, demo |

Every lab produces something that the next labs reuse.

**Do not delete yesterday's work. Build on it.**

---

# 🧩 Repository Structure

```text
SDA-AIE-211-Bayan/
│
├── src/bayan/
│   ├── preprocessing/      # Labs 1 & 4
│   ├── models/             # Lab 3
│   ├── search/             # Lab 5
│   ├── evaluation/         # Lab 6
│   └── serving/            # Lab 7 + Capstone
│
├── notebooks/
│   ├── 00_colab_setup.ipynb
│   ├── 01_tokenizer_audit.py
│   ├── 02_transformer_anatomy.py
│   └── 05_retrieval_eval.py
│
├── scripts/                # Re-runnable training/eval/benchmark scripts
├── tests/                  # Provided contracts and golden tests
├── data/                   # Course datasets and fixtures
├── artifacts/              # Models / indexes generated during the course
├── templates/              # Model-card template
├── docs/
│   ├── LABS.md
│   └── CAPSTONE_CHECKLIST.md
│
├── NOTES.md                # Observations and lab notes
├── BENCHMARKS.md           # Numbers from YOUR runs
├── DECISIONS.md            # Engineering decisions backed by evidence
├── EVALUATION_REPORT.md    # Final evaluation report
├── requirements.txt
├── pyproject.toml
└── Makefile
```

---

# ⚙️ First-Time Setup

The course targets **Python 3.12**.

## macOS / Linux

```bash
python3.12 -m venv .venv
source .venv/bin/activate

python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -e .

python scripts/doctor.py
```

## Windows PowerShell

```powershell
py -3.12 -m venv .venv
.venv\Scripts\Activate.ps1

python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -e .

python scripts/doctor.py
```

When the environment is ready, the doctor check should end with:

```text
ALL GOOD
```

If it does not, fix the environment **before** starting the lab.

---

# 🧪 How Labs Work

This repository is intentionally a **starter**, not a solution repository.

You will see:

```python
# TODO(Lab X)
raise NotImplementedError(...)
```

That is expected.

The workflow is:

```text
Read the task
   ↓
Run the provided test / script
   ↓
See the failure or baseline
   ↓
Implement the missing logic
   ↓
Run again
   ↓
Record evidence
   ↓
Commit + Push
```

The tests are part of the specification. **Do not edit expected outputs just to make a test green.**

---

# 🥇 Lab 1 — Your First Checkpoint

Lab 1 builds the bilingual preprocessing foundation used by training, evaluation, and serving later.

## Step 1 — Defect Safari

Inspect:

```text
data/raw/bayan_raw_sample.csv
```

Document at least six defect classes in:

```text
NOTES.md
```

Look for issues such as Unicode variation, tatweel, code-switching, PII, emoji, and HTML remnants.

## Step 2 — Preprocessing Golden Tests

Implement the missing preprocessing functions, then run:

```bash
pytest tests/test_preprocessing.py -q
```

Target:

```text
25 passed
```

## Step 3 — PII Masking Test

Run:

```bash
pytest tests/test_pii_recall.py -q
```

This checks the complete **60-case PII fixture**.

Target:

```text
PII recall = 100%
```

## Step 4 — Tokenizer Audit

Run:

```bash
python notebooks/01_tokenizer_audit.py
```

Record your measured results in:

```text
BENCHMARKS.md
```

Then justify your tokenizer choice in:

```text
DECISIONS.md#tokenizer
```

Your decision should be based on evidence such as:

- Arabic fertility
- English fertility
- sequence-length distribution
- p95 sequence length
- operational trade-offs

## Lab 1 checkpoint

When your evidence is complete:

```bash
git status
git add .
git commit -m "feat(preprocessing): versioned bilingual pipeline with tokenizer audit"
git push
```

---

# ✅ Lab Commands

Run only the checks for the lab you are currently working on.

```bash
make lab1
make lab2
make lab3
make lab4
make lab5
make lab6
make lab7
```

⚠️ **On Day 1, do not run the entire future test suite expecting everything to pass.**
Labs 2–7 still contain intentional TODOs.

After the full course implementation is complete:

```bash
make test
```

---

# ☁️ Google Colab & GPU

The project has **one codebase**. Colab is only another environment for running that same repository.

You do **not** create a separate Colab version of your solution.

```text
                    GitHub Repo
                        │
             ┌──────────┴──────────┐
             ↓                     ↓
         Local Laptop          Google Colab
        light / CPU work       GPU fine-tuning
```

Recommended use:

| Lab | Environment |
|---|---|
| Lab 1 | Local |
| Lab 2 | Local |
| Lab 3 | **GPU / Colab recommended** |
| Lab 4 | **GPU / Colab for fine-tuning** |
| Lab 5 | Local / course environment |
| Lab 6 | Local / course environment |
| Lab 7 | **CPU benchmark environment** |

When GPU training is needed, open:

```text
notebooks/00_colab_setup.ipynb
```

It helps you:

1. verify CUDA / GPU availability
2. clone the same GitHub repository
3. install the same dependencies
4. optionally mount Google Drive
5. persist large generated model artifacts

The **source code remains the same**.

---

# 📊 Evidence > Claims

A core rule of this course:

> **Don't tell us the model is better. Show the evidence.**

Your repository should gradually accumulate evidence in four main places:

### `NOTES.md`
What you observed, investigated, or diagnosed.

### `BENCHMARKS.md`
Numbers from **your own executions**, not copied reference values.

### `DECISIONS.md`
Engineering decisions and the evidence behind them.

Examples:

```text
Why this tokenizer?
Why this Arabic checkpoint?
Why this retrieval threshold?
Why INT8 for one model but not another?
```

### `EVALUATION_REPORT.md`
The final honest report: slices, confidence intervals, behavioural tests, errors, and limitations.

---

# 🔐 Important Engineering Rules

Throughout the course:

- never allow PII to silently enter model training or logs
- keep preprocessing consistent between train / eval / serve
- do not change frozen evaluation data to improve a score
- do not copy benchmark numbers from reference material
- do not commit secrets, access tokens, or credentials
- preserve rollback/reference artifacts when optimising models
- record meaningful commits across the four days

The goal is not simply to make code run.

The goal is to build something you can **defend as an engineer**.

---

# 🌐 Final Capstone — Bayan Service

After Labs 1–7 are complete, the project is assembled into the final service.

The target flow is:

```text
Citizen Text
     ↓
Validation
     ↓
Versioned Preprocessing
     ↓
┌───────────────┬──────────────┬──────────────────────────┐
│ Classification│ NER          │ Semantic Search          │
│ Topic/Sentiment│ Entities     │ FAISS → CE Re-ranking    │
└───────────────┴──────────────┴──────────────────────────┘
     ↓
Bilingual NLP Response
```

After everything is implemented:

```bash
make test
make serve
```

The finished repository should support a live demo that can take a bilingual request and return meaningful NLP output backed by your own evaluation evidence.

---

# 🏁 What Success Looks Like

At the end of the course, your GitHub repository should tell the story of your work even before you speak:

```text
✅ meaningful commit history
✅ green lab checks
✅ trained model artifacts / manifests
✅ tokenizer and model decisions backed by numbers
✅ semantic-search evaluation
✅ sliced evaluation + confidence intervals
✅ behavioural tests
✅ model cards
✅ optimisation ladder
✅ running Bayan API
✅ final demo
```

You are not finishing the course with seven notebooks.

You are finishing with a **bilingual NLP engineering project** you built step by step.

---

## 📚 Useful Course Files

- [`docs/LABS.md`](docs/LABS.md) — exact lab task/evidence checklist
- [`docs/CAPSTONE_CHECKLIST.md`](docs/CAPSTONE_CHECKLIST.md) — final integration checklist
- [`BENCHMARKS.md`](BENCHMARKS.md) — measured results
- [`DECISIONS.md`](DECISIONS.md) — engineering decision records
- [`EVALUATION_REPORT.md`](EVALUATION_REPORT.md) — final evaluation report
- [`data/DATA_DICTIONARY.md`](data/DATA_DICTIONARY.md) — dataset reference

---

## Ready?

```bash
python scripts/doctor.py
```

If you see:

```text
ALL GOOD
```

then you're ready to build **Bayan**. 🚀
