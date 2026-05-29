# IRT Parameter Drift Across LLM Generations and Architectures on Open LLM Leaderboard v2

CS321M Spring 2026 Final Project — Stanford

## 1. Overview

This project applies Item Response Theory (IRT) to analyze whether benchmark items on the Open LLM Leaderboard v2 (BBH, MATH-Hard, GPQA-Diamond, MuSR) maintain consistent psychometric properties across:
- **Temporal cohorts** of LLMs: Q2-2024, Q3-2024, Q4-2024, Q1-2025
- **Architecture families**: Gemma2, Llama, Mistral, Qwen2

We fit 2PL IRT models to extract per-item difficulty (b) and discrimination (a) parameters, then test whether these parameters drift as the population of evaluated models changes over time and across architectures.

> **Note:** The final paper reports results for BBH and MATH-Hard only. GPQA-Diamond and MuSR have only 3 subtask groups each — too few for reliable rank-order analysis. Their data and IRT fits are retained in this repo for exploratory use.

---

## 2. Repository Structure

```
.
├── data_raw/                          # Raw inputs (not fully tracked in git — see §4)
│   ├── results/                       # Per-model result JSONs from Open LLM Leaderboard v2
│   ├── bbh/                           # BIG-Bench Hard item files
│   ├── MATH-Hard/                     # MATH-Hard test/train splits
│   └── models.parquet                 # Model metadata (included in repo)
├── data_raw_clean/                    # Cleaned score matrices, one file per benchmark
│   ├── {bbh,math,gpqa,musr}_scores_full.parquet    # All evaluation records
│   └── {bbh,math,gpqa,musr}_scores_unique.parquet  # Deduplicated (latest run per model)
├── data_irt/                          # Fitted IRT parameters (output of p2)
│   ├── bbh/
│   │   ├── bbh_difficulties_2pl.csv      # rows = (cohort, arch), cols = subtasks
│   │   └── bbh_discriminations_2pl.csv
│   ├── math/
│   ├── gpqa/
│   └── musr/
├── notebooks/                         # Analysis pipeline — run in order
│   ├── p0_split_model_cohort.ipynb    # (optional) Assign cohort labels; visualize arch mix
│   ├── p1_model_scores_collect.ipynb  # Extract per-model benchmark scores → data_raw_clean/
│   ├── p2_irt_pipe.ipynb              # Fit 2PL IRT per (arch × cohort) → data_irt/
│   ├── p3_post_analysis.ipynb         # Analysis & figures (Fig 1–4)
│   └── fig{1..4}_*.pdf                # Output figures
└── torch_measure_ext/                 # Project-local IRT model extensions
    ├── beta_twopl.py                  # Beta-regularised 2PL model
    ├── beta_rasch.py                  # Beta-regularised Rasch model
    └── loss_fn.py                     # Custom loss functions
```

---

## 3. Environment Setup

**Requirements:** Python 3.9+, pip

**Computation:** 8 GB RAM minimum; 16 GB recommended. No GPU required.

### 3.1 Install `torch_measure`

This project's IRT fitting depends on [`torch_measure`](http://torch-measure.readthedocs.io). Follow the installation guide at that link, then install the project's local extension:

```bash
pip install -e .
```

### 3.2 Install remaining dependencies

```bash
pip install pandas numpy scipy matplotlib statsmodels pyarrow jupyter
```

---

## 4. Public Data Download

All external data lives in `data_raw/`. Two sources are needed:

**1. Model metadata** — already included in this repo as `data_raw/models.parquet`.
Source: [`open-llm-leaderboard/contents`](https://huggingface.co/datasets/open-llm-leaderboard/contents) on Hugging Face.

**2. Per-model benchmark results** — not included due to size (~several GB). Clone directly into `data_raw/`:

```bash
git clone https://huggingface.co/datasets/open-llm-leaderboard/contents data_raw/results
```

> The rest of the pipeline assumes `data_raw/results/` exists and is populated before running any notebook.


---

## 5. Reproducing Results via Notebooks

All notebooks are in `notebooks/`. Run them in the order below; each writes its outputs to the directory the next notebook reads from. Each notebook is self-documented — cells below the header explain parameters and expected outputs.

```bash
# Execute a notebook non-interactively (run from the project root)
jupyter nbconvert --to notebook --execute notebooks/<notebook>.ipynb
```

| # | Notebook | Input | Output                          | Notes |
|---|----------|-------|---------------------------------|-------|
| 0 | `p0_split_model_cohort.ipynb` | `data_raw/models.parquet` | `data_raw_clean/models.parquet` | Assigns quarterly cohort labels and visualizes architecture mix per cohort. |
| 1 | `p1_model_scores_collect.ipynb` | `data_raw/results/` | `data_raw_clean/`               | Extracts per-model benchmark scores from raw JSON files. Run once per benchmark by setting `BENCH` at the top of the notebook. |
| 2 | `p2_irt_pipe.ipynb` | `data_raw_clean/` | `data_irt/`                     | Fits 2PL IRT models for every `(arch × cohort)` group. Results may vary slightly across hardware; reported numbers are stable to ±0.01 in ρ. |
| 3 | `p3_post_analysis.ipynb` | `data_irt/` | `fig{1..4}_*.pdf`               | Computes all statistics and generates the four paper figures. The `data_irt/` CSVs are the exact inputs — figures can be regenerated without re-running IRT. |

---

## 6. Acknowledgements

Special thanks to the [Stanford AI Measure Science (AIMS) Lab](https://aimslab.stanford.edu) for their foundational work on psychometric measurement of AI systems.
