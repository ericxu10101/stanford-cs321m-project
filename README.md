# IRT Parameter Drift Across LLM Generations and Architectures on Open LLM Leaderboard v2

CS321M Spring 2026 Final Project — Stanford

## Overview

This project applies Item Response Theory (IRT) to analyze whether benchmark items on the Open LLM Leaderboard v2 (BBH, MATH-Hard, GPQA-Diamond, MuSR) maintain consistent psychometric properties across:
- **Temporal cohorts** of LLMs: Q2-2024, Q3-2024, Q4-2024, Q1-2025
- **Architecture families**: Gemma2, Llama, Mistral, Qwen2

We fit 2PL IRT models to extract per-item difficulty (b) and discrimination (a) parameters, then test whether these parameters drift as the population of evaluated models changes over time and across architectures.

---

## Repository Structure

```
.
├── data_raw/                    # Raw benchmark data and Open LLM Leaderboard results
│   ├── bbh/                     # BIG-Bench Hard items and subtasks
│   ├── MATH-Hard/               # MATH-Hard test/train splits
│   ├── MMLU-Pro/                # MMLU-Pro items
│   ├── results/                 # Per-model result files from Open LLM Leaderboard v2
│   └── models.parquet           # Model metadata
├── data_raw_clean/              # Cleaned score matrices (parquet)
│   ├── bbh_scores_*.parquet
│   ├── math_scores_*.parquet
│   ├── gpqa_scores_*.parquet
│   ├── musr_scores_*.parquet
│   └── models.parquet
├── data_irt/                    # IRT parameter outputs (CSV)
│   ├── bbh/                     # bbh_difficulties_2pl.csv, bbh_discriminations_2pl.csv
│   ├── math/
│   ├── gpqa/
│   └── musr/
├── notebooks/                   # Analysis pipeline (run in order)
│   ├── p1_data_collect.ipynb    # Step 1: collect and clean leaderboard data
│   ├── p2_population_stats.ipynb # Step 2: cohort descriptive statistics
│   ├── p3_irt.ipynb             # Step 3: fit IRT models per cohort/arch
│   ├── p3_irt_pipe.ipynb        # Step 3 (pipeline version)
│   ├── results.ipynb            # Step 4: compute all reported statistics
│   └── generate_figures.py      # Step 5: generate all 5 paper figures
├── torch_measure/               # Core psychometric library (IRT models, fitting, metrics)
└── torch_measure_ext/           # Extended IRT models (beta-Rasch, beta-2PL)
```

---

## Environment Setup

**Requirements:** Python 3.9+, pip

### 1. Install dependencies

No `requirements.txt` is checked in because the data files already contain pre-fitted IRT parameters. To reproduce everything from scratch, install:

```bash
pip install pandas numpy scipy matplotlib statsmodels pyarrow jupyter
pip install torch  # required by torch_measure
```

Install the local `torch_measure` package in editable mode:

```bash
pip install -e .
```

> **Note:** `torch_measure` requires PyTorch. Install the CPU-only build if you do not have a GPU:
> `pip install torch --index-url https://download.pytorch.org/whl/cpu`

### 2. Verify install

```python
import torch_measure
print(torch_measure.__version__)  # should print 0.1.dev39+...
```

**Estimated runtimes:**
- Data collection (`p1`): ~10–30 min depending on disk I/O
- Population stats (`p2`): < 1 min
- IRT fitting (`p3`): 5–20 min per benchmark (CPU); < 5 min with GPU
- Figure generation: < 1 min

**Computational requirements:** 8 GB RAM minimum; 16 GB recommended for full data. No GPU required (all IRT fitting runs on CPU in reasonable time).

---

## Reproducing Results

Run the notebooks **in order**. Each notebook saves its outputs to the next stage's input directory.

### Step 1 — Data collection

```bash
jupyter nbconvert --to notebook --execute notebooks/p1_data_collect.ipynb
```

Reads raw leaderboard results from `data_raw/results/`, cleans model metadata, and writes `data_raw_clean/*.parquet`.

### Step 2 — Population statistics

```bash
jupyter nbconvert --to notebook --execute notebooks/p2_population_stats.ipynb
```

Computes cohort-level descriptive statistics and KS/chi-square tests for Table 1.

### Step 3 — IRT fitting

```bash
jupyter nbconvert --to notebook --execute notebooks/p3_irt_pipe.ipynb
```

Fits 2PL IRT models for each (benchmark × cohort) and (benchmark × architecture) combination. Writes difficulty and discrimination CSVs to `data_irt/`.

> All stochastic processes use fixed seeds. Default seed is set in the notebook header cell.

### Step 4 — Analysis and statistics

```bash
jupyter nbconvert --to notebook --execute notebooks/results.ipynb
```

Computes Spearman ρ matrices, rank variances, and all numbers reported in the paper.

### Step 5 — Figures

```bash
python notebooks/generate_figures.py \
    --data_dir  data_irt \
    --output_dir notebooks
```

Produces five PDF figures in `notebooks/`:

| Figure | File | What it shows |
|--------|------|---------------|
| Fig 1 | `fig1_rho_heatmaps.pdf` | Spearman ρ heatmaps of subtask rank orderings (BBH & MATH, cohort & arch) |
| Fig 2 | `fig2_dendrograms.pdf` | Architecture clustering by ranking similarity |
| Fig 3 | `fig3_trajectories.pdf` | Discrimination rank trajectories across quarterly cohorts |
| Fig 4 | `fig4_typology.pdf` | Subtask typology: temporal vs architecture rank instability |
| Fig 5 | `fig5_crossbenchmark.pdf` | Cross-benchmark long-range ρ and Δρ by architecture |

---

## Datasets

All benchmark data is included in `data_raw/`. No external downloads are needed.

| Dataset | Source | Location |
|---------|--------|----------|
| BIG-Bench Hard (BBH) | [suzgunm/challenging-big-bench-tasks](https://github.com/suzgunm/challenging-big-bench-tasks) | `data_raw/bbh/` |
| MATH-Hard | [HendrycksTest](https://github.com/hendrycks/math) | `data_raw/MATH-Hard/` |
| Open LLM Leaderboard v2 results | Hugging Face Hub | `data_raw/results/` |

GPQA-Diamond and MuSR items are loaded programmatically via `torch_measure.datasets`. An internet connection is required only if these are not already cached.

---

## Key Findings

- **Difficulty (b-parameter) is stable**: Spearman ρ between Q2-2024 and Q1-2025 cohorts exceeds 0.94 on BBH and 0.85 on MATH.
- **Discrimination (a-parameter) is not**: Long-range ρ drops to 0.75–0.80 on BBH and lower on MATH.
- **Architecture matters more for discrimination** than for difficulty: cross-architecture Δρ = ρ\_diff − ρ\_disc is positive for all architecture pairs.
- **Subtask heterogeneity**: temporal and architectural instability cluster in a distinct subset of subtasks identifiable before deployment.

---

## Reproducibility Notes

- Random seeds are set via `np.random.seed(42)` and `torch.manual_seed(42)` at the top of each notebook.
- IRT fitting uses stochastic variational inference (SVI); results may vary slightly across hardware but reported numbers are stable to ±0.01 in ρ.
- The `data_irt/` CSVs are the exact inputs to `generate_figures.py`; figures can be regenerated without re-running IRT.
