# IRT Parameter Drift Across LLM Generations and Architectures on Open LLM Leaderboard v2

CS321M Spring 2026 Final Project — Stanford

## 1. Overview

This project applies Item Response Theory (IRT) to analyze whether benchmark items on the Open LLM Leaderboard v2 (BBH, MATH-Hard, GPQA-Diamond, MuSR) maintain consistent psychometric properties across:
- **Temporal cohorts** of LLMs: Q2-2024, Q3-2024, Q4-2024, Q1-2025
- **Architecture families**: Gemma2, Llama, Mistral, Qwen2

We fit 2PL IRT models to extract per-item difficulty (b) and discrimination (a) parameters, then test whether these parameters drift as the population of evaluated models changes over time and across architectures.

---

## 2. Repository Structure

```
.
├── data_raw/                    # Raw benchmark data and Open LLM Leaderboard results
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
│   ├── p0_population_stats.ipynb 
│   ├── p1_data_collect.ipynb 
│   ├── p2_irt_pipe.ipynb        
│   ├── p3_post_analysis.ipynb   
└── torch_measure_ext/           # Extended IRT models (beta-Rasch, beta-2PL)
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

- `p0_population_stats.ipynb` *(optional)* — Exploratory analysis of model metadata in `data_raw/models.parquet`. Not required to reproduce paper results.
- `p1_data_collect.ipynb` — Preprocesses raw leaderboard results from `data_raw/`; outputs cleaned score matrices to `data_raw_clean/`.
- `p2_irt_pipe.ipynb` — Core IRT fitting pipeline. Fits 2PL models per cohort and architecture family; outputs parameter CSVs to `data_irt/`. (IRT fitting uses results may vary slightly across hardware.)
- `p3_post_analysis.ipynb` — Full analysis and figure generation. Reproduces all statistics and figures in the paper. (The `data_irt/` CSVs are the exact inputs to `p3_post_analysis.ipynb`; figures can be regenerated without re-running IRT.)


| # | Notebook | Input | Notes | Output |                                                                                                                                                                                                   
|---|----------|-------|-------|--------|
| 0 | `p0_population_stats.ipynb` | `data_raw/models.parquet` | Optional. Exploratory statistics on model metadata. | — |                                                                                                                   
| 1 | `p1_data_collect.ipynb` | `data_raw/` | Cleans and filters leaderboard results into per-benchmark score matrices. | `data_raw_clean/` |                                                                                               
| 2 | `p2_irt_pipe.ipynb` | `data_raw_clean/` | Fits 2PL IRT models for each (benchmark × cohort) and (benchmark × architecture) group. | `data_irt/` |                                                                                     
| 3 | `p3_post_analysis.ipynb` | `data_irt/` | Computes all statistics reported in the paper and generates the five figures. | figures & stats |
