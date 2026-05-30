# SMARTbiomed — Association Testing (GWAS) Practicals

Hands-on Jupyter notebooks for a one-day course on genome-wide association studies (GWAS),
designed to run in **Google Colab** with no setup. The emphasis is on understanding what GWAS
software does *under the hood* — every core routine (linear/logistic regression, QC, HWE,
Manhattan/QQ/trumpet/LocusZoom plots) is implemented from scratch with NumPy/SciPy/Matplotlib.

## Sessions

**Session 1 — Introduction to association testing** (`session1/`)
Quality control (missingness, MAF, a provided HWE test), running linear and logistic GWAS,
interpreting effect sizes, and three simulated phenotypes (a low-h² continuous trait, a
~10%-prevalence binary trait, and an uncorrelated fully-polygenic trait) on ~50k variants across
chromosome 1.
Challenges: an HWE chi-squared test from scratch, dominant/recessive encodings, a manual LocusZoom
plot, a Drosophila linkage analysis à la Sturtevant (1913), age-of-onset ascertainment, and
polygenic scores (PGS).

**Session 2 — Interpreting GWAS** (`session2/`)
Genome-wide Manhattan plots, QQ plots and λ_GC, pleiotropy, and a trumpet/power-curve plot — built
on **real Pan-UKB summary statistics** (EUR) for LDL cholesterol, coronary/ischaemic heart disease,
and height. Challenges: QQ confidence bands, MAF-stratified QQ, winner's curse (discovery/validation
resampling), and linking a real lead variant to the GWAS Catalog.
Individual-level analyses (winner's curse, null-QQ contrast) reuse the simulated Session 1 cohort.

## Open in Colab

Each notebook is self-contained: the first cell downloads the required data from this repo
automatically (no manual download needed).

| Session | Worksheet (fill in the blanks) | Answers (with solutions) | Executed (read-only) |
|---|---|---|---|
| **1: Intro to GWAS** | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/nikbaya/smartbiomed_practicals_2026/blob/main/session1/practical.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/nikbaya/smartbiomed_practicals_2026/blob/main/session1/answers.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/nikbaya/smartbiomed_practicals_2026/blob/main/session1/run.ipynb) |
| **2: Interpreting GWAS** | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/nikbaya/smartbiomed_practicals_2026/blob/main/session2/practical.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/nikbaya/smartbiomed_practicals_2026/blob/main/session2/answers.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/nikbaya/smartbiomed_practicals_2026/blob/main/session2/run.ipynb) |

- **Worksheet** (`practical.ipynb`) — skeleton code with `???` blanks for students to complete.
- **Answers** (`answers.ipynb`) — same notebook with collapsible solution cells.
- **Executed** (`run.ipynb`) — fully run, for reference / instructors. Static HTML renders are in
  `session{1,2}/summary_session{1,2}.html`.

## Running locally

```bash
pip install numpy scipy pandas matplotlib jupyter
jupyter notebook session1/practical.ipynb
```

Data lives in `data/` (tracked with **Git LFS**): `gwas_data.npz` (simulated genotypes +
phenotypes), `fly_data.csv` (Drosophila cross), `sumstats_real.npz` (thinned Pan-UKB sumstats).
With Git LFS installed, `git clone` fetches them; otherwise the notebooks download them on first run.

## Regenerating the materials (instructors)

| Script | Purpose |
|---|---|
| `generate_data.py` | Simulate the Session 1 genotypes + phenotypes and the fly cross → `data/`. |
| `fetch_sumstats.py` | Download + thin the real Pan-UKB LDL/CAD/height sumstats → `data/sumstats_real.npz`. |
| `create_notebooks.py` | Build all six notebooks from the shared source. |

```bash
python generate_data.py        # simulated data
python fetch_sumstats.py       # real Pan-UKB sumstats (streams ~7 GB, writes ~11 MB)
python create_notebooks.py     # (re)build the notebooks
```

Data sources: [Pan-UKB](https://pan.ukbb.broadinstitute.org/) summary statistics (EUR).
