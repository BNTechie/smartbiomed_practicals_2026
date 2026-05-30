# SMARTbiomed — Association Testing (GWAS) Practicals

Hands-on Jupyter notebooks for a one-day course on genome-wide association studies (GWAS),
designed to run in **Google Colab** with no setup. The emphasis is on understanding what GWAS
software does *under the hood* — every core routine (linear/logistic regression, QC, HWE,
Manhattan/QQ/trumpet/LocusZoom plots, PCA, fine-mapping) is implemented from scratch with
NumPy/SciPy/Matplotlib. Four sessions: a morning on running & interpreting GWAS, an afternoon on
its complexities (population structure, relatedness) and fine-mapping.

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

**Session 3 — Complexities of GWAS: population structure & relatedness** (`session3/`)
See stratification inflate a GWAS, compute **PCA from scratch** (SVD of standardised genotypes) on an
**"All of Us"-style diverse cohort anchored by a "1000 Genomes"-style reference panel**, assign
continental ancestry with a **random-forest classifier + confidence threshold** (admixed individuals
fall out as "Unassigned"), and correct the confounding with PC covariates. Challenges: who the
classifier leaves unassigned (and the threshold trade-off), and the genetic relatedness matrix (GRM)
finding hidden siblings.

**Session 4 — Fine-mapping** (`session4/`)
Go from a peak of LD-correlated significant variants to a short list of causal candidates:
LocusZoom by r², conditional ("poor man's") fine-mapping, and posterior inclusion probabilities /
95% credible sets via Wakefield's approximate Bayes factor. Challenges: a PIP LocusZoom, resolution
vs sample size, and why a single-causal credible set fails when a locus has two signals (SuSiE idea).

The afternoon question design is written up in [`sessions_3_4_outline.md`](sessions_3_4_outline.md).

## Open in Colab

Each notebook is self-contained: the first cell downloads the required data from this repo
automatically (no manual download needed).

| Session | Worksheet (fill in the blanks) | Answers (with solutions) | Executed (read-only) |
|---|---|---|---|
| **1: Intro to GWAS** | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/nikbaya/smartbiomed_practicals_2026/blob/main/session1/practical.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/nikbaya/smartbiomed_practicals_2026/blob/main/session1/answers.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/nikbaya/smartbiomed_practicals_2026/blob/main/session1/run.ipynb) |
| **2: Interpreting GWAS** | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/nikbaya/smartbiomed_practicals_2026/blob/main/session2/practical.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/nikbaya/smartbiomed_practicals_2026/blob/main/session2/answers.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/nikbaya/smartbiomed_practicals_2026/blob/main/session2/run.ipynb) |
| **3: Structure & relatedness** | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/nikbaya/smartbiomed_practicals_2026/blob/main/session3/practical.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/nikbaya/smartbiomed_practicals_2026/blob/main/session3/answers.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/nikbaya/smartbiomed_practicals_2026/blob/main/session3/run.ipynb) |
| **4: Fine-mapping** | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/nikbaya/smartbiomed_practicals_2026/blob/main/session4/practical.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/nikbaya/smartbiomed_practicals_2026/blob/main/session4/answers.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/nikbaya/smartbiomed_practicals_2026/blob/main/session4/run.ipynb) |

- **Worksheet** (`practical.ipynb`) — skeleton code with `???` blanks for students to complete.
- **Answers** (`answers.ipynb`) — same notebook with collapsible solution cells.
- **Executed** (`run.ipynb`) — fully run, for reference / instructors. Static HTML renders are in
  `session{1,2}/summary_session{1,2}.html`.

## Running locally

```bash
pip install numpy scipy pandas matplotlib scikit-learn jupyter
jupyter notebook session1/practical.ipynb
```

Data lives in `data/` (tracked with **Git LFS**): `gwas_data.npz` (Session 1/2 genotypes +
phenotypes), `fly_data.csv` (Drosophila cross), `sumstats_real.npz` (thinned Pan-UKB sumstats),
`pca_data.npz` (Session 3 cohort + reference panel), `finemap_data.npz` (Session 4 loci).
With Git LFS installed, `git clone` fetches them; otherwise the notebooks download them on first run.

## Regenerating the materials (instructors)

| Script | Purpose |
|---|---|
| `generate_data.py` | Simulate the Session 1/2 genotypes + phenotypes and the fly cross → `data/`. |
| `fetch_sumstats.py` | Download + thin the real Pan-UKB LDL/CAD/height sumstats → `data/sumstats_real.npz`. |
| `generate_pca_data.py` | Simulate Session 3 structured-population data → `data/pca_data.npz`. |
| `generate_finemap_data.py` | Simulate Session 4 fine-mapping loci → `data/finemap_data.npz`. |
| `create_notebooks.py` | Build all twelve notebooks from the shared source. |

```bash
python generate_data.py          # Session 1/2 simulated data
python fetch_sumstats.py         # real Pan-UKB sumstats (streams ~7 GB, writes ~11 MB)
python generate_pca_data.py      # Session 3 data
python generate_finemap_data.py  # Session 4 data
python create_notebooks.py       # (re)build the notebooks
```

Data sources: [Pan-UKB](https://pan.ukbb.broadinstitute.org/) summary statistics (EUR).
