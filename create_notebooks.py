#!/usr/bin/env python3
"""
Generate SMARTbiomed GWAS practical notebooks for Sessions 1 & 2.
Produces:
  session1/practical.ipynb  — student worksheet
  session1/answers.ipynb    — same with collapsed solution cells
  session2/practical.ipynb
  session2/answers.ipynb

Run: python create_notebooks.py
"""
import json
import os
import itertools

# ─── Helpers ──────────────────────────────────────────────────────────────────

_ids = (f"cell{i:04d}" for i in itertools.count(1))

def md(source):
    return {"cell_type": "markdown", "id": next(_ids), "metadata": {}, "source": source}

def code(source, tags=None):
    meta = {"tags": tags} if tags else {}
    return {
        "cell_type": "code", "execution_count": None, "id": next(_ids),
        "metadata": meta, "outputs": [], "source": source,
    }

def solution(source):
    """A collapsed solution cell for the answer notebook."""
    header = "# @title Solution { display-mode: \"form\" }\n# ▼ Click ▶ to reveal solution\n\n"
    return code(header + source)

def notebook(cells):
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"codemirror_mode": {"name": "ipython", "version": 3},
                              "name": "python", "version": "3.10.0"},
            "colab": {"provenance": []},
        },
        "nbformat": 4, "nbformat_minor": 5,
    }

def save(nb, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(nb, f, indent=1)
    print(f"  Wrote {path}")


# ─── Shared content ───────────────────────────────────────────────────────────

# GitHub repo the notebooks fetch their data from (public; data tracked with Git LFS)
REPO   = "nikbaya/smartbiomed_practicals_2026"
BRANCH = "main"


def colab_badge(nb_path):
    """Markdown 'Open in Colab' badge linking to nb_path within the repo."""
    return ("[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)]"
            f"(https://colab.research.google.com/github/{REPO}/blob/{BRANCH}/{nb_path})")


def data_bootstrap(files):
    """
    Return code that locates data/ (../data or ./data) or, failing that (e.g. a fresh
    Colab session), downloads the required files from the public GitHub repo.
    LFS-tracked .npz are served via media.githubusercontent.com; plain files via raw.
    Leaves DATA_DIR pointing at the directory that holds the files.
    """
    flist = ", ".join(repr(f) for f in files)
    return f'''# Locate the data directory, downloading from GitHub if needed (e.g. on Colab).
import os, urllib.request
_NEED = [{flist}]
_LFS  = {{'gwas_data.npz', 'sumstats_real.npz'}}   # tracked with Git LFS (media URL)
def _has_all(d):
    return d and all(os.path.exists(os.path.join(d, f)) for f in _NEED)
DATA_DIR = next((d for d in ('../data', 'data') if _has_all(d)), None)
if DATA_DIR is None:
    DATA_DIR = 'data'; os.makedirs(DATA_DIR, exist_ok=True)
    for _f in _NEED:
        _dest = os.path.join(DATA_DIR, _f)
        if os.path.exists(_dest):
            continue
        _base = ('https://media.githubusercontent.com/media' if _f in _LFS
                 else 'https://raw.githubusercontent.com')
        _url = f'{{_base}}/{REPO}/{BRANCH}/data/{{_f}}'
        print(f'Downloading {{_f}} from GitHub ...')
        urllib.request.urlretrieve(_url, _dest)
'''


IMPORTS = """\
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from scipy import stats, special
import warnings
warnings.filterwarnings('ignore')

plt.rcParams.update({'figure.dpi': 80, 'font.size': 11})
print("Libraries loaded.")
"""

# Fallback block-LD data generation (used when pre-generated files are absent)
FALLBACK_GEN = """\
# ── Fallback: generate a synthetic dataset with block LD ──────────────────────
# Used when the instructor-provided data files are not present.
# Results are reproducible (fixed seed) but use simulated rather than real LD.
np.random.seed(2026)
N, M_raw = 10_000, 12_000
CHROM = 1          # all variants on chr1 (simulated)
POS_START = 1_000  # kbp

# EUR-like MAF distribution: right-skewed, min 5% for block-LD variants
maf_true = np.random.beta(0.5, 2.0, M_raw) * 0.45 + 0.05

# Block LD: groups of 50 variants share a common ancestor haplotype (r²≈0.64)
block_size = 50
G_hap1 = np.zeros((N, M_raw), dtype=np.int8)
G_hap2 = np.zeros((N, M_raw), dtype=np.int8)
for b in range((M_raw - 1) // block_size + 1):
    s = b * block_size;  e = min(s + block_size, M_raw);  nb = e - s
    anc = np.random.rand(N) < maf_true[s:e].mean()
    alpha = 0.8
    for hap_arr in [G_hap1, G_hap2]:
        mix  = np.random.rand(N, nb) < alpha
        indp = (np.random.rand(N, nb) < maf_true[None, s:e])
        hap_arr[:, s:e] = np.where(mix, anc[:, None], indp).astype(np.int8)

G_raw = G_hap1 + G_hap2   # diploid (0/1/2), shape (N, M_raw)

# Inject rare variants (MAF 0.1–1%) — fail MAF QC
rare_cols_fb = np.random.choice(M_raw, 1000, replace=False)
for _j in rare_cols_fb:
    _af = np.random.uniform(0.001, 0.01)
    G_raw[:, _j] = np.random.binomial(2, _af, N).astype(float)

# Inject other QC failures
non_rare_fb = np.setdiff1d(np.arange(M_raw), rare_cols_fb)
hwe_bad = np.random.choice(non_rare_fb, 500, replace=False)
remaining = np.setdiff1d(non_rare_fb, hwe_bad)
miss_bad = np.random.choice(remaining, 300, replace=False)
low_miss_vars = np.random.choice(np.setdiff1d(remaining, miss_bad), 2500, replace=False)
for j in hwe_bad:
    mask = G_raw[:, j] == 1
    G_raw[mask, j] = np.random.choice([0, 2], mask.sum())
G_raw_f = G_raw.astype(float)
for j in miss_bad:
    n_miss = np.random.randint(int(0.06 * N), int(0.20 * N))
    idx = np.random.choice(N, n_miss, replace=False)
    G_raw_f[idx, j] = np.nan
for j in low_miss_vars:
    n_miss = np.random.randint(int(0.005 * N), int(0.045 * N))
    idx = np.random.choice(N, n_miss, replace=False)
    G_raw_f[idx, j] = np.nan
G_raw = G_raw_f

pos = (POS_START + np.arange(M_raw) * 2.5).astype(int)  # ~2.5 kb spacing (kbp)
rsids = np.array([f"rs{100000 + i}" for i in range(M_raw)])

# Covariates (UKB-like): age-at-recruitment distribution (range 37–73, median 58)
_p_age = [0, .10, .20, .30, .40, .50, .60, .70, .80, .90, 1.0]
_v_age = [37, 44, 48, 52, 55, 58, 60, 62, 64, 67, 73]
age = np.interp(np.random.rand(N), _p_age, _v_age)
sex = np.random.binomial(1, 0.5, N)  # 0=female, 1=male

# Simulate continuous phenotype: spike-and-slab prior (h²≈0.035), few causal variants
n_causal = 3   # 3 causal variants, low h², realistic QQ plot
causal_idx = np.sort(np.random.choice(M_raw, n_causal, replace=False))
true_betas = np.zeros(M_raw)
# MAF-dependent effect sizes (rare variants get larger effects)
het_c = 2 * maf_true[causal_idx] * (1 - maf_true[causal_idx])
sigma_c = (het_c + 1e-8) ** (-0.375); sigma_c /= sigma_c.mean()
true_betas[causal_idx] = np.random.normal(0, 0.3 * sigma_c)

G_clean = np.where(np.isnan(G_raw), 0, G_raw)
G_std = (G_clean - np.nanmean(G_raw, 0)) / (np.nanstd(G_raw, 0) + 1e-8)
genetic_cont = G_std @ true_betas
genetic_cont = genetic_cont / (genetic_cont.std() + 1e-8) * np.sqrt(0.025)
y_cont = (genetic_cont
          + 0.15 * (sex - 0.5)
          + 0.10 * (age - 57) / 8
          + np.random.normal(0, np.sqrt(0.965), N))
y_cont = (y_cont - y_cont.mean()) / y_cont.std()

# Binary phenotype via liability threshold (~10% cases)
threshold = np.percentile(y_cont, 90)
y_bin = (y_cont >= threshold).astype(np.int8)

# Fly cross dataset
np.random.seed(42)
n_fly = 2000
x_pos_cM = np.array([0.0, 5.0, 15.0, 30.0, 35.0, 50.0])
fly_sex = np.array([1]*(n_fly//2) + [0]*(n_fly - n_fly//2))
male_haps = np.zeros((n_fly//2, 6), dtype=np.int8)
for i in range(n_fly//2):
    curr = np.random.randint(0, 2)
    male_haps[i, 0] = curr
    for j in range(1, 6):
        curr = (curr + np.random.poisson((x_pos_cM[j] - x_pos_cM[j-1]) / 100)) % 2
        male_haps[i, j] = curr
x_phenos = np.vstack([male_haps, np.zeros((n_fly - n_fly//2, 6), dtype=np.int8)])
auto_tr = np.column_stack([np.random.binomial(1, 0.35, n_fly),
                            np.random.binomial(1, 0.48, n_fly)])
perm = np.random.permutation(n_fly)
fly_df = pd.DataFrame({
    'sex':          fly_sex[perm],
    'trait_eye':    x_phenos[perm, 0], 'trait_thorax': auto_tr[perm, 0],
    'trait_wing':   x_phenos[perm, 1], 'trait_leg':    x_phenos[perm, 2],
    'trait_notch':  x_phenos[perm, 3], 'trait_vein':   x_phenos[perm, 4],
    'trait_scute':  x_phenos[perm, 5], 'trait_band':   auto_tr[perm, 1],
})
# Polygenic trait: fully polygenic, very low h², uncorrelated with y_cont
G_clean_p = np.where(np.isnan(G_raw), 0, G_raw)
betas_poly_fb = np.random.normal(0, 1, M_raw) / np.sqrt(M_raw)
y_poly_g = G_clean_p @ betas_poly_fb
y_poly_g = y_poly_g / (y_poly_g.std() + 1e-8) * np.sqrt(0.02)
y_poly = (y_poly_g + np.random.normal(0, np.sqrt(0.98), N))
y_poly = (y_poly - y_poly.mean()) / y_poly.std()

dom_idx_qc = -1; rec_idx_qc = -1   # non-additive loci not available in fallback
print("Fallback dataset ready (block-LD simulation, spike-and-slab phenotype)")
"""

LOAD_DATA = """\
# ─────────────────────────────────────────────────────────────────────────────
# SETUP — Run this cell once. No modification needed.
# Loads the pre-generated GWAS dataset (or falls back to a synthetic version).
# ─────────────────────────────────────────────────────────────────────────────
import os, numpy as np, pandas as pd
from scipy import stats, special
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import warnings; warnings.filterwarnings('ignore')
plt.rcParams.update({'figure.dpi': 80, 'font.size': 11})

""" + data_bootstrap(['gwas_data.npz', 'fly_data.csv']) + """
if os.path.exists(os.path.join(DATA_DIR, 'gwas_data.npz')):
    print("Loading pre-generated GWAS data...")
    d = np.load(os.path.join(DATA_DIR, 'gwas_data.npz'), allow_pickle=True)
    _G    = d['G_raw']                        # int8: 0/1/2, -1=missing (compact storage)
    G_raw = np.where(_G == -1, np.nan, _G.astype(np.float32))  # float32 NaN for missing
    del _G
    pos        = d['pos']                    # variant positions (kbp)
    rsids      = d['rsids']                  # variant RSIDs
    age        = d['age']
    sex        = d['sex']                    # 0=female, 1=male
    y_cont     = d['y_cont']                 # continuous trait (h²≈0.035, few causal variants)
    y_poly     = d['y_poly']                 # polygenic trait  (h²≈0.02, fully polygenic)
    y_bin      = d['y_bin']                  # binary trait     (~10% cases, liability threshold)
    true_betas = d['true_betas']             # true causal effect sizes
    dom_idx_qc = int(d['dom_idx_qc'][0])     # post-QC column index of dominant locus
    rec_idx_qc = int(d['rec_idx_qc'][0])     # post-QC column index of recessive locus
    fly_df     = pd.read_csv(os.path.join(DATA_DIR, 'fly_data.csv'))
    N, M_raw = G_raw.shape
    CHROM = 1
    print(f"Loaded: {N:,} samples × {M_raw:,} variants (chr{CHROM}, pre-QC)")
    print(f"  Continuous trait: standardised liability (h²≈0.035)")
    print(f"  Binary trait: {y_bin.sum():,} cases ({100*y_bin.mean():.1f}%, liability threshold)")
else:
""" + "\n".join("    " + line for line in FALLBACK_GEN.splitlines()) + """

N, M_raw = G_raw.shape
print(f"\\nReady: N={N:,} samples, M_raw={M_raw:,} variants (pre-QC)")
"""

# Student version: no fallback simulation — raises a clear error if data is absent
LOAD_DATA_STUDENT = """\
# ─────────────────────────────────────────────────────────────────────────────
# SETUP — Run this cell once. No modification needed.
# Loads the pre-generated GWAS dataset.
# ─────────────────────────────────────────────────────────────────────────────
import os, numpy as np, pandas as pd
from scipy import stats, special
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import warnings; warnings.filterwarnings('ignore')
plt.rcParams.update({'figure.dpi': 80, 'font.size': 11})

""" + data_bootstrap(['gwas_data.npz', 'fly_data.csv']) + """
if not os.path.exists(os.path.join(DATA_DIR, 'gwas_data.npz')):
    raise FileNotFoundError(
        "Could not find or download gwas_data.npz. Check your internet connection "
        "(the notebook downloads data from GitHub on first run), or ask your instructor."
    )

print("Loading pre-generated GWAS data...")
d = np.load(os.path.join(DATA_DIR, 'gwas_data.npz'), allow_pickle=True)
_G    = d['G_raw']                        # int8: 0/1/2, -1=missing (compact storage)
G_raw = np.where(_G == -1, np.nan, _G.astype(np.float32))  # float32 NaN for missing
del _G
pos        = d['pos']                    # variant positions (kbp)
rsids      = d['rsids']                  # variant RSIDs
age        = d['age']
sex        = d['sex']                    # 0=female, 1=male
y_cont     = d['y_cont']                 # continuous phenotype (standardised)
y_poly     = d['y_poly']                 # polygenic phenotype (h²≈0.02, uncorrelated with y_cont)
y_bin      = d['y_bin']                  # binary phenotype (0/1, ~10% cases)
true_betas = d['true_betas']             # true causal effect sizes
dom_idx_qc = int(d['dom_idx_qc'][0])     # post-QC column index of dominant locus
rec_idx_qc = int(d['rec_idx_qc'][0])     # post-QC column index of recessive locus
fly_df     = pd.read_csv(os.path.join(DATA_DIR, 'fly_data.csv'))
N, M_raw = G_raw.shape
CHROM = 1
print(f"Loaded: {N:,} samples × {M_raw:,} variants (chr{CHROM}, pre-QC)")
print(f"  Continuous trait: standardised liability (h²≈0.035)")
print(f"  Binary trait: {y_bin.sum():,} cases ({100*y_bin.mean():.1f}%, liability threshold)")

N, M_raw = G_raw.shape
print(f"\\nReady: N={N:,} samples, M_raw={M_raw:,} variants (pre-QC)")
"""

# Provided run_gwas function (vectorised OLS)
RUN_GWAS_FN = """\
def run_gwas(y, G, covars=None, chunk=5_000):
    \"\"\"
    Vectorised OLS GWAS: regress phenotype y on each column of G.
    Processes variants in batches to keep peak memory usage low.

    Parameters
    ----------
    y      : (N,)    phenotype (will be mean-centred internally)
    G      : (N, M)  genotype matrix (0/1/2), NaN = missing (treated as mean)
    covars : (N, k)  covariate matrix (optional); age and sex recommended
    chunk  : int     variants per batch (default 5,000)

    Returns
    -------
    betas  : (M,)  per-variant OLS effect size estimate
    ses    : (M,)  standard error of beta
    pvals  : (M,)  two-sided p-value
    \"\"\"
    N, M = G.shape
    if covars is None:
        C = np.ones((N, 1))
    else:
        C = np.column_stack([np.ones(N), covars])

    # Residualise y on covariates once (cheap)
    Q, _  = np.linalg.qr(C, mode='reduced')
    y_r   = y - Q @ (Q.T @ y)
    ss_y  = float(np.dot(y_r, y_r))
    n_df  = N - C.shape[1] - 1

    betas = np.empty(M); ses = np.empty(M)

    for s in range(0, M, chunk):
        e    = min(s + chunk, M)
        G_c  = G[:, s:e].astype(float)
        mu   = np.nanmean(G_c, axis=0)
        ri, ci = np.where(np.isnan(G_c)); G_c[ri, ci] = mu[ci]   # mean-impute
        G_r  = G_c - Q @ (Q.T @ G_c)                              # residualise
        ss_G = (G_r**2).sum(0)
        b    = G_r.T @ y_r / ss_G
        betas[s:e] = b
        rss  = ss_y - b**2 * ss_G
        ses[s:e]   = np.sqrt(np.clip(rss, 0, None) / n_df / ss_G)

    t_stats = betas / (ses + 1e-300)
    pvals   = 2 * stats.t.sf(np.abs(t_stats), df=n_df)
    return betas, ses, pvals
"""

# Provided run_logistic_gwas function (Wald test)
RUN_LOGISTIC_FN = """\
def run_logistic_gwas(y_bin, G, covars=None):
    \"\"\"
    Logistic regression GWAS using the score (Wald) test.
    Fits intercept + covariates as null model, then adds each variant.

    Returns
    -------
    log_ors : (M,) log-odds ratio per variant
    ses     : (M,) standard error
    pvals   : (M,) two-sided Wald p-value
    \"\"\"
    from scipy.special import expit
    from scipy.optimize import minimize

    N, M = G.shape
    col_means = np.nanmean(G, axis=0)
    G_imp = np.where(np.isnan(G), col_means[None, :], G)

    # Null model: fit intercept + covariates
    if covars is None:
        C0 = np.ones((N, 1))
    else:
        C0 = np.column_stack([np.ones(N), covars])
    k0 = C0.shape[1]

    def neg_ll(coef, X, y):
        mu = expit(X @ coef)
        return -np.sum(y * np.log(mu + 1e-15) + (1-y) * np.log(1 - mu + 1e-15))

    res0 = minimize(neg_ll, np.zeros(k0), args=(C0, y_bin), method='L-BFGS-B')
    coef0 = res0.x

    log_ors = np.zeros(M)
    ses     = np.zeros(M)
    pvals   = np.ones(M)

    for j in range(M):
        x_j = G_imp[:, j:j+1]
        X   = np.column_stack([C0, x_j])
        coef_init = np.append(coef0, 0.0)
        res = minimize(neg_ll, coef_init, args=(X, y_bin), method='L-BFGS-B')
        log_ors[j] = res.x[-1]
        # Observed information for the last coefficient (numerical Hessian approx.)
        mu   = expit(X @ res.x)
        W    = mu * (1 - mu)
        info = (x_j.ravel()**2 * W).sum()
        ses[j]   = 1.0 / np.sqrt(info + 1e-15)
        z         = log_ors[j] / ses[j]
        pvals[j]  = 2 * stats.norm.sf(np.abs(z))

    return log_ors, ses, pvals
"""

# Note: run_logistic_gwas is slow on the full dataset; we offer a fast Wald-test version
RUN_LOGISTIC_FAST_FN = """\
def run_logistic_gwas_fast(y_bin, G, covars=None, chunk=5_000):
    \"\"\"
    Fast logistic GWAS: score test (no iterative fitting per variant).
    Fits null model once; processes variants in batches to limit memory.

    Returns
    -------
    log_ors : (M,) approximate log-OR
    pvals   : (M,) score test p-values
    \"\"\"
    from scipy.special import expit
    from scipy.optimize import minimize

    N, M = G.shape
    if covars is None:
        C0 = np.ones((N, 1))
    else:
        C0 = np.column_stack([np.ones(N), covars])
    k0 = C0.shape[1]

    def neg_ll(coef, X, y):
        mu = expit(X @ coef)
        return -np.sum(y * np.log(mu + 1e-15) + (1-y) * np.log(1 - mu + 1e-15))

    res0  = minimize(neg_ll, np.zeros(k0), args=(C0, y_bin), method='L-BFGS-B')
    mu0   = expit(C0 @ res0.x)
    resid = y_bin.astype(float) - mu0
    W0    = mu0 * (1 - mu0)

    # Precompute weighted projection: P_W = C0 @ (C0'WC0)^{-1} C0'W
    WC       = W0[:, None] * C0                              # (N, k0)
    CWWC_inv = np.linalg.inv(C0.T @ WC + 1e-12*np.eye(k0))  # (k0, k0)

    log_ors = np.empty(M); pvals = np.ones(M)

    for s in range(0, M, chunk):
        e    = min(s + chunk, M)
        G_c  = G[:, s:e].astype(float)
        mu   = np.nanmean(G_c, 0); ri, ci = np.where(np.isnan(G_c)); G_c[ri, ci] = mu[ci]
        G_rc = G_c - C0 @ (CWWC_inv @ (WC.T @ G_c))        # weighted residualise
        score  = G_rc.T @ resid
        var_sc = (G_rc**2 * W0[:, None]).sum(0)
        z      = score / np.sqrt(var_sc + 1e-15)
        pvals[s:e]   = 2 * stats.norm.sf(np.abs(z))
        log_ors[s:e] = score / (var_sc + 1e-15)

    return log_ors, pvals
"""

# Provided HWE mid-p exact test
HWE_MIDP_FN = """\
def compute_hwe_midp(G):
    \"\"\"
    Vectorised heterozygosity-based chi-squared test for HWE.
    Uses the heterozygote deviation formula: chi2 = n*(obs_het - exp_het)^2 / exp_het.
    Returns p-values for each variant (M,).
    \"\"\"
    G_int  = np.where(np.isnan(G), -1, G).astype(int)
    n_samp = (G_int >= 0).sum(0).astype(float)
    n_AA   = (G_int == 0).sum(0).astype(float)
    n_AB   = (G_int == 1).sum(0).astype(float)
    # p_hat = REF allele freq; exp_het = 2*p*(1-p) is symmetric
    p_hat   = (2*n_AA + n_AB) / (2*n_samp + 1e-15)
    obs_het = n_AB / (n_samp + 1e-15)
    exp_het = 2 * p_hat * (1 - p_hat)
    chi2    = n_samp * (obs_het - exp_het)**2 / (exp_het + 1e-15)
    return stats.chi2.sf(chi2, df=1)
"""



# ─── Session 1 cells ──────────────────────────────────────────────────────────

S1_TITLE = """\
# Session 1: Introduction to GWAS — Practical

**Timing**: This practical is designed for ~45 minutes.
- Parts 1–3 should take ~30–35 minutes.
- Challenge questions are for fast finishers.

**Dataset**: Simulated GWAS data for ~100,000 variants with realistic block LD across chr1 (1–250 Mb).
- Continuous trait: simulated liability phenotype (h² ≈ 0.035, MAF-dependent spike-and-slab prior).
- Binary trait: derived from the continuous liability via a threshold model (~10% cases).

**Tip**: If you get stuck on any exercise, hints are in the comments. Solutions are in `answers.ipynb`.
"""

S1_PHENOTYPE_PLOTS = """\
# ── Phenotype and covariate distributions ────────────────────────────────────
# Three traits to characterise:
#   y_cont — continuous (h²≈0.035, few large-effect causal variants)
#   y_poly — continuous (h²≈0.02, fully polygenic — every variant has a tiny effect)
#   y_bin  — binary (~10% cases, derived from y_cont via liability threshold)
fig, axes = plt.subplots(2, 3, figsize=(14, 7))

for ax, y, col, lbl in zip(
        [axes[0,0], axes[0,1]],
        [y_cont, y_poly],
        ['steelblue', '#59a14f'],
        ['Continuous (h²≈0.035)', 'Polygenic (h²≈0.02)']):
    ax.hist(y, bins=60, color=col, edgecolor='white', linewidth=0.3)
    ax.set_xlabel('Standardised phenotype'); ax.set_ylabel('Individuals')
    ax.set_title(lbl)

axes[0,2].bar(['Control (0)', 'Case (1)'],
              [(y_bin==0).sum(), (y_bin==1).sum()],
              color=['#4e79a7', '#e15759'], width=0.5)
axes[0,2].set_ylabel('Count')
axes[0,2].set_title(f'Binary trait ({100*y_bin.mean():.0f}% cases)')

axes[1,0].scatter(y_cont[::5], y_poly[::5], s=2, alpha=0.3, color='grey')
axes[1,0].set_xlabel('y_cont'); axes[1,0].set_ylabel('y_poly')
rho_cp = np.corrcoef(y_cont, y_poly)[0,1]
axes[1,0].set_title(f'y_cont vs y_poly  (r={rho_cp:.2f})')

axes[1,1].hist(age, bins=40, color='#f28e2b', edgecolor='white', linewidth=0.3)
axes[1,1].set_xlabel('Age'); axes[1,1].set_title('Age distribution')

axes[1,2].boxplot([y_cont[y_bin==0], y_cont[y_bin==1]], labels=['Control', 'Case'],
                  patch_artist=True,
                  boxprops=dict(facecolor='#4e79a7'), medianprops=dict(color='white'))
axes[1,2].set_ylabel('y_cont'); axes[1,2].set_title('y_cont by case/control status')

plt.suptitle('Dataset overview — three phenotypes', fontsize=13, y=1.01)
plt.tight_layout(); plt.show()
print(f"N = {N:,} individuals  |  M_raw = {M_raw:,} variants (pre-QC)  |  "
      f"Cases: {y_bin.sum():,} ({100*y_bin.mean():.0f}%)")
print(f"Pearson r(y_cont, y_poly) = {rho_cp:.3f}  "
      f"— in Session 2 you will test which trait pairs are genetically correlated.")
"""

S1_PART1_MD = """\
## Part 1: GWAS Quality Control

Before running a GWAS, we need to filter out low-quality variants.
Standard filters remove variants with:
- **High missingness** (call rate < 95%, i.e., >5% samples missing a genotype)
- **Low MAF** (minor allele frequency < 1%; rare variants have low power)
- **HWE violation** (excess homozygosity often indicates genotyping error)

In the following exercises, `G_raw` is the raw genotype matrix (N × M_raw).
Genotypes are coded as 0/1/2 (copies of the effect allele); `NaN` = missing.
"""

S1_EX11_STUDENT = """\
# ── Exercise 1.1: Per-variant missingness ─────────────────────────────────────
# Missingness rate = fraction of samples with a missing genotype (NaN).
# A variant "fails" QC if its missingness rate exceeds 5%.

# YOUR CODE HERE
# Hint: np.isnan(G_raw).mean(axis=0) gives the fraction of NaN per column
miss_rate = ???                         # shape (M_raw,)

# ─────────────────────────────────────────────────────────────────────────────
print(f"Variants with missingness > 5%: {(miss_rate > 0.05).sum():,}")

fig, ax = plt.subplots(figsize=(7, 3))
ax.hist(miss_rate, bins=50, color='steelblue', edgecolor='white', linewidth=0.4)
ax.axvline(0.05, color='red', linestyle='--', label='5% threshold')
ax.set_xlabel('Missingness rate'); ax.set_ylabel('Variants (log scale)')
ax.set_yscale('log'); ax.set_title('Per-variant missingness distribution'); ax.legend()
plt.tight_layout(); plt.show()
"""

S1_EX11_SOL = """\
miss_rate = np.isnan(G_raw).mean(axis=0)

print(f"Variants with missingness > 5%: {(miss_rate > 0.05).sum():,}")

fig, ax = plt.subplots(figsize=(7, 3))
ax.hist(miss_rate, bins=50, color='steelblue', edgecolor='white', linewidth=0.4)
ax.axvline(0.05, color='red', linestyle='--', label='5% threshold')
ax.set_xlabel('Missingness rate'); ax.set_ylabel('Variants (log scale)')
ax.set_yscale('log'); ax.set_title('Per-variant missingness distribution'); ax.legend()
plt.tight_layout(); plt.show()
"""

S1_EX12_STUDENT = """\
# ── Exercise 1.2: Minor Allele Frequency ─────────────────────────────────────
# MAF = frequency of the less common allele.
# Genotype dosage 0/1/2 → ALT allele frequency = mean(G) / 2.

# YOUR CODE HERE
# Hint: use np.nanmean to ignore NaN; compute frequency of ALT allele, then take the minor
alt_freq = ???                          # ALT allele frequency, shape (M_raw,)
maf      = ???                          # MAF = min(alt_freq, 1 - alt_freq)

# ─────────────────────────────────────────────────────────────────────────────
print(f"Variants with MAF < 1%:  {(maf < 0.01).sum():,}")
print(f"Variants with MAF < 5%:  {(maf < 0.05).sum():,}")

fig, ax = plt.subplots(figsize=(7, 3))
ax.hist(maf, bins=60, color='teal', edgecolor='white', linewidth=0.4)
ax.axvline(0.01, color='red',    linestyle='--', label='1% threshold')
ax.axvline(0.05, color='orange', linestyle='--', label='5% threshold')
ax.set_xlabel('MAF'); ax.set_ylabel('Variants')
ax.set_title('MAF distribution (pre-QC)'); ax.legend()
plt.tight_layout(); plt.show()
"""

S1_EX12_SOL = """\
alt_freq = np.nanmean(G_raw, axis=0) / 2
maf      = np.minimum(alt_freq, 1 - alt_freq)

print(f"Variants with MAF < 1%:  {(maf < 0.01).sum():,}")
print(f"Variants with MAF < 5%:  {(maf < 0.05).sum():,}")

fig, ax = plt.subplots(figsize=(7, 3))
ax.hist(maf, bins=60, color='teal', edgecolor='white', linewidth=0.4)
ax.axvline(0.01, color='red',    linestyle='--', label='1% threshold')
ax.axvline(0.05, color='orange', linestyle='--', label='5% threshold')
ax.set_xlabel('MAF'); ax.set_ylabel('Variants')
ax.set_title('MAF distribution (pre-QC)'); ax.legend()
plt.tight_layout(); plt.show()
"""

S1_EX13_MD = """\
### Exercise 1.3: Hardy-Weinberg Equilibrium (HWE) test

Under HWE, genotype counts follow: $n_{AA} \\approx n p^2$, $n_{AB} \\approx n 2pq$, $n_{BB} \\approx n q^2$,
where $p$ = ALT allele frequency and $q = 1-p$.

Violations can indicate genotyping errors (excess homozygosity is most common).
Standard GWAS threshold: HWE $p < 10^{-6}$ (remove variants that fail).

We **provide** a vectorised HWE test (`compute_hwe_midp`, based on the heterozygote deviation) —
run it below and use it for QC. In Challenge 1 you'll implement the classic 3-class chi-squared
test yourself and compare.
"""

# Provided HWE test cell (regular Part-1 flow): students just run it and use hwe_pvals for QC.
S1_PROVIDE_HWE = """\
# ── Provided HWE test (heterozygote-deviation form) — run and use for QC ─────
""" + HWE_MIDP_FN + """
hwe_pvals = compute_hwe_midp(G_raw)
print(f"Variants failing HWE (p < 1e-6): {(hwe_pvals < 1e-6).sum():,}")

fig, ax = plt.subplots(figsize=(7, 3))
ax.hist(-np.log10(hwe_pvals + 1e-15), bins=60, color='salmon', edgecolor='white', linewidth=0.4)
ax.axvline(-np.log10(1e-6), color='red', linestyle='--', label='p = 1e-6')
ax.set_xlabel('-log10(HWE p)'); ax.set_ylabel('Variants (log scale)'); ax.set_yscale('log')
ax.set_title('HWE test p-values'); ax.legend()
plt.tight_layout(); plt.show()
"""

S1_QC_APPLY = """\
# ── Apply QC filters ──────────────────────────────────────────────────────────
# Standard GWAS filters (thresholds given; no coding required here)
MISS_THRESH = 0.05     # remove variants with >5% missing
MAF_THRESH  = 0.01     # remove variants with MAF < 1%
HWE_THRESH  = 1e-6     # remove variants with HWE p < 1e-6

pass_miss = miss_rate < MISS_THRESH
pass_maf  = maf       > MAF_THRESH
pass_hwe  = hwe_pvals > HWE_THRESH   # provided HWE test from Exercise 1.3

qc_pass = pass_miss & pass_maf & pass_hwe

G_qc  = np.where(np.isnan(G_raw[:, qc_pass]), 0, G_raw[:, qc_pass])  # impute remaining NaN
pos_qc   = pos[qc_pass]
rsids_qc = rsids[qc_pass]
M_qc  = qc_pass.sum()

print("QC Summary")
print(f"  Pre-QC:               {M_raw:>7,} variants")
print(f"  After missingness QC: {pass_miss.sum():>7,} variants")
print(f"  After MAF QC:         {pass_maf.sum():>7,} variants")
print(f"  After HWE QC:         {pass_hwe.sum():>7,} variants")
print(f"  After all filters:    {M_qc:>7,} variants  ← G_qc")
# (G_raw is kept so the HWE chi-squared challenge can re-run on the raw genotypes.)
"""

S1_PART2_MD = """\
## Part 2: Running GWAS — Continuous Trait

We'll use a vectorised OLS function (provided) that runs all M regressions efficiently.
The model for each variant $j$ is:

$$y_i = \\mu + \\beta_j \\cdot G_{ij} + \\gamma_1 \\cdot \\text{age}_i + \\gamma_2 \\cdot \\text{sex}_i + \\varepsilon_i$$

where $\\hat{\\beta}_j$ is the per-allele effect size on the standardised phenotype (in SD units).
"""

S1_PROVIDE_GWAS_FN = RUN_GWAS_FN + "\nprint('run_gwas() ready.')"

S1_EX21_STUDENT = """\
# ── Exercise 2.1: GWAS without covariates ────────────────────────────────────
# Run GWAS for the continuous trait without adjusting for age or sex.

# YOUR CODE HERE — call run_gwas with appropriate arguments
betas_nocov, ses_nocov, pvals_nocov = run_gwas(???, ???)

# ─────────────────────────────────────────────────────────────────────────────
n_sig = (pvals_nocov < 5e-8).sum()
print(f"Genome-wide significant hits (p < 5e-8): {n_sig:,}")
print(f"Min p-value: {pvals_nocov.min():.2e}")
"""

S1_EX21_SOL = """\
betas_nocov, ses_nocov, pvals_nocov = run_gwas(y_cont, G_qc)


n_sig = (pvals_nocov < 5e-8).sum()
print(f"Genome-wide significant hits (p < 5e-8): {n_sig:,}")
print(f"Min p-value: {pvals_nocov.min():.2e}")
"""

S1_EX22_STUDENT = """\
# ── Exercise 2.2: GWAS with covariates ───────────────────────────────────────
# Now include age and sex as covariates.
# The covariate matrix should have shape (N, 2).

# YOUR CODE HERE
covars = np.column_stack([???])         # age and sex (standardise age!)
betas_cov, ses_cov, pvals_cov = run_gwas(???, ???, ???)

# ─────────────────────────────────────────────────────────────────────────────
# Compare lambda GC before and after covariate adjustment
def lambda_gc(pvals):
    chi2_obs = stats.chi2.isf(np.clip(pvals, 1e-300, 1), df=1)
    return np.median(chi2_obs) / stats.chi2.median(df=1)

print(f"Lambda GC without covariates: {lambda_gc(pvals_nocov):.3f}")
print(f"Lambda GC with covariates:    {lambda_gc(pvals_cov):.3f}")
print("Q: Did adding covariates change lambda GC? What does lambda GC > 1.0 imply?")
"""

S1_EX22_SOL = """\
covars = np.column_stack([(age - age.mean()) / age.std(), sex])
betas_cov, ses_cov, pvals_cov = run_gwas(y_cont, G_qc, covars)

def lambda_gc(pvals):
    chi2_obs = stats.chi2.isf(np.clip(pvals, 1e-300, 1), df=1)
    return np.median(chi2_obs) / stats.chi2.median(df=1)

print(f"Lambda GC without covariates: {lambda_gc(pvals_nocov):.3f}")
print(f"Lambda GC with covariates:    {lambda_gc(pvals_cov):.3f}")
"""

S1_EX23_STUDENT = """\
# ── Exercise 2.3: Interpreting beta — strip plot by genotype ──────────────────
# The OLS beta estimates the per-allele shift in phenotype.
# For an additive variant: E[y | g] ≈ μ + g × beta.
# A strip plot makes the genotype-phenotype trend visible on individual level.
import seaborn as sns

j_top    = np.argmin(pvals_cov)
beta_top = betas_cov[j_top]
geno     = G_qc[:, j_top].astype(int)
print(f"Lead variant: {rsids_qc[j_top]}  pos={pos_qc[j_top]:,} kbp  "
      f"beta={beta_top:.4f}  p={pvals_cov[j_top]:.2e}")

# YOUR CODE HERE
means  = ???   # mean phenotype for each of the 3 genotype classes (0, 1, 2)
counts = ???   # number of individuals per class

# ─────────────────────────────────────────────────────────────────────────────
_df = pd.DataFrame({'Genotype': geno, 'Phenotype': y_cont})
_df['Genotype'] = _df['Genotype'].map({0:'AA (0)', 1:'AB (1)', 2:'BB (2)'})

fig, ax = plt.subplots(figsize=(6, 4))
sns.stripplot(data=_df.sample(min(2000, len(_df)), random_state=0),
              x='Genotype', y='Phenotype', alpha=0.25, size=2, jitter=True,
              palette=['#4e79a7','#f28e2b','#e15759'], ax=ax)
ax.plot([0,1,2], means, color='black', linewidth=1.5, zorder=4)
ax.errorbar([0,1,2], means, fmt='_', color='black', markersize=20, linewidth=2,
            capsize=0, zorder=5)
ax.set_ylabel('Phenotype (standardised SD)')
ax.set_title(f'Lead SNP — per-allele shift ≈ {beta_top:.3f} SD')
plt.tight_layout(); plt.show()

print(f"Expected BB−AA ≈ 2 × beta = {2*beta_top:.4f} SD")
print(f"Observed BB−AA:              {means[2]-means[0]:.4f} SD")
print("Q: Is the heterozygote (AB) midway between the two homozygotes? What does that imply?")
"""

S1_EX23_SOL = """\
import seaborn as sns
j_top    = np.argmin(pvals_cov)
beta_top = betas_cov[j_top]
geno     = G_qc[:, j_top].astype(int)
means    = np.array([y_cont[geno == g].mean() for g in range(3)])
counts   = np.array([(geno == g).sum()        for g in range(3)])

_df = pd.DataFrame({'Genotype': geno, 'Phenotype': y_cont})
_df['Genotype'] = _df['Genotype'].map({0:'AA (0)', 1:'AB (1)', 2:'BB (2)'})
fig, ax = plt.subplots(figsize=(6, 4))
sns.stripplot(data=_df.sample(min(2000, len(_df)), random_state=0),
              x='Genotype', y='Phenotype', alpha=0.25, size=2, jitter=True,
              palette=['#4e79a7','#f28e2b','#e15759'], ax=ax)
ax.plot([0,1,2], means, color='black', linewidth=1.5, zorder=4)
ax.errorbar([0,1,2], means, fmt='_', color='black', markersize=20, linewidth=2, zorder=5)
ax.set_ylabel('Phenotype (standardised SD)')
ax.set_title(f'Lead SNP — per-allele shift ≈ {beta_top:.3f} SD')
plt.tight_layout(); plt.show()
print(f"Expected BB−AA ≈ 2 × beta = {2*beta_top:.4f} SD")
print(f"Observed BB−AA:              {means[2]-means[0]:.4f} SD")
"""

S1_PART3_MD = """\
## Part 3: GWAS — Binary Trait (Liability Threshold Model)

For binary (case/control) phenotypes, logistic regression is standard.
The model gives a **log-odds ratio** (log-OR) per allele.

The binary phenotype `y_bin` (1=case, 0=control) was derived from the continuous phenotype
via a **liability threshold**: individuals above the 90th percentile of liability are cases (~10% prevalence).
This is the standard liability threshold model for complex diseases.

We provide `run_logistic_gwas_fast()` — a score-test approximation that runs quickly.
"""

S1_PROVIDE_LOGISTIC_FN = RUN_LOGISTIC_FAST_FN + "\nprint('run_logistic_gwas_fast() ready.')"

S1_EX31_STUDENT = """\
# ── Exercise 3.1: Logistic GWAS for CAD ──────────────────────────────────────
# Run the logistic GWAS for CAD using the fast score test.

# YOUR CODE HERE
covars = np.column_stack([???])    # same covariates as before
log_ors, pvals_bin = run_logistic_gwas_fast(???, ???, ???)

# ─────────────────────────────────────────────────────────────────────────────
n_sig_bin = (pvals_bin < 5e-8).sum()
print(f"Genome-wide significant binary-trait hits: {n_sig_bin:,}")

# Compare LDL and CAD p-values at the same variants
top_ldl_idx = np.argsort(pvals_cov)[:20]
fig, ax = plt.subplots(figsize=(6, 5))
ax.scatter(-np.log10(pvals_cov[top_ldl_idx]+1e-15),
           -np.log10(pvals_bin[top_ldl_idx]+1e-15),
           alpha=0.8, s=40, color='steelblue')
ax.set_xlabel('-log10(p) continuous trait')
ax.set_ylabel('-log10(p) binary trait')
ax.set_title('Top 20 continuous-trait hits: continuous vs binary p-values')
for i, j in enumerate(top_ldl_idx[:5]):
    ax.annotate(f"rs{j}", ((-np.log10(pvals_cov[j]+1e-15)),
                            (-np.log10(pvals_bin[j]+1e-15))), fontsize=7)
plt.tight_layout(); plt.show()
print("Q: Do the same variants drive both continuous and binary trait associations?")
"""

S1_EX31_SOL = """\
covars = np.column_stack([(age - age.mean()) / age.std(), sex])
log_ors, pvals_bin = run_logistic_gwas_fast(y_bin, G_qc, covars)

n_sig_bin = (pvals_bin < 5e-8).sum()
print(f"Genome-wide significant binary-trait hits: {n_sig_bin:,}")

top_ldl_idx = np.argsort(pvals_cov)[:20]
fig, ax = plt.subplots(figsize=(6, 5))
ax.scatter(-np.log10(pvals_cov[top_ldl_idx]+1e-15),
           -np.log10(pvals_bin[top_ldl_idx]+1e-15),
           alpha=0.8, s=40, color='steelblue')
ax.set_xlabel('-log10(p) continuous'); ax.set_ylabel('-log10(p) binary')
ax.set_title('Top 20 continuous-trait hits: continuous vs binary p-values')
plt.tight_layout(); plt.show()
"""

S1_CQ_MD = """\
---

## Challenge Questions

These questions are for fast finishers and are not required in the 45-minute session.
They connect directly to ideas from the lecture.
"""

S1_CQHWE_MD = """\
### Challenge 1: HWE chi-squared test from scratch

For QC we used the provided heterozygosity-based test (`compute_hwe_midp`). The classic HWE test
is a **3-class chi-squared** comparing observed genotype counts (AA/AB/BB) to their HWE
expectations $n p^2, 2npq, nq^2$. Implement it yourself and compare your p-values to the provided
test — where do they agree, and where do they diverge?
"""

S1_CQHWE_STUDENT = """\
# Challenge 1: HWE chi-squared test (vectorised), compared to the provided test
def compute_hwe_chisq(G):
    \"\"\"Vectorised 3-class chi-squared HWE test; returns p-values of shape (M,).\"\"\"
    G_int  = np.where(np.isnan(G), -1, G).astype(int)
    n_samp = (G_int >= 0).sum(0).astype(float)   # non-missing count per variant
    n_AA   = (G_int == 0).sum(0).astype(float)
    n_AB   = (G_int == 1).sum(0).astype(float)
    n_BB   = (G_int == 2).sum(0).astype(float)

    # YOUR CODE — all vectorised over all M variants simultaneously
    # ALT allele frequency: each copy of the ALT allele (BB = 2, AB = 1) contributes
    p      = ???   # (2*n_BB + n_AB) / (2*n_samp)

    # Expected genotype counts under HWE: P(0 ALT) = (1-p)², P(1 ALT) = 2p(1-p), P(2 ALT) = p²
    exp_AA = ???
    exp_AB = ???
    exp_BB = ???

    # Chi-squared: sum (obs - exp)² / exp, 1 degree of freedom
    chi2   = ???

    return stats.chi2.sf(chi2, df=1)

hwe_chisq = compute_hwe_chisq(G_raw)

# Compare to the provided heterozygosity test (hwe_pvals, from Exercise 1.3)
fig, ax = plt.subplots(figsize=(5, 5))
ax.scatter(-np.log10(hwe_pvals + 1e-15), -np.log10(hwe_chisq + 1e-15),
           s=1, alpha=0.2, color='steelblue', rasterized=True)
lim = max((-np.log10(hwe_pvals + 1e-15)).max(), (-np.log10(hwe_chisq + 1e-15)).max()) * 1.05
ax.plot([0, lim], [0, lim], 'r--', linewidth=1)
ax.set_xlabel('-log10(p) provided het test'); ax.set_ylabel('-log10(p) your chi-squared')
ax.set_title('HWE: chi-squared vs heterozygosity test'); plt.tight_layout(); plt.show()

print(f"Fail HWE (chi-sq, p<1e-6):   {(hwe_chisq < 1e-6).sum():,}")
print(f"Fail HWE (het test, p<1e-6): {(hwe_pvals < 1e-6).sum():,}")
print("Q: Where do the two tests agree? Where do they differ, and why?")
"""

S1_CQHWE_SOL = """\
def compute_hwe_chisq(G):
    G_int  = np.where(np.isnan(G), -1, G).astype(int)
    n_samp = (G_int >= 0).sum(0).astype(float)
    n_AA   = (G_int == 0).sum(0).astype(float)
    n_AB   = (G_int == 1).sum(0).astype(float)
    n_BB   = (G_int == 2).sum(0).astype(float)
    p      = (2*n_BB + n_AB) / (2*n_samp + 1e-15)
    exp_AA = n_samp * (1-p)**2
    exp_AB = n_samp * 2*p*(1-p)
    exp_BB = n_samp * p**2
    chi2   = ((n_AA-exp_AA)**2/(exp_AA+1e-8) +
              (n_AB-exp_AB)**2/(exp_AB+1e-8) +
              (n_BB-exp_BB)**2/(exp_BB+1e-8))
    return stats.chi2.sf(chi2, df=1)

hwe_chisq = compute_hwe_chisq(G_raw)
fig, ax = plt.subplots(figsize=(5, 5))
ax.scatter(-np.log10(hwe_pvals + 1e-15), -np.log10(hwe_chisq + 1e-15),
           s=1, alpha=0.2, color='steelblue', rasterized=True)
lim = max((-np.log10(hwe_pvals + 1e-15)).max(), (-np.log10(hwe_chisq + 1e-15)).max()) * 1.05
ax.plot([0, lim], [0, lim], 'r--', linewidth=1)
ax.set_xlabel('-log10(p) provided het test'); ax.set_ylabel('-log10(p) your chi-squared')
ax.set_title('HWE: chi-squared vs heterozygosity test'); plt.tight_layout(); plt.show()
print(f"Fail HWE (chi-sq, p<1e-6):   {(hwe_chisq < 1e-6).sum():,}")
print(f"Fail HWE (het test, p<1e-6): {(hwe_pvals < 1e-6).sum():,}")
"""

S1_CQ1_MD = """\
### Challenge 2: Additive, Dominant, and Recessive Models

The standard GWAS uses an **additive** model: genotype is encoded 0/1/2 (copies of ALT allele),
so the heterozygote AB is midway between AA and BB.

But what if the true effect is **dominant** (AB ≈ BB, one copy is enough) or
**recessive** (AB ≈ AA, two copies are needed)?

Recoding:
- **Dominant**: 0 → 0, 1 → 1, 2 → 1  (any ALT copy counts)
- **Recessive**: 0 → 0, 1 → 0, 2 → 1  (only homozygous ALT counts)

Two variants in this dataset have non-additive effects — one dominant, one recessive.
They may not be the genome-wide lead hit under the additive model!
"""

S1_CQ1_STUDENT = """\
# Challenge 2: Find the non-additive loci
import seaborn as sns

# YOUR CODE HERE
# Recode G_qc under dominant and recessive models
G_dom = ???   # 0→0, 1→1, 2→1
G_rec = ???   # 0→0, 1→0, 2→1

covars = np.column_stack([(age - age.mean()) / age.std(), sex])
betas_dom, _, pvals_dom = run_gwas(y_cont, G_dom, covars)
betas_rec, _, pvals_rec = run_gwas(y_cont, G_rec, covars)

# Find the best hit under each non-additive model
j_dom = np.argmin(pvals_dom)
j_rec = np.argmin(pvals_rec)
print(f"Top dominant hit: {rsids_qc[j_dom]}  pos={pos_qc[j_dom]:,} kbp  p={pvals_dom[j_dom]:.2e}")
print(f"Top recessive hit: {rsids_qc[j_rec]}  pos={pos_qc[j_rec]:,} kbp  p={pvals_rec[j_rec]:.2e}")

# Strip plots for each — does the AB group look more like AA or BB?
fig, axes = plt.subplots(1, 2, figsize=(10, 4))
for ax, j, title in zip(axes, [j_dom, j_rec], ['Putative dominant', 'Putative recessive']):
    geno = G_qc[:, j].astype(int)
    _df  = pd.DataFrame({'Genotype': geno, 'Phenotype': y_cont})
    _df['Genotype'] = _df['Genotype'].map({0:'AA', 1:'AB', 2:'BB'})
    sns.stripplot(data=_df.sample(min(2000, len(_df)), random_state=0),
                  x='Genotype', y='Phenotype', alpha=0.25, size=2, jitter=True,
                  palette=['#4e79a7','#f28e2b','#e15759'], ax=ax)
    means = [y_cont[geno==g].mean() for g in range(3)]
    ax.plot([0,1,2], means, color='black', linewidth=1.5, zorder=4)
    ax.errorbar([0,1,2], means, fmt='_', color='black', markersize=20, linewidth=2, zorder=5)
    ax.set_title(title); ax.set_ylabel('Phenotype')
plt.tight_layout(); plt.show()
print("Q: For the dominant hit, is AB closer to AA or BB? What about the recessive hit?")
"""

S1_CQ1_SOL = """\
import seaborn as sns
G_dom = np.where(G_qc >= 1, 1, 0).astype(float)
G_rec = np.where(G_qc == 2, 1, 0).astype(float)

covars = np.column_stack([(age - age.mean()) / age.std(), sex])
betas_dom, _, pvals_dom = run_gwas(y_cont, G_dom, covars)
betas_rec, _, pvals_rec = run_gwas(y_cont, G_rec, covars)

j_dom = np.argmin(pvals_dom); j_rec = np.argmin(pvals_rec)
print(f"Top dominant hit: {rsids_qc[j_dom]}  p={pvals_dom[j_dom]:.2e}")
print(f"Top recessive hit: {rsids_qc[j_rec]}  p={pvals_rec[j_rec]:.2e}")
# True injected loci: dom_idx_qc, rec_idx_qc
print(f"True dominant locus was at col {dom_idx_qc}: p_additive={pvals_cov[dom_idx_qc]:.2e}, p_dominant={pvals_dom[dom_idx_qc]:.2e}")
print(f"True recessive locus was at col {rec_idx_qc}: p_additive={pvals_cov[rec_idx_qc]:.2e}, p_recessive={pvals_rec[rec_idx_qc]:.2e}")

fig, axes = plt.subplots(1, 2, figsize=(10, 4))
for ax, j, title in zip(axes, [dom_idx_qc, rec_idx_qc], ['True dominant locus', 'True recessive locus']):
    geno = G_qc[:, j].astype(int)
    _df  = pd.DataFrame({'Genotype': geno, 'Phenotype': y_cont})
    _df['Genotype'] = _df['Genotype'].map({0:'AA', 1:'AB', 2:'BB'})
    sns.stripplot(data=_df.sample(min(2000, len(_df)), random_state=0),
                  x='Genotype', y='Phenotype', alpha=0.25, size=2, jitter=True,
                  palette=['#4e79a7','#f28e2b','#e15759'], ax=ax)
    means = [y_cont[geno==g].mean() for g in range(3)]
    ax.plot([0,1,2], means, color='black', linewidth=1.5, zorder=4)
    ax.errorbar([0,1,2], means, fmt='_', color='black', markersize=20, linewidth=2, zorder=5)
    ax.set_title(title); ax.set_ylabel('Phenotype')
plt.tight_layout(); plt.show()
"""

S1_LZ_MD = """\
### Challenge 3: Manual LocusZoom plot

A **LocusZoom plot** shows $-\\log_{10}(p)$ vs. position for a region around a hit, with points
**coloured by LD** ($r^2$) with the lead variant. It reveals the LD structure that produces the
"tower" you'll see in a Manhattan plot (Session 2) — neighbouring variants are associated only
because they are correlated with the causal one.

We use the covariate-adjusted continuous-trait GWAS (`pvals_cov`) and the genotypes `G_qc`.
"""

S1_LZ_STUDENT = """\
# Challenge 3: Manual LocusZoom plot for the lead locus (continuous trait)

# Region: ±1 Mb around the lead variant
j_lead    = np.argmin(pvals_cov)
pos_lead  = pos_qc[j_lead]
region_mask = np.abs(pos_qc - pos_lead) < 1000   # ±1000 kbp = ±1 Mb

print(f"Lead variant: {rsids_qc[j_lead]}  pos={pos_lead:,} kbp  p={pvals_cov[j_lead]:.2e}")
print(f"Variants in region: {region_mask.sum():,}")

# YOUR CODE HERE
# Compute r² between each variant in the region and the lead variant.
# Hint: r = np.corrcoef(g_lead, G_qc[:, k])[0,1]; r² = r**2
g_lead = G_qc[:, j_lead]
r2 = ???    # shape (region_mask.sum(),)

cmap = plt.cm.RdYlBu_r   # red = high LD, blue = low LD
norm = mcolors.Normalize(vmin=0, vmax=1)
fig, ax = plt.subplots(figsize=(12, 4))
sc = ax.scatter(pos_qc[region_mask] / 1000,
                -np.log10(pvals_cov[region_mask] + 1e-300),
                c=r2, cmap=cmap, norm=norm, s=20, zorder=3)
ax.scatter([pos_lead/1000], [-np.log10(pvals_cov[j_lead]+1e-300)],
           s=100, marker='D', color='black', zorder=5, label='Lead SNP')
plt.colorbar(sc, ax=ax, label=r'$r^2$ with lead SNP')
ax.axhline(-np.log10(5e-8), color='red', linestyle='--', linewidth=1, label='5×10⁻⁸')
ax.set_xlabel('Position on chr1 (Mb)'); ax.set_ylabel(r'$-\\log_{10}(p)$')
ax.set_title('LocusZoom: lead locus on chr1 (±1 Mb)')
ax.legend(fontsize=8); plt.tight_layout(); plt.show()
print("Q: Do the high-r² (red) variants form the tower? What happens to p as r² drops?")
"""

S1_LZ_SOL = """\
j_lead = np.argmin(pvals_cov); pos_lead = pos_qc[j_lead]
region_mask = np.abs(pos_qc - pos_lead) < 1000
g_lead = G_qc[:, j_lead]
G_region = G_qc[:, region_mask]
r2 = np.array([np.corrcoef(g_lead, G_region[:, k])[0,1]**2 for k in range(G_region.shape[1])])

cmap = plt.cm.RdYlBu_r; norm = mcolors.Normalize(vmin=0, vmax=1)
fig, ax = plt.subplots(figsize=(12, 4))
sc = ax.scatter(pos_qc[region_mask]/1000, -np.log10(pvals_cov[region_mask]+1e-300),
                c=r2, cmap=cmap, norm=norm, s=20, zorder=3)
ax.scatter([pos_lead/1000], [-np.log10(pvals_cov[j_lead]+1e-300)],
           s=100, marker='D', color='black', zorder=5, label='Lead SNP')
plt.colorbar(sc, ax=ax, label=r'$r^2$ with lead SNP')
ax.axhline(-np.log10(5e-8), color='red', linestyle='--', linewidth=1)
ax.set_xlabel('Position on chr1 (Mb)'); ax.set_ylabel(r'$-\\log_{10}(p)$')
ax.set_title('LocusZoom: lead locus (±1 Mb)'); ax.legend(fontsize=8)
plt.tight_layout(); plt.show()
"""

S1_CQ2_MD = """\
### Challenge 4: Drosophila Linkage Analysis (Hard!)

*(Inspired by Sturtevant, 1913 — the first genetic map.)*

We have 2,000 fly offspring from a test cross:
- **Mother**: heterozygous for 6 X-linked traits (carrier) + 2 autosomal traits
- **Father**: hemizygous wild-type

The dataset `fly_df` has columns: `sex` (1=male, 0=female) and 8 trait columns.
You are told there are 6 X-linked traits and 2 autosomal traits — but the column names
don't tell you which is which.

**Your tasks**:
1. Identify which traits are X-linked (hint: look at trait frequency in males vs females).
2. For the X-linked traits, compute pairwise recombination frequencies.
3. Use the recombination frequencies to infer the order and spacing of the 6 genes on the chromosome.
"""

S1_CQ2_STUDENT = """\
# Challenge 4, Part A: Identify X-linked traits
# For X-linked recessive traits in a test cross: males show the trait ~50% of the time,
# females show it ~0% of the time (they are carriers).
# For autosomal traits: frequency is similar in males and females.

fly_df.head()
"""

S1_CQ2B_STUDENT = """\
# Challenge 4, Part A (continued)
trait_cols = [c for c in fly_df.columns if c.startswith('trait_')]

# YOUR CODE HERE
# For each trait, compute the mean (= frequency of showing the trait) in males and females separately
# Hint: fly_df.groupby('sex')[trait_cols].mean()
freq_by_sex = ???

print(freq_by_sex.T)
print("\\nWhich traits are X-linked? (large difference between males and females)")
x_linked_traits = ???   # list of column names that are X-linked
"""

S1_CQ2B_SOL = """\
trait_cols = [c for c in fly_df.columns if c.startswith('trait_')]
freq_by_sex = fly_df.groupby('sex')[trait_cols].mean()
print(freq_by_sex.T)
# X-linked traits have freq ~0.5 in males and ~0 in females
x_linked_traits = freq_by_sex.columns[(freq_by_sex.loc[1] > 0.3) & (freq_by_sex.loc[0] < 0.1)].tolist()
print(f"\\nX-linked traits: {x_linked_traits}")
"""

S1_CQ2C_STUDENT = """\
# Challenge 4, Part B: Pairwise recombination frequencies
# For X-linked traits, recombination frequency between genes A and B =
# fraction of MALE offspring where the phenotype for A DIFFERS from phenotype for B.
# (Non-recombinants have all traits in coupling; recombinants show a 'break' in the pattern.)

males = fly_df[fly_df['sex'] == 1]

# YOUR CODE HERE
# Compute the pairwise recombination frequency matrix (n_X × n_X)
# Hint: for traits i and j, recomb_freq[i,j] = fraction of males where trait_i != trait_j
n_x = len(x_linked_traits)
recomb_freq = np.zeros((n_x, n_x))
for i, ti in enumerate(x_linked_traits):
    for j, tj in enumerate(x_linked_traits):
        recomb_freq[i, j] = ???

print("Pairwise recombination frequencies (males only):")
pd.DataFrame(recomb_freq, index=x_linked_traits, columns=x_linked_traits).round(3)
"""

S1_CQ2C_SOL = """\
males = fly_df[fly_df['sex'] == 1]
n_x = len(x_linked_traits)
recomb_freq = np.zeros((n_x, n_x))
for i, ti in enumerate(x_linked_traits):
    for j, tj in enumerate(x_linked_traits):
        recomb_freq[i, j] = (males[ti] != males[tj]).mean()
print("Pairwise recombination frequencies:")
print(pd.DataFrame(recomb_freq, index=x_linked_traits, columns=x_linked_traits).round(3))
"""

S1_CQ2D_STUDENT = """\
# Challenge 4, Part C: Infer gene order
# The pair with the SMALLEST recombination frequency are the CLOSEST together.
# Iteratively build up the genetic map by placing genes relative to each other.
#
# Hint: Start with the two most closely linked genes, then find which other gene
# is closest to one end or the other.

# YOUR CODE HERE
# 1. Find the gene pair with the smallest recombination frequency
# 2. Build up the map order step by step
# 3. Convert recombination frequencies to genetic distances (Haldane map function)
#    d = -50 * log(1 - 2r) cM    [Haldane]

def haldane_d(r):
    \"\"\"Convert recombination frequency r to map distance in cM.\"\"\"
    return ???

# Print your inferred gene order and map distances
"""

S1_CQ2D_SOL = """\
def haldane_d(r):
    return -50 * np.log(1 - 2*np.clip(r, 0, 0.499))

rf_df = pd.DataFrame(recomb_freq, index=x_linked_traits, columns=x_linked_traits)
# Find closest pair
np.fill_diagonal(recomb_freq, 1.0)
i_min, j_min = np.unravel_index(recomb_freq.argmin(), recomb_freq.shape)
print(f"Closest pair: {x_linked_traits[i_min]} — {x_linked_traits[j_min]}: "
      f"r={recomb_freq[i_min,j_min]:.3f}, d={haldane_d(recomb_freq[i_min,j_min]):.1f} cM")
"""

# ── Challenge 5: Ascertainment by age of onset ───────────────────────────────
S1_CQ4_MD = """\
### Challenge 5: Ascertainment by age of onset (Medium)

Disease cohorts are often **ascertained** — individuals only enter as *cases* once they have
been diagnosed. For a late-onset disease, someone who will eventually develop it but is still
young looks like a *control* at recruitment.

Model this: treat the binary trait as a late-onset disease and **recode every case younger than
60 as a control**, then re-run the logistic GWAS. Compare the hits to the fully-ascertained
baseline. What happens to power, and why?
"""

S1_CQ4_STUDENT = """\
# Challenge 5: Age-of-onset ascertainment
covars = np.column_stack([(age - age.mean()) / age.std(), sex])

# Baseline: full ascertainment (all cases observed)
_, pvals_base = run_logistic_gwas_fast(y_bin, G_qc, covars)

# YOUR CODE HERE
# Late-onset ascertainment: a case younger than 60 has not yet been diagnosed → recode to control
y_asc = y_bin.copy()
y_asc[???] = 0                     # cases with age < 60 become controls

print(f"Cases: {int(y_bin.sum())} -> {int(y_asc.sum())} after requiring onset age >= 60")
_, pvals_asc = run_logistic_gwas_fast(y_asc, G_qc, covars)
print(f"Genome-wide-sig hits:  baseline {int((pvals_base<5e-8).sum())}, "
      f"ascertained {int((pvals_asc<5e-8).sum())}")

# Compare significance at the suggestive variants
sugg = np.where(pvals_base < 1e-4)[0]
fig, ax = plt.subplots(figsize=(5.5, 5.5))
ax.scatter(-np.log10(pvals_base[sugg]+1e-300), -np.log10(pvals_asc[sugg]+1e-300),
           s=30, alpha=0.7, color='steelblue')
lim = -np.log10(min(pvals_base[sugg].min(), pvals_asc[sugg].min()) + 1e-300) * 1.05
ax.plot([0, lim], [0, lim], 'r--', label='y = x')
ax.axhline(-np.log10(5e-8), color='orange', ls=':', lw=1)
ax.axvline(-np.log10(5e-8), color='orange', ls=':', lw=1, label='5e-8')
ax.set_xlabel(r'$-\\log_{10}p$ baseline'); ax.set_ylabel(r'$-\\log_{10}p$ ascertained')
ax.set_title('Effect of age-of-onset ascertainment on power'); ax.legend(fontsize=8)
plt.tight_layout(); plt.show()
print("Q: Why does dropping young cases weaken the signal? (hint: case count and power)")
"""

S1_CQ4_SOL = """\
covars = np.column_stack([(age - age.mean()) / age.std(), sex])
_, pvals_base = run_logistic_gwas_fast(y_bin, G_qc, covars)

y_asc = y_bin.copy()
y_asc[(y_bin == 1) & (age < 60)] = 0
print(f"Cases: {int(y_bin.sum())} -> {int(y_asc.sum())} after requiring onset age >= 60")
_, pvals_asc = run_logistic_gwas_fast(y_asc, G_qc, covars)
print(f"Genome-wide-sig hits:  baseline {int((pvals_base<5e-8).sum())}, "
      f"ascertained {int((pvals_asc<5e-8).sum())}")

sugg = np.where(pvals_base < 1e-4)[0]
fig, ax = plt.subplots(figsize=(5.5, 5.5))
ax.scatter(-np.log10(pvals_base[sugg]+1e-300), -np.log10(pvals_asc[sugg]+1e-300),
           s=30, alpha=0.7, color='steelblue')
lim = -np.log10(min(pvals_base[sugg].min(), pvals_asc[sugg].min()) + 1e-300) * 1.05
ax.plot([0, lim], [0, lim], 'r--', label='y = x')
ax.axhline(-np.log10(5e-8), color='orange', ls=':', lw=1)
ax.axvline(-np.log10(5e-8), color='orange', ls=':', lw=1, label='5e-8')
ax.set_xlabel(r'$-\\log_{10}p$ baseline'); ax.set_ylabel(r'$-\\log_{10}p$ ascertained')
ax.set_title('Effect of age-of-onset ascertainment on power'); ax.legend(fontsize=8)
plt.tight_layout(); plt.show()
# Dropping young cases ~halves the case count → less power → points fall below y=x.
"""

# ── Challenge 6: Polygenic scores (PGS) ──────────────────────────────────────
S1_CQ5_MD = """\
### Challenge 6: Polygenic scores — predicting the genetic component (Hard)

A **polygenic score** is the predicted genetic value $\\hat g_i = \\sum_j x_{ij}\\,\\hat\\beta_j$,
built from GWAS effect estimates. To judge prediction honestly we fit the effects in a
**training** half and score an independent **test** half.

Build a PGS for the continuous trait two ways — using **all** variants vs only the
**genome-wide-significant** ones — and correlate each with the true phenotype in the test set.
Then do the same for the binary trait and show the PGS split by case/control.

How would you push the correlation higher still? (LD clumping, p-value thresholding, penalized
or Bayesian shrinkage of the effects.)
"""

S1_CQ5_STUDENT = """\
# Challenge 6: Polygenic scores (train/test split)
import seaborn as sns
rng = np.random.default_rng(7)
perm = rng.permutation(N); tr, te = perm[:N//2], perm[N//2:]   # train / test halves
cov_tr = np.column_stack([(age[tr]-age.mean())/age.std(), sex[tr]])

# Fit effects on the TRAINING half
b_tr, _, p_tr = run_gwas(y_cont[tr], G_qc[tr], cov_tr)

# YOUR CODE HERE
# PGS on the TEST half = test genotypes @ training effects, for (a) all variants, (b) sig only
pgs_all = ???                       # G_qc[te] @ b_tr
sig = p_tr < 5e-8
pgs_sig = ???                       # G_qc[te][:, sig] @ b_tr[sig]

for nm, pgs in [('all variants', pgs_all), ('genome-wide-sig only', pgs_sig)]:
    print(f"  corr(PGS [{nm}], y_cont) on test set = {np.corrcoef(pgs, y_cont[te])[0,1]:.3f}")

fig, axes = plt.subplots(1, 2, figsize=(11, 5))
for ax, (nm, pgs) in zip(axes, [('all variants', pgs_all), ('sig only', pgs_sig)]):
    ax.scatter(pgs, y_cont[te], s=5, alpha=0.2, color='steelblue')
    ax.set_xlabel(f'PGS — {nm}'); ax.set_ylabel('y_cont (test)')
    ax.set_title(f'r = {np.corrcoef(pgs, y_cont[te])[0,1]:.3f}')
plt.tight_layout(); plt.show()

# Binary trait: PGS split by case/control on the test set
log_or_tr, pbin_tr = run_logistic_gwas_fast(y_bin[tr], G_qc[tr], cov_tr)
sb = pbin_tr < 5e-8
pgs_bin = G_qc[te][:, sb] @ log_or_tr[sb] if sb.sum() else G_qc[te] @ log_or_tr
dfb = pd.DataFrame({'status': np.where(y_bin[te]==1, 'case', 'control'), 'PGS': pgs_bin})
fig, ax = plt.subplots(figsize=(5, 4))
sns.stripplot(data=dfb.sample(min(2000, len(dfb)), random_state=0),
              x='status', y='PGS', alpha=0.3, size=2,
              palette=['#4e79a7', '#e15759'], ax=ax)
ax.set_title('Binary-trait PGS by case/control (test set)'); plt.tight_layout(); plt.show()
print("Q: Why does restricting to significant variants improve the correlation out-of-sample?")
"""

S1_CQ5_SOL = """\
import seaborn as sns
rng = np.random.default_rng(7)
perm = rng.permutation(N); tr, te = perm[:N//2], perm[N//2:]
cov_tr = np.column_stack([(age[tr]-age.mean())/age.std(), sex[tr]])

b_tr, _, p_tr = run_gwas(y_cont[tr], G_qc[tr], cov_tr)
pgs_all = G_qc[te] @ b_tr
sig = p_tr < 5e-8
pgs_sig = G_qc[te][:, sig] @ b_tr[sig]
for nm, pgs in [('all variants', pgs_all), ('genome-wide-sig only', pgs_sig)]:
    print(f"  corr(PGS [{nm}], y_cont) on test set = {np.corrcoef(pgs, y_cont[te])[0,1]:.3f}")

fig, axes = plt.subplots(1, 2, figsize=(11, 5))
for ax, (nm, pgs) in zip(axes, [('all variants', pgs_all), ('sig only', pgs_sig)]):
    ax.scatter(pgs, y_cont[te], s=5, alpha=0.2, color='steelblue')
    ax.set_xlabel(f'PGS — {nm}'); ax.set_ylabel('y_cont (test)')
    ax.set_title(f'r = {np.corrcoef(pgs, y_cont[te])[0,1]:.3f}')
plt.tight_layout(); plt.show()

log_or_tr, pbin_tr = run_logistic_gwas_fast(y_bin[tr], G_qc[tr], cov_tr)
sb = pbin_tr < 5e-8
pgs_bin = G_qc[te][:, sb] @ log_or_tr[sb] if sb.sum() else G_qc[te] @ log_or_tr
dfb = pd.DataFrame({'status': np.where(y_bin[te]==1, 'case', 'control'), 'PGS': pgs_bin})
fig, ax = plt.subplots(figsize=(5, 4))
sns.stripplot(data=dfb.sample(min(2000, len(dfb)), random_state=0),
              x='status', y='PGS', alpha=0.3, size=2,
              palette=['#4e79a7', '#e15759'], ax=ax)
ax.set_title('Binary-trait PGS by case/control (test set)'); plt.tight_layout(); plt.show()
# All-variant PGS is mostly noise out-of-sample; restricting to sig variants raises the correlation.
"""


# ─── Session 2 cells ──────────────────────────────────────────────────────────

S2_TITLE = """\
# Session 2: Interpreting GWAS — Practical

**Timing**: 45 minutes.
- Parts 1–2 (Manhattan + QQ plots): ~30 minutes.
- Challenge questions: for fast finishers.

**Data**: We now use **real genome-wide Pan-UKB summary statistics** (EUR) for three traits —
LDL cholesterol (continuous), chronic ischaemic heart disease (binary), and standing height
(highly polygenic) — for the Manhattan, QQ, trumpet and pleiotropy plots. The setup cell also
re-runs the Session 1 *simulated* GWAS, which we still use for the analyses that need
individual-level genotypes (null-QQ contrast, winner's curse).

**Setup**: Run the setup cell once at the top (loads real sumstats + re-runs the simulated
GWAS, ~30s) before any exercises.
"""

# Loader for the bundled real Pan-UKB summary statistics (LDL / CAD / height, EUR).
# Powers the Manhattan, QQ, lambda_GC, MAF-stratified QQ, trumpet and pleiotropy plots.
# p-values are stored in -log10 units ('nlog10p') to avoid float underflow in the tail.
LOAD_REAL_SUMSTATS = """\
# ── Load bundled real GWAS summary statistics (Pan-UKB, EUR) ──────────────────
# Three real traits, parallel to the simulated ones:
#   ldl    — LDL direct          (continuous biomarker)        ~ like y_cont
#   cad    — chronic ischaemic heart disease (I25, binary)     ~ like y_bin
#   height — standing height     (highly polygenic continuous) ~ like y_poly
# Each trait provides: chrom, pos, maf, beta, se, nlog10p (= -log10 p), and two masks:
#   sig  = genome-wide significant (p < 5e-8); rand = unbiased random subset for QQ/lambda.
_real_path = os.path.join(DATA_DIR, 'sumstats_real.npz')
if not os.path.exists(_real_path):
    _url = f'https://media.githubusercontent.com/media/{REPO_SLUG}/{BRANCH_NAME}/data/sumstats_real.npz'
    print('Downloading sumstats_real.npz from GitHub ...')
    import urllib.request; urllib.request.urlretrieve(_url, _real_path)
_rs = np.load(_real_path, allow_pickle=True)
REAL_TRAITS = ['ldl', 'cad', 'height']
REAL_LABELS = {'ldl': 'LDL direct (continuous)',
               'cad': 'Chronic ischaemic heart disease (binary)',
               'height': 'Standing height (polygenic)'}
real = {}
for _t in REAL_TRAITS:
    real[_t] = {k: _rs[f'{_t}_{k}'] for k in
                ['chrom', 'pos', 'maf', 'beta', 'se', 'nlog10p', 'sig', 'rand']}
print("Real Pan-UKB summary statistics loaded:")
for _t in REAL_TRAITS:
    d = real[_t]
    print(f"  {_t:6s} ({REAL_LABELS[_t]}): {len(d['pos']):,} SNPs  "
          f"({int(d['sig'].sum()):,} genome-wide sig, {int(d['rand'].sum()):,} in QQ subset)")

def _thin(n, k=12000, seed=0):
    \"\"\"Index subset of size <=k for plotting only (keeps figures/notebooks small).\"\"\"
    return (np.arange(n) if n <= k
            else np.random.default_rng(seed).choice(n, k, replace=False))
"""
# Bake the repo/branch into the download URL (tokens are unique; other {…} untouched).
LOAD_REAL_SUMSTATS = (LOAD_REAL_SUMSTATS
                      .replace('{REPO_SLUG}', REPO)
                      .replace('{BRANCH_NAME}', BRANCH))

S2_SETUP = LOAD_DATA.replace(
    "print(f\"\\nReady: N={N:,} samples, M_raw={M_raw:,} variants (pre-QC)\")",
    """print(f"\\nReady: N={N:,} samples, M_raw={M_raw:,} variants (pre-QC)")
""") + """
# ── Re-run QC and GWAS from Session 1 ────────────────────────────────────────
# (This reproduces Session 1 results so Session 2 is self-contained)

# QC
miss_rate = np.isnan(G_raw).mean(axis=0)
alt_freq  = np.nanmean(G_raw, axis=0) / 2
maf       = np.minimum(alt_freq, 1 - alt_freq)

# HWE (fast version using scipy)
G_int  = np.where(np.isnan(G_raw), -1, G_raw).astype(int)
n_samp = (~np.isnan(G_raw)).sum(0).astype(float)
n_AA   = (G_int == 0).sum(0).astype(float)
n_AB   = (G_int == 1).sum(0).astype(float)
n_BB   = (G_int == 2).sum(0).astype(float)
p_hat  = (2*n_BB + n_AB) / (2*n_samp + 1e-8)   # ALT allele frequency
e_AA   = n_samp*(1-p_hat)**2; e_AB = n_samp*2*p_hat*(1-p_hat); e_BB = n_samp*p_hat**2
hwe_chi2 = ((n_AA-e_AA)**2/(e_AA+1e-8) + (n_AB-e_AB)**2/(e_AB+1e-8)
             + (n_BB-e_BB)**2/(e_BB+1e-8))
hwe_pval = stats.chi2.sf(hwe_chi2, df=1)

qc_pass = (miss_rate < 0.05) & (maf > 0.01) & (hwe_pval > 1e-6)
G_qc    = np.where(np.isnan(G_raw[:, qc_pass]), 0, G_raw[:, qc_pass])
pos_qc  = pos[qc_pass];  rsids_qc = rsids[qc_pass];  M_qc = qc_pass.sum()
maf_qc  = maf[qc_pass]
del G_raw  # free ~4 GB; use G_qc from here on

""" + RUN_GWAS_FN + RUN_LOGISTIC_FAST_FN + """
covars = np.column_stack([(age - age.mean()) / age.std(), sex])
betas_cont, ses_cont, pvals_cont = run_gwas(y_cont, G_qc, covars)
betas_poly, ses_poly, pvals_poly = run_gwas(y_poly, G_qc, covars)
log_ors_bin, pvals_bin           = run_logistic_gwas_fast(y_bin, G_qc, covars)

print(f"QC: {M_raw:,} → {M_qc:,} variants")
print(f"Continuous-trait hits (p<5e-8):  {(pvals_cont < 5e-8).sum():,}")
print(f"Polygenic-trait hits  (p<5e-8):  {(pvals_poly < 5e-8).sum():,}  ← expect ≈0")
print(f"Binary-trait hits     (p<5e-8):  {(pvals_bin  < 5e-8).sum():,}")
""" + "\n" + LOAD_REAL_SUMSTATS

_S2_QC_GWAS = """
# ── Re-run QC and GWAS from Session 1 ────────────────────────────────────────
# (This reproduces Session 1 results so Session 2 is self-contained)

# QC
miss_rate = np.isnan(G_raw).mean(axis=0)
alt_freq  = np.nanmean(G_raw, axis=0) / 2
maf       = np.minimum(alt_freq, 1 - alt_freq)

# HWE (fast version using scipy)
G_int  = np.where(np.isnan(G_raw), -1, G_raw).astype(int)
n_samp = (~np.isnan(G_raw)).sum(0).astype(float)
n_AA   = (G_int == 0).sum(0).astype(float)
n_AB   = (G_int == 1).sum(0).astype(float)
n_BB   = (G_int == 2).sum(0).astype(float)
p_hat  = (2*n_BB + n_AB) / (2*n_samp + 1e-8)   # ALT allele frequency
e_AA   = n_samp*(1-p_hat)**2; e_AB = n_samp*2*p_hat*(1-p_hat); e_BB = n_samp*p_hat**2
hwe_chi2 = ((n_AA-e_AA)**2/(e_AA+1e-8) + (n_AB-e_AB)**2/(e_AB+1e-8)
             + (n_BB-e_BB)**2/(e_BB+1e-8))
hwe_pval = stats.chi2.sf(hwe_chi2, df=1)

qc_pass = (miss_rate < 0.05) & (maf > 0.01) & (hwe_pval > 1e-6)
G_qc    = np.where(np.isnan(G_raw[:, qc_pass]), 0, G_raw[:, qc_pass])
pos_qc  = pos[qc_pass];  rsids_qc = rsids[qc_pass];  M_qc = qc_pass.sum()
maf_qc  = maf[qc_pass]
del G_raw  # free ~4 GB; use G_qc from here on

""" + RUN_GWAS_FN + RUN_LOGISTIC_FAST_FN + """
covars = np.column_stack([(age - age.mean()) / age.std(), sex])
betas_cont, ses_cont, pvals_cont = run_gwas(y_cont, G_qc, covars)
betas_poly, ses_poly, pvals_poly = run_gwas(y_poly, G_qc, covars)
log_ors_bin, pvals_bin           = run_logistic_gwas_fast(y_bin, G_qc, covars)

print(f"QC: {M_raw:,} → {M_qc:,} variants")
print(f"Continuous-trait hits (p<5e-8):  {(pvals_cont < 5e-8).sum():,}")
print(f"Polygenic-trait hits  (p<5e-8):  {(pvals_poly < 5e-8).sum():,}  ← expect ≈0")
print(f"Binary-trait hits     (p<5e-8):  {(pvals_bin  < 5e-8).sum():,}")
""" + "\n" + LOAD_REAL_SUMSTATS

_replace_target = "print(f\"\\nReady: N={N:,} samples, M_raw={M_raw:,} variants (pre-QC)\")"
_replace_with = "print(f\"\\nReady: N={N:,} samples, M_raw={M_raw:,} variants (pre-QC)\")\n"
S2_SETUP_STUDENT = LOAD_DATA_STUDENT.replace(_replace_target, _replace_with) + _S2_QC_GWAS

S2_PART1_MD = """\
## Part 1: The Manhattan Plot

A Manhattan plot shows $-\\log_{10}(p)$ for each variant, ordered by genomic position.
Strong association signals appear as 'towers' — clusters of variants with high $-\\log_{10}(p)$.

**Why are there towers?** Variants near a causal variant are often in **linkage disequilibrium (LD)**
with it, so they also show association even if they are not causal themselves.

We plot real Pan-UKB summary statistics across all 22 autosomes, laying chromosomes side by
side. (For clarity/speed we only plot variants with $p < 10^{-2}$.)
"""

S2_EX11_STUDENT = """\
# ── Exercise 1.1: Build a genome-wide Manhattan plot ─────────────────────────
# We now use REAL Pan-UKB summary statistics (LDL, CAD, height) across all
# chromosomes. Each trait dict `real[t]` has: chrom, pos, maf, beta, se, nlog10p.
# NOTE: p-values are stored as nlog10p = -log10(p) already (avoids underflow at
# the extreme tail), so you do NOT need to recompute -log10(p).
# To lay out all chromosomes on one axis we add a cumulative offset per chromosome.

def manhattan_plot(chrom, pos, nlog10p, title='Manhattan plot', ax=None, thresh_plot=2.0):
    \"\"\"
    chrom    : (M,) chromosome (1..22)
    pos      : (M,) base-pair position within chromosome
    nlog10p  : (M,) -log10(p)
    thresh_plot : only plot variants with -log10(p) >= this (declutter; p<1e-2 dropped)
    \"\"\"
    if ax is None:
        fig, ax = plt.subplots(figsize=(13, 4))
    keep = nlog10p >= thresh_plot
    chrom, pos, y = chrom[keep], pos[keep], nlog10p[keep]

    # Build a cumulative x-offset so chromosomes sit side by side
    offset, xticks, xlabels, x = 0, [], [], np.zeros(len(pos))
    for c in range(1, 23):
        m = chrom == c
        if not m.any():
            continue
        x[m] = pos[m] + offset
        cmax = pos[m].max()
        xticks.append(offset + cmax / 2); xlabels.append(str(c))
        # YOUR CODE HERE: colour points for this chromosome (alternate two shades),
        # then scatter x[m] vs y[m]
        ???
        offset += cmax

    # YOUR CODE HERE: add genome-wide (5e-8) and suggestive (1e-5) threshold lines
    ???

    ax.set_xticks(xticks); ax.set_xticklabels(xlabels, fontsize=7)
    ax.set_xlabel('Chromosome'); ax.set_ylabel(r'$-\\log_{10}(p)$')
    ax.set_title(title); ax.set_xlim(0, offset)
    return ax

fig, axes = plt.subplots(3, 1, figsize=(13, 10))
for ax, t in zip(axes, REAL_TRAITS):
    manhattan_plot(real[t]['chrom'], real[t]['pos'], real[t]['nlog10p'],
                   title=f"{REAL_LABELS[t]}", ax=ax)
plt.tight_layout(); plt.show()
print("Q: Which trait is most polygenic (most independent towers genome-wide)?")
"""

S2_EX11_SOL = """\
def manhattan_plot(chrom, pos, nlog10p, title='Manhattan plot', ax=None, thresh_plot=2.0):
    if ax is None:
        fig, ax = plt.subplots(figsize=(13, 4))
    keep = nlog10p >= thresh_plot
    chrom, pos, y = chrom[keep], pos[keep], nlog10p[keep]
    offset, xticks, xlabels, x = 0, [], [], np.zeros(len(pos))
    shades = ['#3b6ea5', '#9ecae1']
    for c in range(1, 23):
        m = chrom == c
        if not m.any():
            continue
        x[m] = pos[m] + offset
        mi = np.where(m)[0]; sel = mi[_thin(len(mi), k=2500, seed=c)]   # thin for display
        ax.scatter(x[sel], y[sel], s=2, alpha=0.6, color=shades[c % 2], rasterized=True)
        cmax = pos[m].max()
        xticks.append(offset + cmax / 2); xlabels.append(str(c))
        offset += cmax
    ax.axhline(-np.log10(5e-8), color='red',    linestyle='--', linewidth=1, label='5×10⁻⁸')
    ax.axhline(-np.log10(1e-5), color='orange', linestyle='--', linewidth=1, label='1×10⁻⁵')
    ax.set_xticks(xticks); ax.set_xticklabels(xlabels, fontsize=7)
    ax.set_xlabel('Chromosome'); ax.set_ylabel(r'$-\\log_{10}(p)$')
    ax.set_title(title); ax.set_xlim(0, offset); ax.legend(fontsize=8, loc='upper right')
    return ax

fig, axes = plt.subplots(3, 1, figsize=(13, 10))
for ax, t in zip(axes, REAL_TRAITS):
    manhattan_plot(real[t]['chrom'], real[t]['pos'], real[t]['nlog10p'],
                   title=f"{REAL_LABELS[t]}", ax=ax)
plt.tight_layout(); plt.show()
"""

S2_PART2_MD = """\
## Part 2: The QQ Plot and Lambda GC

A **QQ (quantile-quantile) plot** compares the observed distribution of $-\\log_{10}(p)$
to what we'd expect under the null (uniform distribution).

- **On the null diagonal** ($y = x$): observed ≈ expected — no inflation or deflation.
- **Above the diagonal**: more significant p-values than expected — could be true signal, population stratification, or model miscalibration.
- **Below the diagonal**: conservative p-values (uncommon).

**Lambda GC** ($\\lambda_{GC}$) summarises inflation as a single number:
$$\\lambda_{GC} = \\frac{\\text{median}(\\chi^2_{obs})}{\\text{expected median of } \\chi^2_1}$$
where the expected median of $\\chi^2_1 \\approx 0.4549$.
"""

S2_EX21_STUDENT = """\
# ── Exercise 2.1: QQ plot ──────────────────────────────────────────────────────
# We pass observed -log10(p) directly (nlog10p) and use the unbiased random subset
# of each real trait (the `rand` mask) so the null distribution isn't distorted by
# the force-kept genome-wide-significant variants.

def qq_plot(nlog10p, label='', ax=None, color='steelblue'):
    \"\"\"QQ plot given observed -log10(p) values.\"\"\"
    if ax is None:
        fig, ax = plt.subplots(figsize=(5, 5))
    n = len(nlog10p)

    # YOUR CODE HERE
    # Step 1: sort observed -log10(p) ascending (least to most significant)
    obs = ???

    # Step 2: expected -log10(p) for order statistics of Uniform[0,1].
    # The k-th smallest p-value (k=1..n) has expected value k/(n+1);
    # so expected -log10(p) for the k-th LARGEST observed = -log10(k/(n+1)).
    exp = ???   # ascending, same length as obs

    # Step 3: scatter expected (x) vs observed (y)
    ???

    # Step 4: add the y=x diagonal, then truncate the x-axis to the max expected value
    exp_max = -np.log10(1.0 / (n + 1))
    ???                                  # plot y=x line up to exp_max
    ax.set_xlim(0, exp_max)              # crop empty whitespace; tail rises in y, not x

    if label:
        ax.legend()
    ax.set_xlabel(r'Expected $-\\log_{10}(p)$')
    ax.set_ylabel(r'Observed $-\\log_{10}(p)$')
    return ax

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
for ax, t, col in zip(axes, REAL_TRAITS, ['steelblue', 'coral', '#59a14f']):
    qq_plot(real[t]['nlog10p'][real[t]['rand']], label=t, ax=ax, color=col)
    ax.set_title(f'{t} QQ plot')
plt.tight_layout(); plt.show()
"""

S2_EX21_SOL = """\
def qq_plot(nlog10p, label='', ax=None, color='steelblue'):
    if ax is None:
        fig, ax = plt.subplots(figsize=(5, 5))
    n = len(nlog10p)
    obs = np.sort(nlog10p)                       # ascending observed -log10(p)
    exp = -np.log10(np.arange(n, 0, -1) / (n + 1))  # ascending expected -log10(p)
    d = np.unique(np.r_[_thin(n, k=6000), np.where(obs >= 2)[0]])  # thin bulk, keep tail
    ax.scatter(exp[d], obs[d], s=2, alpha=0.6, color=color, label=label)
    exp_max = -np.log10(1.0 / (n + 1))
    ax.plot([0, exp_max], [0, exp_max], 'r--', linewidth=1.2, label='y=x (null)')
    ax.set_xlim(0, exp_max)
    if label:
        ax.legend()
    ax.set_xlabel(r'Expected $-\\log_{10}(p)$'); ax.set_ylabel(r'Observed $-\\log_{10}(p)$')
    return ax

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
for ax, t, col in zip(axes, REAL_TRAITS, ['steelblue', 'coral', '#59a14f']):
    qq_plot(real[t]['nlog10p'][real[t]['rand']], label=t, ax=ax, color=col)
    ax.set_title(f'{t} QQ plot')
plt.tight_layout(); plt.show()
"""

S2_EX22_STUDENT = """\
# ── Exercise 2.2: Lambda GC ───────────────────────────────────────────────────
# Lambda GC measures genomic inflation.
# Values close to 1.0 indicate no inflation.
# Values > 1.05 suggest inflation (population stratification OR true polygenicity).
# We compute it on the unbiased random subset of each real trait (the `rand` mask).

def lambda_gc(nlog10p):
    # YOUR CODE HERE
    # Step 1: convert -log10(p) back to p, then to chi-squared statistics (1 df)
    # Hint: p = 10**(-nlog10p);  chi2 = stats.chi2.isf(p, df=1)
    p    = ???
    chi2 = ???
    # Step 2: lambda_GC = median(chi2_obs) / expected median of chi2(1)
    return ???

for t in REAL_TRAITS:
    lam = lambda_gc(real[t]['nlog10p'][real[t]['rand']])
    print(f"Lambda GC — {t:6s}: {lam:.3f}")

print("\\nQ: Height is the most polygenic — how does its lambda_GC compare to CAD?")
print("Q: A high lambda_GC here is driven by true polygenic signal, not stratification.")
print("   How could you tell the two apart? (hint: LDSC intercept, null/synonymous variants)")
"""

S2_EX22_SOL = """\
def lambda_gc(nlog10p):
    p    = np.power(10.0, -np.clip(nlog10p, 0, 300))
    chi2 = stats.chi2.isf(np.clip(p, 1e-300, 1), df=1)
    return np.median(chi2) / stats.chi2.ppf(0.5, df=1)   # expected median ≈ 0.4549

for t in REAL_TRAITS:
    lam = lambda_gc(real[t]['nlog10p'][real[t]['rand']])
    print(f"Lambda GC — {t:6s}: {lam:.3f}")
"""

S2_NULL_QQ = """\
# ── Sanity check: QQ plot under the null (simulated cohort) ──────────────────
# The real traits above show inflation from genuine polygenic signal. To see what
# NO signal looks like, we go back to the SIMULATED individual-level data and shuffle
# the phenotype to destroy any genotype-phenotype relationship, then re-run GWAS.
# The QQ plot should lie on the diagonal and lambda GC ≈ 1.0.

def qq_from_pvals(pvals, label='', ax=None, color='steelblue'):
    \"\"\"Wrapper: QQ plot from raw p-values (simulated GWAS returns p, not -log10 p).\"\"\"
    return qq_plot(-np.log10(np.clip(pvals, 1e-300, 1)), label=label, ax=ax, color=color)

np.random.seed(0)
y_null = y_cont.copy(); np.random.shuffle(y_null)
_, _, pvals_null = run_gwas(y_null, G_qc, covars)

fig, ax = plt.subplots(figsize=(5, 5))
qq_from_pvals(pvals_cont, label='Real (simulated) phenotype', ax=ax, color='steelblue')
qq_from_pvals(pvals_null, label='Null (shuffled)', ax=ax, color='grey')
ax.set_title('Simulated cohort: real vs shuffled phenotype'); ax.legend()
plt.tight_layout(); plt.show()

lambda_null = np.median(stats.chi2.isf(np.clip(pvals_null,1e-300,1),df=1)) / stats.chi2.ppf(0.5,df=1)
print(f"Lambda GC (null): {lambda_null:.3f}  ← should be ≈ 1.0")
"""

S2_PLEIOTROPY = """\
# ── Pleiotropy: signed-beta scatters across all three traits ──────────────────
# A pleiotropy scatter plots the signed effect size (beta) of one trait vs another,
# for variants present in BOTH GWAS. Shared genetic effects show up as a tilted cloud.
# Using the REAL Pan-UKB betas: which trait pairs share signal — and which don't?

def merge_betas(ta, tb):
    \"\"\"Align two real traits on chrom:pos; return (beta_a, beta_b, sig_a, sig_b).\"\"\"
    da, db = real[ta], real[tb]
    key_a = da['chrom'].astype(np.int64) * 10**9 + da['pos']
    key_b = db['chrom'].astype(np.int64) * 10**9 + db['pos']
    common, ia, ib = np.intersect1d(key_a, key_b, return_indices=True)
    return (da['beta'][ia], db['beta'][ib],
            da['sig'][ia],  db['sig'][ib], len(common))

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
trait_pairs = [('ldl', 'cad'), ('ldl', 'height'), ('cad', 'height')]

for ax, (ta, tb) in zip(axes, trait_pairs):
    ba, bb, sa, sb, n_common = merge_betas(ta, tb)
    sig = sa | sb
    ns = np.where(~sig)[0]; ns = ns[_thin(len(ns), k=8000)]   # thin grey cloud for display
    ax.scatter(ba[ns], bb[ns], s=2, alpha=0.1, color='grey', rasterized=True)
    si = np.where(sig)[0]; si = si[_thin(len(si), k=6000, seed=2)]   # thin significant pts
    ax.scatter(ba[si],  bb[si],  s=12, alpha=0.6, color='#e15759', zorder=5,
               label=f'sig in either (n={int(sig.sum()):,})')
    ax.axhline(0, color='black', lw=0.5); ax.axvline(0, color='black', lw=0.5)
    ax.set_xlabel(f'beta — {ta}'); ax.set_ylabel(f'beta — {tb}')
    title = f'{ta} vs {tb}  ({n_common:,} shared SNPs)'
    if sig.sum() > 2:
        r = np.corrcoef(ba[sig], bb[sig])[0, 1]
        title = f'{ta} vs {tb}\\n(r={r:.2f} at sig variants, {n_common:,} shared SNPs)'
    ax.set_title(title); ax.legend(fontsize=7)

plt.suptitle('Pleiotropy: signed effect sizes across three real traits', y=1.02)
plt.tight_layout(); plt.show()
print("Q: LDL and CAD are causally linked — does the LDL–CAD beta cloud tilt positive?")
print("Q: Height is largely independent of lipids/heart disease — what does its cloud look like?")
print("Q: A variant with a large LDL effect but near-zero height effect — what does that imply?")
"""

S2_CQ_MD = """\
---

## Challenge Questions
"""

S2_CQ1_MD = """\
### Challenge 1: QQ plot with 95% confidence interval

Under the null, the $k$-th smallest p-value follows a Beta$(k, n-k+1)$ distribution.
Use `scipy.stats.beta.ppf` to add a 95% confidence band to the QQ plot.
"""

S2_CQ1_STUDENT = """\
# Challenge 1: QQ plot with 95% CI  (real height GWAS, unbiased random subset)

def qq_plot_ci(nlog10p, ax=None, color='steelblue', ci_alpha=0.05, label=''):
    if ax is None:
        fig, ax = plt.subplots(figsize=(5, 5))
    n   = len(nlog10p)
    obs = np.sort(nlog10p)              # ascending observed -log10(p)
    ks  = np.arange(n, 0, -1)           # matching order index for ascending obs

    # YOUR CODE HERE
    # Expected quantiles and 95% CI bounds using the Beta distribution.
    # The k-th smallest p-value ~ Beta(k, n-k+1).
    exp_med  = ???   # stats.beta.ppf(0.5, ks, n-ks+1)
    ci_lower = ???   # stats.beta.ppf(ci_alpha/2, ks, n-ks+1)
    ci_upper = ???   # stats.beta.ppf(1 - ci_alpha/2, ks, n-ks+1)

    ax.fill_between(-np.log10(exp_med),
                    -np.log10(ci_upper),    # upper p-bound → lower -log10
                    -np.log10(ci_lower),
                    alpha=0.2, color=color, label='95% CI')
    ax.scatter(-np.log10(exp_med), obs, s=2, alpha=0.6, color=color, label=label)
    exp_max = (-np.log10(exp_med)).max()
    ax.plot([0, exp_max], [0, exp_max], 'r--', linewidth=1.0)
    ax.set_xlim(0, exp_max)              # truncate x-axis to max expected
    ax.set_xlabel(r'Expected $-\\log_{10}(p)$'); ax.set_ylabel(r'Observed $-\\log_{10}(p)$')
    ax.legend(fontsize=8)
    return ax

fig, ax = plt.subplots(figsize=(6, 6))
qq_plot_ci(real['height']['nlog10p'][real['height']['rand']], ax=ax, label='height')
ax.set_title('QQ plot with 95% CI — real height GWAS'); plt.tight_layout(); plt.show()
"""

S2_CQ1_SOL = """\
def qq_plot_ci(nlog10p, ax=None, color='steelblue', ci_alpha=0.05, label=''):
    if ax is None:
        fig, ax = plt.subplots(figsize=(5, 5))
    n = len(nlog10p); obs = np.sort(nlog10p); ks = np.arange(n, 0, -1)
    d = np.unique(np.r_[_thin(n, k=4000), np.where(obs >= 2)[0]])   # thin bulk, keep tail
    obs, ks = obs[d], ks[d]
    exp_med  = stats.beta.ppf(0.5, ks, n-ks+1)
    ci_lower = stats.beta.ppf(ci_alpha/2, ks, n-ks+1)
    ci_upper = stats.beta.ppf(1 - ci_alpha/2, ks, n-ks+1)
    ax.fill_between(-np.log10(exp_med), -np.log10(ci_upper), -np.log10(ci_lower),
                    alpha=0.2, color=color, label='95% CI')
    ax.scatter(-np.log10(exp_med), obs, s=2, alpha=0.6, color=color, label=label)
    exp_max = (-np.log10(exp_med)).max()
    ax.plot([0, exp_max], [0, exp_max], 'r--', linewidth=1.0); ax.set_xlim(0, exp_max)
    ax.set_xlabel(r'Expected $-\\log_{10}(p)$'); ax.set_ylabel(r'Observed $-\\log_{10}(p)$')
    ax.legend(fontsize=8); return ax

fig, ax = plt.subplots(figsize=(6, 6))
qq_plot_ci(real['height']['nlog10p'][real['height']['rand']], ax=ax, label='height')
ax.set_title('QQ plot with 95% CI — real height GWAS')
plt.tight_layout(); plt.show()
"""

S2_CQ2_MD = """\
### Challenge 2: MAF-stratified QQ plot

Are rare variants (low MAF) better or worse behaved than common variants?
Stratify the real height variants into MAF bins and overlay QQ plots (random subset).
"""

S2_CQ2_STUDENT = """\
# Challenge 2: MAF-stratified QQ plot (real height GWAS, random subset)
sub      = real['height']['rand']
maf_h    = real['height']['maf'][sub]
nlp_h    = real['height']['nlog10p'][sub]
maf_bins = [(0.005, 0.01), (0.01, 0.05), (0.05, 0.15), (0.15, 0.50)]
colours  = ['#b07aa1', '#e15759', '#f28e2b', '#4e79a7']

fig, ax = plt.subplots(figsize=(6, 6)); exp_max = 0
for (lo, hi), col in zip(maf_bins, colours):
    mask = (maf_h >= lo) & (maf_h < hi)
    if mask.sum() < 10:
        continue
    # YOUR CODE HERE: build the QQ coordinates for this stratum and scatter them
    # obs = sorted nlog10p in this bin (ascending); exp = -log10 of expected order stats
    obs = ???
    exp = ???
    ax.scatter(exp, obs, s=3, alpha=0.5, color=col, label=f'MAF [{lo:.1%}, {hi:.0%})')
    exp_max = max(exp_max, exp.max())

ax.plot([0, exp_max], [0, exp_max], 'k--', linewidth=1); ax.set_xlim(0, exp_max)
ax.set_title('MAF-stratified QQ plot — real height GWAS')
ax.set_xlabel(r'Expected $-\\log_{10}(p)$'); ax.set_ylabel(r'Observed $-\\log_{10}(p)$')
ax.legend(); plt.tight_layout(); plt.show()
print("Q: Common variants carry most of height's association signal — does the plot agree?")
"""

S2_CQ2_SOL = """\
sub      = real['height']['rand']
maf_h    = real['height']['maf'][sub]
nlp_h    = real['height']['nlog10p'][sub]
maf_bins = [(0.005, 0.01), (0.01, 0.05), (0.05, 0.15), (0.15, 0.50)]
colours  = ['#b07aa1', '#e15759', '#f28e2b', '#4e79a7']

fig, ax = plt.subplots(figsize=(6, 6)); exp_max = 0
for (lo, hi), col in zip(maf_bins, colours):
    mask = (maf_h >= lo) & (maf_h < hi)
    if mask.sum() < 10:
        continue
    obs = np.sort(nlp_h[mask]); m = len(obs)
    exp = -np.log10(np.arange(m, 0, -1) / (m + 1))
    d = np.unique(np.r_[_thin(m, k=4000, seed=int(lo*1000)), np.where(obs >= 2)[0]])
    ax.scatter(exp[d], obs[d], s=3, alpha=0.5, color=col, label=f'MAF [{lo:.1%}, {hi:.0%})')
    exp_max = max(exp_max, exp.max())
ax.plot([0, exp_max], [0, exp_max], 'k--', linewidth=1); ax.set_xlim(0, exp_max)
ax.set_title('MAF-stratified QQ plot — real height GWAS')
ax.set_xlabel(r'Expected $-\\log_{10}(p)$'); ax.set_ylabel(r'Observed $-\\log_{10}(p)$')
ax.legend(); plt.tight_layout(); plt.show()
"""

S2_CQ3_MD = """\
### Challenge 3: Trumpet plot and power curves

A **trumpet plot** shows the signed effect size ($\\hat{\\beta}$) vs. MAF for each variant.
The "trumpet" shape emerges because the power curves form a ±symmetric band:
variants outside the band are detectable, those inside are below the noise floor.
Points beyond the band on the left (rare variants) require large effects to be detected.

We use the **real LDL GWAS** (Pan-UKB, EUR, N≈400k). Real discovered variants should sit
on or beyond the power band for a study of that size.
"""

S2_CQ3_STUDENT = """\
# Challenge 3: Trumpet plot with power curves (signed beta) — real LDL GWAS

def power_curve(n, maf_arr, alpha=5e-8):
    \"\"\"
    Minimum detectable |beta| at given MAF and power=0.8.
    ncp = n * 2*f*(1-f) * beta^2; at threshold: beta_min = sqrt(ncp_80 / (n * 2*f*(1-f)))
    \"\"\"
    from scipy.stats import norm
    ncp_target = (norm.isf(alpha / 2) + norm.isf(0.2))**2   # ≈ 24.0 for alpha=5e-8
    return np.sqrt(ncp_target / (n * 2 * maf_arr * (1 - maf_arr) + 1e-8))

maf_l  = real['ldl']['maf']
beta_l = real['ldl']['beta']
sig_l  = real['ldl']['sig']
rnd_l  = real['ldl']['rand']
maf_grid = np.logspace(np.log10(0.005), np.log10(0.5), 300)

fig, ax = plt.subplots(figsize=(9, 6))

# YOUR CODE HERE
# 1. Scatter signed beta vs MAF for the non-significant random-subset variants (grey, small)
???

# 2. Scatter signed beta vs MAF for genome-wide-significant variants (coloured, larger)
???

# 3. Overlay ±power curves for N=10,000, N=100,000, N=400,000 (≈ the real study size)
for n_samples, col in [(10_000, '#e15759'), (100_000, '#f28e2b'), (400_000, '#59a14f')]:
    pb = power_curve(n_samples, maf_grid)
    ax.plot(maf_grid, +pb, color=col, linewidth=1.5, label=f'N={n_samples:,}')
    ax.plot(maf_grid, -pb, color=col, linewidth=1.5, linestyle='--')

ax.axhline(0, color='black', linewidth=0.5)
ax.set_xscale('log')
ax.set_xlabel('MAF (log scale)'); ax.set_ylabel(r'$\\hat{\\beta}$ (SD units)')
ax.set_title('Trumpet plot: GWAS discovery space — real LDL GWAS')
ax.legend(fontsize=8); plt.tight_layout(); plt.show()
print("Q: Do the discovered (significant) variants fall outside the N=400k power band?")
print("Q: Why are there no common variants with very large effects (top-left/right empty)?")
"""

S2_CQ3_SOL = """\
def power_curve(n, maf_arr, alpha=5e-8):
    from scipy.stats import norm
    ncp_target = (norm.isf(alpha/2) + norm.isf(0.2))**2
    return np.sqrt(ncp_target / (n * 2 * maf_arr * (1-maf_arr) + 1e-8))

maf_l, beta_l = real['ldl']['maf'], real['ldl']['beta']
sig_l, rnd_l  = real['ldl']['sig'], real['ldl']['rand']
maf_grid = np.logspace(np.log10(0.005), np.log10(0.5), 300)

fig, ax = plt.subplots(figsize=(9, 6))
bgi = np.where(rnd_l & ~sig_l)[0]; bgi = bgi[_thin(len(bgi), k=8000)]   # thin background
ax.scatter(maf_l[bgi], beta_l[bgi], s=1, alpha=0.15, color='grey', rasterized=True)
si = np.where(sig_l)[0]; si = si[_thin(len(si), k=6000, seed=1)]        # thin significant pts
ax.scatter(maf_l[si], beta_l[si], s=10, alpha=0.6,
           c=np.sign(beta_l[si]), cmap='coolwarm', vmin=-1, vmax=1, zorder=5, label='p<5e-8')
for n_s, col in [(10_000,'#e15759'),(100_000,'#f28e2b'),(400_000,'#59a14f')]:
    pb = power_curve(n_s, maf_grid)
    ax.plot(maf_grid, +pb, color=col, linewidth=1.5, label=f'N={n_s:,}')
    ax.plot(maf_grid, -pb, color=col, linewidth=1.5, linestyle='--')
ax.axhline(0, color='black', linewidth=0.5)
ax.set_xscale('log')
ax.set_xlabel('MAF (log scale)'); ax.set_ylabel(r'$\\hat{{\\beta}}$')
ax.set_title('Trumpet plot — real LDL GWAS'); ax.legend(fontsize=8)
plt.tight_layout(); plt.show()
"""

S2_CQ4_MD = """\
### Challenge 4: Winner's curse

The **winner's curse**: an effect-size estimate for a *discovered* variant is upward-biased,
because a variant is only declared significant when its estimated effect happens to land large
enough to cross the threshold. The honest effect is what you measure in an **independent
validation** sample.

Using the simulated cohort (N=10,000), for each replicate we randomly split into a **5,000
discovery** half and a disjoint **5,000 validation** half:
1. Run the discovery GWAS; keep replicates that yield ≥1 genome-wide-significant variant
   (resample the split until we have 3 such replicates).
2. For each significant variant, re-estimate its effect **once** in that replicate's validation
   half.
3. Scatter discovery vs validation effect sizes across all replicates. Winner's curse shows up
   as validation effects sitting **closer to zero** than the discovery effects (below the y=x
   line in magnitude).
"""

S2_CQ4_STUDENT = """\
# Challenge 4: Winner's curse — discovery vs validation across resampled 5k/5k splits
rng = np.random.default_rng(0)
def _covs(idx):
    return np.column_stack([(age[idx]-age.mean())/age.std(), sex[idx]])

disc_betas, val_betas = [], []
n_rep, n_attempt = 0, 0
while n_rep < 3:                         # resample splits until 3 replicates have a hit
    n_attempt += 1
    perm = rng.permutation(N)
    disc, val = perm[:5000], perm[5000:]            # disjoint 5k discovery / 5k validation

    # Discovery GWAS on this 5k half
    bd, _, pd = run_gwas(y_cont[disc], G_qc[disc], _covs(disc))
    hits = np.where(pd < 5e-8)[0]
    if len(hits) == 0:
        continue                                     # no hit this split — resample

    # YOUR CODE HERE: estimate those hit variants' effects ONCE in the validation half
    # Hint: run_gwas on just the hit columns, G_qc[:, hits][val]
    bv, _, _ = run_gwas(???, ???, ???)

    disc_betas.extend(bd[hits]); val_betas.extend(bv)
    n_rep += 1
    print(f"  replicate {n_rep}: {len(hits)} hit(s)  (found on attempt {n_attempt})")

disc_betas = np.array(disc_betas); val_betas = np.array(val_betas)
print(f"Used {n_attempt} resampled splits to get 3 replicates with a hit; "
      f"{len(disc_betas)} discovery hits total")

lim = np.abs(np.r_[disc_betas, val_betas]).max() * 1.15
fig, ax = plt.subplots(figsize=(5.5, 5.5))
ax.scatter(disc_betas, val_betas, s=45, alpha=0.8, color='steelblue', zorder=3)
ax.plot([-lim, lim], [-lim, lim], 'r--', label='y = x (no bias)')
ax.axhline(0, color='grey', lw=0.5); ax.axvline(0, color='grey', lw=0.5)
ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
ax.set_xlabel('Discovery effect size (5k)'); ax.set_ylabel('Validation effect size (disjoint 5k)')
ax.set_title("Winner's curse: discovery vs validation"); ax.legend(fontsize=8)
plt.tight_layout(); plt.show()
print("Q: Are the validation effects systematically closer to zero than the discovery effects?")
"""

S2_CQ4_SOL = """\
rng = np.random.default_rng(0)
def _covs(idx):
    return np.column_stack([(age[idx]-age.mean())/age.std(), sex[idx]])

disc_betas, val_betas = [], []
n_rep, n_attempt = 0, 0
while n_rep < 3:
    n_attempt += 1
    perm = rng.permutation(N)
    disc, val = perm[:5000], perm[5000:]
    bd, _, pd = run_gwas(y_cont[disc], G_qc[disc], _covs(disc))
    hits = np.where(pd < 5e-8)[0]
    if len(hits) == 0:
        continue
    bv, _, _ = run_gwas(y_cont[val], G_qc[:, hits][val], _covs(val))
    disc_betas.extend(bd[hits]); val_betas.extend(bv)
    n_rep += 1
    print(f"  replicate {n_rep}: {len(hits)} hit(s)  (found on attempt {n_attempt})")

disc_betas = np.array(disc_betas); val_betas = np.array(val_betas)
print(f"Used {n_attempt} resampled splits to get 3 replicates with a hit; "
      f"{len(disc_betas)} discovery hits total")

lim = np.abs(np.r_[disc_betas, val_betas]).max() * 1.15
fig, ax = plt.subplots(figsize=(5.5, 5.5))
ax.scatter(disc_betas, val_betas, s=45, alpha=0.8, color='steelblue', zorder=3)
ax.plot([-lim, lim], [-lim, lim], 'r--', label='y = x (no bias)')
ax.axhline(0, color='grey', lw=0.5); ax.axvline(0, color='grey', lw=0.5)
ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
ax.set_xlabel('Discovery effect size (5k)'); ax.set_ylabel('Validation effect size (disjoint 5k)')
ax.set_title("Winner's curse: discovery vs validation"); ax.legend(fontsize=8)
plt.tight_layout(); plt.show()
# Validation effects tend to sit closer to zero (below |y=x|) → winner's curse.
"""

S2_CQ6_MD = """\
### Challenge 5: Linking a real lead variant to known associations

Our real LDL lead variant sits at a genuine genomic locus. Query the GWAS Catalog REST API
for known associations near it — because this is real LDL data (not simulated), you should
recover real, biologically sensible associations (think *APOE*, *LDLR*, *PCSK9*, *SORT1* ...).
"""

S2_CQ6_STUDENT = """\
# Challenge 5: Query GWAS Catalog API for the REAL LDL lead variant's locus
import requests

# Lead variant from the real LDL GWAS (largest -log10 p)
j_lead  = int(np.argmax(real['ldl']['nlog10p']))
chr_num = int(real['ldl']['chrom'][j_lead])
pos_bp  = int(real['ldl']['pos'][j_lead])     # Pan-UKB positions are GRCh37 bp

print(f"Real LDL lead variant: chr{chr_num}:{pos_bp:,}  "
      f"-log10(p)={real['ldl']['nlog10p'][j_lead]:.1f}")
print("Querying GWAS Catalog for known associations within ±100 kb ...")

window = 100_000
url = (f"https://www.ebi.ac.uk/gwas/rest/api/associations/search/findByLocation"
       f"?chromosomeName={chr_num}"
       f"&chromosomePosition={pos_bp - window}"
       f"&chromosomePositionEnd={pos_bp + window}"
       f"&size=20")
try:
    resp = requests.get(url, timeout=20)
    hits = resp.json().get('_embedded', {}).get('associations', [])
    for h in hits[:12]:
        snp   = h.get('loci', [{}])[0].get('strongestRiskAlleles', [{}])[0].get('riskAlleleName','?')
        trait = h.get('study', {}).get('diseaseTrait', {}).get('trait', '?')
        pval  = h.get('pvalue', '?')
        print(f"  {snp:22s}  trait: {trait:45s}  p={pval}")
    if not hits:
        print("  No associations returned (try widening the window or check the API).")
except Exception as e:
    print(f"  API request failed: {e}")
    print(f"  Try manually: https://www.ebi.ac.uk/gwas/regions/{chr_num}:{pos_bp-window}-{pos_bp+window}")
print("\\nQ: Are the nearby associations lipid-related, as you'd expect for an LDL locus?")
"""

S2_CQ7_MD = """\
### Challenge 6: Manual LocusZoom plot

A **LocusZoom plot** shows $-\\log_{10}(p)$ vs. position for a region,
with points **coloured by LD** ($r^2$) with the lead variant.
This reveals the LD structure underlying a GWAS peak.

Computing $r^2$ needs individual-level genotypes, which the real summary statistics don't
include — so this uses the **simulated cohort** (where we have `G_qc`).
"""

S2_CQ7_STUDENT = """\
# Challenge 6: Manual LocusZoom plot for the lead locus

# Define the region to zoom into (±1 Mb around the lead variant)
j_lead    = np.argmin(pvals_cont)
pos_lead  = pos_qc[j_lead]
region_mask = np.abs(pos_qc - pos_lead) < 1000  # ±1000 kbp = ±1 Mb

print(f"Region: chr1:{pos_lead-1000:,}–{pos_lead+1000:,} kbp")
print(f"Variants in region: {region_mask.sum():,}")

# YOUR CODE HERE
# Step 1: compute r² between each variant in the region and the lead variant
# Hint: Pearson correlation r = corrcoef(G_lead, G_j); r² = r**2
g_lead = G_qc[:, j_lead]
r2 = ???    # shape (n_region,)

# Step 2: create the LocusZoom scatter plot
# x = position (Mb), y = -log10(p), colour = r²
cmap = plt.cm.RdYlBu_r  # red = high LD, blue = low LD
norm = mcolors.Normalize(vmin=0, vmax=1)

fig, ax = plt.subplots(figsize=(12, 4))
sc = ax.scatter(pos_qc[region_mask] / 1000,
                -np.log10(pvals_cont[region_mask] + 1e-300),
                c=r2, cmap=cmap, norm=norm, s=20, zorder=3)
# Highlight lead variant
ax.scatter([pos_lead/1000], [-np.log10(pvals_cont[j_lead]+1e-300)],
           s=100, marker='D', color='black', zorder=5, label='Lead SNP')
plt.colorbar(sc, ax=ax, label=r'$r^2$ with lead SNP')
ax.axhline(-np.log10(5e-8), color='red', linestyle='--', linewidth=1, label='5×10⁻⁸')
ax.set_xlabel('Position on chr1 (Mb)'); ax.set_ylabel(r'$-\\log_{10}(p)$')
ax.set_title(f'LocusZoom: lead locus on chr1 (±1 Mb) — continuous trait')
ax.legend(fontsize=8); plt.tight_layout(); plt.show()
"""

S2_CQ7_SOL = """\
j_lead = np.argmin(pvals_cont); pos_lead = pos_qc[j_lead]
region_mask = np.abs(pos_qc - pos_lead) < 1000
g_lead = G_qc[:, j_lead]
G_region = G_qc[:, region_mask]
r2 = np.array([np.corrcoef(g_lead, G_region[:, k])[0,1]**2 for k in range(G_region.shape[1])])

cmap = plt.cm.RdYlBu_r; norm = mcolors.Normalize(vmin=0, vmax=1)
fig, ax = plt.subplots(figsize=(12, 4))
sc = ax.scatter(pos_qc[region_mask]/1000, -np.log10(pvals_cont[region_mask]+1e-300),
                c=r2, cmap=cmap, norm=norm, s=20, zorder=3)
ax.scatter([pos_lead/1000], [-np.log10(pvals_cont[j_lead]+1e-300)],
           s=100, marker='D', color='black', zorder=5, label='Lead SNP')
plt.colorbar(sc, ax=ax, label=r'$r^2$ with lead SNP')
ax.axhline(-np.log10(5e-8), color='red', linestyle='--', linewidth=1)
ax.set_xlabel('Position on chr1 (Mb)'); ax.set_ylabel(r'$-\\log_{10}(p)$')
ax.set_title('LocusZoom: lead locus — continuous trait'); ax.legend(fontsize=8)
plt.tight_layout(); plt.show()
"""


# ─── Assemble notebooks ───────────────────────────────────────────────────────

def build_session1(answers=False, run=False, nb_path=None):
    """
    answers=True  → student cell + collapsed solution cell (for students to reveal)
    run=True      → solution code only, no student cells (instructor-executable)
    nb_path       → repo path of this notebook, used for the Open-in-Colab badge
    """
    def ex(student_src, sol_src):
        if run:
            return [code(sol_src)]
        cells = [code(student_src)]
        if answers:
            cells.append(solution(sol_src))
        return cells

    title = (colab_badge(nb_path) + "\n\n" + S1_TITLE) if nb_path else S1_TITLE
    cells = [
        md(title),
        code(IMPORTS),
        code(LOAD_DATA if (answers or run) else LOAD_DATA_STUDENT),
        code(S1_PHENOTYPE_PLOTS),
        md(S1_PART1_MD),
        *ex(S1_EX11_STUDENT, S1_EX11_SOL),
        *ex(S1_EX12_STUDENT, S1_EX12_SOL),
        md(S1_EX13_MD),
        code(S1_PROVIDE_HWE),
        code(S1_QC_APPLY),
        md(S1_PART2_MD),
        code(S1_PROVIDE_GWAS_FN),
        *ex(S1_EX21_STUDENT, S1_EX21_SOL),
        *ex(S1_EX22_STUDENT, S1_EX22_SOL),
        *ex(S1_EX23_STUDENT, S1_EX23_SOL),
        md(S1_PART3_MD),
        code(S1_PROVIDE_LOGISTIC_FN),
        *ex(S1_EX31_STUDENT, S1_EX31_SOL),
        md(S1_CQ_MD),
        md(S1_CQHWE_MD),
        *ex(S1_CQHWE_STUDENT, S1_CQHWE_SOL),
        md(S1_CQ1_MD),
        *ex(S1_CQ1_STUDENT, S1_CQ1_SOL),
        md(S1_LZ_MD),
        *ex(S1_LZ_STUDENT, S1_LZ_SOL),
        md(S1_CQ2_MD),
        code(S1_CQ2_STUDENT) if not run else md(""),
        *ex(S1_CQ2B_STUDENT, S1_CQ2B_SOL),
        *ex(S1_CQ2C_STUDENT, S1_CQ2C_SOL),
        *ex(S1_CQ2D_STUDENT, S1_CQ2D_SOL),
        md(S1_CQ4_MD),
        *ex(S1_CQ4_STUDENT, S1_CQ4_SOL),
        md(S1_CQ5_MD),
        *ex(S1_CQ5_STUDENT, S1_CQ5_SOL),
    ]
    return notebook(cells)


def build_session2(answers=False, run=False, nb_path=None):
    def ex(student_src, sol_src):
        if run:
            return [code(sol_src)]
        cells = [code(student_src)]
        if answers:
            cells.append(solution(sol_src))
        return cells

    title = (colab_badge(nb_path) + "\n\n" + S2_TITLE) if nb_path else S2_TITLE
    cells = [
        md(title),
        code(S2_SETUP if (answers or run) else S2_SETUP_STUDENT),
        md(S2_PART1_MD),
        *ex(S2_EX11_STUDENT, S2_EX11_SOL),
        md(S2_PART2_MD),
        *ex(S2_EX21_STUDENT, S2_EX21_SOL),
        *ex(S2_EX22_STUDENT, S2_EX22_SOL),
        code(S2_NULL_QQ),
        code(S2_PLEIOTROPY),
        md(S2_CQ_MD),
        md(S2_CQ1_MD),
        *ex(S2_CQ1_STUDENT, S2_CQ1_SOL),
        md(S2_CQ2_MD),
        *ex(S2_CQ2_STUDENT, S2_CQ2_SOL),
        md(S2_CQ3_MD),
        *ex(S2_CQ3_STUDENT, S2_CQ3_SOL),
        md(S2_CQ4_MD),
        *ex(S2_CQ4_STUDENT, S2_CQ4_SOL),
        md(S2_CQ6_MD),
        code(S2_CQ6_STUDENT) if not run else md(""),
    ]
    return notebook(cells)


# ─── Write files ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    BASE = os.path.dirname(os.path.abspath(__file__))
    print("Generating notebooks...")
    save(build_session1(answers=False, nb_path="session1/practical.ipynb"), os.path.join(BASE, "session1", "practical.ipynb"))
    save(build_session1(answers=True,  nb_path="session1/answers.ipynb"),   os.path.join(BASE, "session1", "answers.ipynb"))
    save(build_session1(run=True,      nb_path="session1/run.ipynb"),       os.path.join(BASE, "session1", "run.ipynb"))
    save(build_session2(answers=False, nb_path="session2/practical.ipynb"), os.path.join(BASE, "session2", "practical.ipynb"))
    save(build_session2(answers=True,  nb_path="session2/answers.ipynb"),   os.path.join(BASE, "session2", "answers.ipynb"))
    save(build_session2(run=True,      nb_path="session2/run.ipynb"),       os.path.join(BASE, "session2", "run.ipynb"))
    print("Done. 6 notebooks written.")
