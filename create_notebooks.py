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
_LFS  = {{'gwas_data.npz', 'sumstats_real.npz', 'pca_data.npz', 'finemap_data.npz'}}   # Git LFS
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

dom_idx_qc = -1; rec_idx_qc = -1; sexspec_idx_qc = -1   # special loci not available in fallback
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
    sexspec_idx_qc = int(d['sexspec_idx_qc'][0]) if 'sexspec_idx_qc' in d.files else -1  # sex-specific locus
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
sexspec_idx_qc = int(d['sexspec_idx_qc'][0]) if 'sexspec_idx_qc' in d.files else -1  # sex-specific locus
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

S1_CQ1_MD = """\
### Challenge {N}: Additive, dominant, and recessive models — ~10 min

The standard GWAS uses an **additive** model: genotype is encoded 0/1/2 (copies of ALT allele),
so the heterozygote AB sits midway between AA and BB. We can write that encoding as a little
look-up table, `additive = [0, 1, 2]` — "an AA is worth 0, an AB is worth 1, a BB is worth 2".

But what if the true effect is **dominant** (one ALT copy is already enough → AB behaves like BB)
or **recessive** (you need two ALT copies → AB behaves like AA)?

**Your task:** write the look-up tables `dominant` and `recessive` that capture those two genetic
models — i.e. decide what value AA, AB and BB should map to under each. We give you a `recode()`
helper that applies a look-up table to the genotype matrix; you do **not** need to write any
indexing logic.

Two variants in this dataset have non-additive effects — one dominant, one recessive.
They may not be the genome-wide lead hit under the additive model!
"""

S1_CQ1_STUDENT = """\
# Find the non-additive loci
import seaborn as sns

# Provided: recode() applies a genotype look-up table to the whole genotype matrix.
# mapping[g] is the value a genotype g (0=AA, 1=AB, 2=BB) is recoded to.
def recode(G, mapping):
    \"\"\"Recode genotypes 0/1/2 via a 3-element look-up table, e.g. [0, 1, 2] = additive.\"\"\"
    return np.asarray(mapping, dtype=float)[G.astype(int)]

additive = [0, 1, 2]            # worked example: AA→0, AB→1, BB→2  (the standard model)
# YOUR CODE HERE: write the genetic look-up tables for the two non-additive models.
dominant  = ???                 # e.g. one ALT copy is already "enough"
recessive = ???                 # e.g. you need two ALT copies to see an effect

G_dom = recode(G_qc, dominant)
G_rec = recode(G_qc, recessive)

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
def recode(G, mapping):
    return np.asarray(mapping, dtype=float)[G.astype(int)]

additive  = [0, 1, 2]
dominant  = [0, 1, 1]          # AB behaves like BB: one ALT copy is enough
recessive = [0, 0, 1]          # AB behaves like AA: need two ALT copies
G_dom = recode(G_qc, dominant)
G_rec = recode(G_qc, recessive)

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
### Challenge {N}: Manual LocusZoom plot — ~12 min

A **LocusZoom plot** shows $-\\log_{10}(p)$ vs. position for a region around a hit, with points
**coloured by LD** ($r^2$) with the lead variant. It reveals the LD structure that produces the
"tower" you'll see in a Manhattan plot (Session 2) — neighbouring variants are associated only
because they are correlated with the causal one.

We use the covariate-adjusted continuous-trait GWAS (`pvals_cov`) and the genotypes `G_qc`.
"""

S1_LZ_STUDENT = """\
# Manual LocusZoom plot for the lead locus (continuous trait)

# Region: ±1 Mb around the lead variant
j_lead    = np.argmin(pvals_cov)
pos_lead  = pos_qc[j_lead]
region_mask = np.abs(pos_qc - pos_lead) < 1000   # ±1000 kbp = ±1 Mb

print(f"Lead variant: {rsids_qc[j_lead]}  pos={pos_lead:,} kbp  p={pvals_cov[j_lead]:.2e}")
print(f"Variants in region: {region_mask.sum():,}")

# Provided: r2_with() returns the LD (r²) between one variant's genotypes and every column.
def r2_with(genotypes, lead_genotype):
    \"\"\"r² (LD) between `lead_genotype` and every column of `genotypes`.\"\"\"
    gc = genotypes - genotypes.mean(0); lc = lead_genotype - lead_genotype.mean()
    r = (gc * lc[:, None]).sum(0) / np.sqrt((gc**2).sum(0) * (lc @ lc) + 1e-300)
    return r ** 2

G_region = G_qc[:, region_mask]
# YOUR CODE HERE: a LocusZoom colours each variant by its LD with the LEAD variant.
# Which variant's genotype column should we measure LD against? Pass it to r2_with().
r2 = r2_with(G_region, ???)              # the lead variant's genotypes: G_qc[:, j_lead]

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
def r2_with(genotypes, lead_genotype):
    gc = genotypes - genotypes.mean(0); lc = lead_genotype - lead_genotype.mean()
    r = (gc * lc[:, None]).sum(0) / np.sqrt((gc**2).sum(0) * (lc @ lc) + 1e-300)
    return r ** 2

j_lead = np.argmin(pvals_cov); pos_lead = pos_qc[j_lead]
region_mask = np.abs(pos_qc - pos_lead) < 1000
G_region = G_qc[:, region_mask]
r2 = r2_with(G_region, G_qc[:, j_lead])

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
### Challenge {N}: Drosophila linkage analysis — ~25 min

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
# Part A: Identify X-linked traits
# For X-linked recessive traits in a test cross: males show the trait ~50% of the time,
# females show it ~0% of the time (they are carriers).
# For autosomal traits: frequency is similar in males and females.

fly_df.head()
"""

S1_CQ2B_STUDENT = """\
# Part A (continued)
trait_cols = [c for c in fly_df.columns if c.startswith('trait_')]

# Provided: the frequency of each trait in females (sex=0) and males (sex=1).
freq_by_sex = fly_df.groupby('sex')[trait_cols].mean()
print(freq_by_sex.T)

# An X-linked recessive trait shows up in ~half of males (one X) but almost no females (need two
# copies); an autosomal trait appears at a similar frequency in both sexes.
# YOUR CODE HERE: read the table above and list the trait names that look X-linked.
x_linked_traits = ???           # e.g. ['trait_eye', 'trait_wing', ...]
print("\\nX-linked traits:", x_linked_traits)
"""

S1_CQ2B_SOL = """\
trait_cols = [c for c in fly_df.columns if c.startswith('trait_')]
freq_by_sex = fly_df.groupby('sex')[trait_cols].mean()
print(freq_by_sex.T)
# X-linked traits have frequency ~0.5 in males and ~0 in females (read straight off the table).
x_linked_traits = freq_by_sex.columns[(freq_by_sex.loc[1] > 0.3) & (freq_by_sex.loc[0] < 0.1)].tolist()
print("\\nX-linked traits:", x_linked_traits)
"""

S1_CQ2C_STUDENT = """\
# Part B: Pairwise recombination frequencies (X-linked genes, scored in males).
males = fly_df[fly_df['sex'] == 1]

# The recombination frequency between two genes is the fraction of offspring that are
# *recombinant* — i.e. carry a NEW combination of the two genes' alleles. In these males a
# recombinant shows up as the phenotype for one trait DISAGREEING with the other.
def recombination_frequency(trait_a, trait_b):
    \"\"\"Fraction of males that are recombinant between trait_a and trait_b.\"\"\"
    # YOUR CODE HERE: a male is recombinant when his two trait phenotypes disagree.
    return ???                      # e.g. compare males[trait_a] with males[trait_b]

# Provided: build the pairwise recombination-frequency matrix using your function.
n_x = len(x_linked_traits)
recomb_freq = np.zeros((n_x, n_x))
for i, ti in enumerate(x_linked_traits):
    for j, tj in enumerate(x_linked_traits):
        recomb_freq[i, j] = recombination_frequency(ti, tj)
print("Pairwise recombination frequencies (males only):")
print(pd.DataFrame(recomb_freq, index=x_linked_traits, columns=x_linked_traits).round(3))
"""

S1_CQ2C_SOL = """\
males = fly_df[fly_df['sex'] == 1]
def recombination_frequency(trait_a, trait_b):
    return (males[trait_a] != males[trait_b]).mean()
n_x = len(x_linked_traits)
recomb_freq = np.zeros((n_x, n_x))
for i, ti in enumerate(x_linked_traits):
    for j, tj in enumerate(x_linked_traits):
        recomb_freq[i, j] = recombination_frequency(ti, tj)
print("Pairwise recombination frequencies (males only):")
print(pd.DataFrame(recomb_freq, index=x_linked_traits, columns=x_linked_traits).round(3))
"""

S1_CQ2D_STUDENT = """\
# Part C (provided): turn your recombination frequencies into a genetic map and plot it.
# This step is pure book-keeping — the genetics was in Parts A and B.
# Provided: the Haldane map function (recombination frequency → cM), the gene ordering, the
# ground-truth positions, and a plotting helper.

def haldane_d(r):
    \"\"\"Recombination frequency r → map distance in cM (Haldane).\"\"\"
    return -50 * np.log(1 - 2*np.clip(r, 0, 0.499))

D = haldane_d(recomb_freq)            # pairwise genetic distances (cM)
np.fill_diagonal(D, 0.0)              # distance from a gene to itself is 0

# On a *linear* chromosome the two genes that are FARTHEST apart are the two ENDS;
# ordering the rest by their distance from one end reconstructs the gene order.
i_end, j_end   = np.unravel_index(np.argmax(D), D.shape)
order          = np.argsort(D[i_end])
ordered_traits = [x_linked_traits[k] for k in order]
ordered_pos    = D[i_end][order]

print("Inferred gene order (cM from one end):")
for t, p in zip(ordered_traits, ordered_pos):
    print(f"  {t:14s} {p:5.1f} cM")

# Ground truth — check your inferred order/spacing against this (up to a left-right flip):
TRUE_POS = {'trait_eye': 0, 'trait_wing': 5, 'trait_leg': 15,
            'trait_notch': 30, 'trait_vein': 35, 'trait_scute': 50}

def plot_gene_map(predicted_pos, true_pos):
    \"\"\"Provided: plot your estimated gene map against the ground truth on two cM axes.
    Gene order is only defined up to a left-right flip, so we flip the estimate if that
    lines it up better with the truth.\"\"\"
    genes = list(true_pos)
    tpos  = np.array([true_pos[g] for g in genes], float)
    ppos  = np.array([predicted_pos[g] for g in genes], float)
    if np.corrcoef(ppos, tpos)[0, 1] < 0:        # flip so left/right match the truth
        ppos = ppos.max() - ppos
    pmap = dict(zip(genes, ppos))
    fig, ax = plt.subplots(figsize=(9, 2.8))
    for y, pos, col in [(1, tpos, '#4e79a7'), (0, ppos, '#e15759')]:
        ax.hlines(y, min(tpos.min(), ppos.min()), max(tpos.max(), ppos.max()),
                  color='lightgrey', zorder=1)
        ax.scatter(pos, [y]*len(pos), color=col, s=40, zorder=3)
        for g, x in zip(genes, pos):
            ax.text(x, y + 0.10, g.replace('trait_', ''), rotation=40,
                    ha='left', va='bottom', fontsize=8)
    for g in genes:                               # connect the same gene across the two maps
        ax.plot([true_pos[g], pmap[g]], [1, 0], color='grey', lw=0.6, ls=':', zorder=2)
    ax.set_yticks([0, 1]); ax.set_yticklabels(['estimated', 'true'])
    ax.set_ylim(-0.5, 1.7); ax.set_xlabel('position (cM)')
    ax.set_title('Estimated vs true gene order'); plt.tight_layout(); plt.show()

# Build your inferred map as a dict and plot it against the ground truth.
predicted_pos = dict(zip(ordered_traits, ordered_pos))
plot_gene_map(predicted_pos, TRUE_POS)
print("\\nGround truth (cM):", TRUE_POS)
"""

# ── Challenge 5: Ascertainment by age of onset ───────────────────────────────
S1_CQ4_MD = """\
### Challenge {N}: Ascertainment by age of onset — ~10 min

Disease cohorts are often **ascertained** — individuals only enter as *cases* once they have
been diagnosed. For a late-onset disease, someone who will eventually develop it but is still
young looks like a *control* at recruitment.

Model this: treat the binary trait as a late-onset disease and **recode every case younger than
60 as a control**, then re-run the logistic GWAS. Compare the hits to the fully-ascertained
baseline. What happens to power, and why?
"""

S1_CQ4_STUDENT = """\
# Age-of-onset ascertainment
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

# ── Challenge 6: Sex-stratified GWAS ─────────────────────────────────────────
S1_CQSEX_MD = """\
### Challenge {N}: Sex-stratified GWAS — ~10 min

Effects can differ between the sexes. The standard pooled GWAS estimates a single effect per
variant — so a variant that acts in *opposite* directions in males and females averages out to
look null. Run the continuous-trait GWAS **separately in males and females** and compare the
per-variant effect sizes. Is there a variant that the pooled GWAS misses?
"""

S1_CQSEX_STUDENT = """\
# GWAS separately in males and females
male = sex == 1
female = sex == 0

# Within a single sex, sex is constant — adjust for age only.
cov_m = ((age[male]   - age.mean()) / age.std()).reshape(-1, 1)
cov_f = ((age[female] - age.mean()) / age.std()).reshape(-1, 1)

# YOUR CODE HERE: run the GWAS in each sex
betas_m, _, pvals_m = run_gwas(???, ???, ???)
betas_f, _, pvals_f = run_gwas(???, ???, ???)

# Compare effect sizes at variants significant in either stratum
sel = np.where((pvals_m < 1e-5) | (pvals_f < 1e-5))[0]
fig, ax = plt.subplots(figsize=(5.5, 5.5))
ax.scatter(betas_m[sel], betas_f[sel], s=25, alpha=0.6, color='steelblue', label='sig in either')
if sexspec_idx_qc >= 0:
    ax.scatter(betas_m[sexspec_idx_qc], betas_f[sexspec_idx_qc], s=160, marker='*',
               color='red', zorder=5, label='sex-specific locus')
lim = np.abs(np.r_[betas_m[sel], betas_f[sel]]).max() * 1.15
ax.plot([-lim, lim], [-lim, lim], 'k--', label='y = x'); ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
ax.axhline(0, color='grey', lw=0.5); ax.axvline(0, color='grey', lw=0.5)
ax.set_xlabel('Effect in males'); ax.set_ylabel('Effect in females')
ax.set_title('Sex-stratified effect sizes'); ax.legend(fontsize=8)
plt.tight_layout(); plt.show()

if sexspec_idx_qc >= 0:
    print(f"Sex-specific locus {rsids_qc[sexspec_idx_qc]}:")
    print(f"  pooled   p = {pvals_cov[sexspec_idx_qc]:.1e}  (looks null)")
    print(f"  males    beta={betas_m[sexspec_idx_qc]:+.3f}  p={pvals_m[sexspec_idx_qc]:.1e}")
    print(f"  females  beta={betas_f[sexspec_idx_qc]:+.3f}  p={pvals_f[sexspec_idx_qc]:.1e}")
print("Q: Which variant lies off the y=x line, and why does the pooled GWAS miss it?")
"""

S1_CQSEX_SOL = """\
male = sex == 1
female = sex == 0
cov_m = ((age[male]   - age.mean()) / age.std()).reshape(-1, 1)
cov_f = ((age[female] - age.mean()) / age.std()).reshape(-1, 1)

betas_m, _, pvals_m = run_gwas(y_cont[male],   G_qc[male],   cov_m)
betas_f, _, pvals_f = run_gwas(y_cont[female], G_qc[female], cov_f)

sel = np.where((pvals_m < 1e-5) | (pvals_f < 1e-5))[0]
fig, ax = plt.subplots(figsize=(5.5, 5.5))
ax.scatter(betas_m[sel], betas_f[sel], s=25, alpha=0.6, color='steelblue', label='sig in either')
if sexspec_idx_qc >= 0:
    ax.scatter(betas_m[sexspec_idx_qc], betas_f[sexspec_idx_qc], s=160, marker='*',
               color='red', zorder=5, label='sex-specific locus')
lim = np.abs(np.r_[betas_m[sel], betas_f[sel]]).max() * 1.15
ax.plot([-lim, lim], [-lim, lim], 'k--', label='y = x'); ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
ax.axhline(0, color='grey', lw=0.5); ax.axvline(0, color='grey', lw=0.5)
ax.set_xlabel('Effect in males'); ax.set_ylabel('Effect in females')
ax.set_title('Sex-stratified effect sizes'); ax.legend(fontsize=8)
plt.tight_layout(); plt.show()

if sexspec_idx_qc >= 0:
    print(f"Sex-specific locus {rsids_qc[sexspec_idx_qc]}:")
    print(f"  pooled   p = {pvals_cov[sexspec_idx_qc]:.1e}  (looks null)")
    print(f"  males    beta={betas_m[sexspec_idx_qc]:+.3f}  p={pvals_m[sexspec_idx_qc]:.1e}")
    print(f"  females  beta={betas_f[sexspec_idx_qc]:+.3f}  p={pvals_f[sexspec_idx_qc]:.1e}")
# The sex-specific locus sits off the y=x line (opposite-sign effects); pooled it averages ~0.
"""

# ── Challenge 7: Polygenic scores (PGS) ──────────────────────────────────────
S1_CQ5_MD = """\
### Challenge {N}: Polygenic scores — predicting the genetic component — ~15 min

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
# Polygenic scores (train/test split)
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
LDL cholesterol (continuous), chronic ischaemic heart disease (binary), and body mass index
(highly polygenic) — for the Manhattan, QQ, trumpet and pleiotropy plots. We also bundle
**Genebass whole-exome** single-variant results for BMI (for the Miami-plot challenge). The setup
cell also re-runs the Session 1 *simulated* GWAS, which we still use for the analyses that need
individual-level genotypes (null-QQ contrast, winner's curse).

**Setup**: Run the setup cell once at the top (loads real sumstats + re-runs the simulated
GWAS, ~30s) before any exercises.
"""

# Loader for the bundled real Pan-UKB summary statistics (LDL / CAD / BMI, EUR).
# Powers the Manhattan, QQ, lambda_GC, MAF-stratified QQ, trumpet and pleiotropy plots.
# p-values are stored in -log10 units ('nlog10p') to avoid float underflow in the tail.
LOAD_REAL_SUMSTATS = """\
# ── Load bundled real GWAS summary statistics (Pan-UKB, EUR) ──────────────────
# Three real traits, parallel to the simulated ones:
#   ldl    — LDL direct          (continuous biomarker)        ~ like y_cont
#   cad    — chronic ischaemic heart disease (I25, binary)     ~ like y_bin
#   bmi    — body mass index     (highly polygenic continuous) ~ like y_poly
# Each trait provides: chrom, pos, maf, beta, se, nlog10p (= -log10 p), and two masks:
#   sig  = genome-wide significant (p < 5e-8); rand = unbiased random subset for QQ/lambda.
_real_path = os.path.join(DATA_DIR, 'sumstats_real.npz')
if not os.path.exists(_real_path):
    _url = f'https://media.githubusercontent.com/media/{REPO_SLUG}/{BRANCH_NAME}/data/sumstats_real.npz'
    print('Downloading sumstats_real.npz from GitHub ...')
    import urllib.request; urllib.request.urlretrieve(_url, _real_path)
_rs = np.load(_real_path, allow_pickle=True)
REAL_TRAITS = ['ldl', 'cad', 'bmi']
REAL_LABELS = {'ldl': 'LDL direct (continuous)',
               'cad': 'Chronic ischaemic heart disease (binary)',
               'bmi': 'Body mass index (polygenic)'}
real = {}
for _t in REAL_TRAITS:
    real[_t] = {k: _rs[f'{_t}_{k}'] for k in
                ['chrom', 'pos', 'maf', 'beta', 'se', 'nlog10p', 'sig', 'rand']}
print("Real Pan-UKB summary statistics loaded:")
for _t in REAL_TRAITS:
    d = real[_t]
    print(f"  {_t:6s} ({REAL_LABELS[_t]}): {len(d['pos']):,} SNPs  "
          f"({int(d['sig'].sum()):,} genome-wide sig, {int(d['rand'].sum()):,} in QQ subset)")

# ── Load bundled Genebass BMI exome single-variant results (for the Miami challenge) ──
# Whole-exome single-variant associations for BMI (Genebass, UK Biobank). Columns include
# 'Variant ID' (GRCh38, formatted chr-pos-ref-alt), 'CSQ' (variant consequence), 'Beta',
# 'P-Value'. NOTE: these are GRCh38 coordinates, whereas Pan-UKB above is GRCh37.
_gb_path = os.path.join(DATA_DIR, 'genebass_bmi_exomes.csv')
if not os.path.exists(_gb_path):
    _gburl = f'https://raw.githubusercontent.com/{REPO_SLUG}/{BRANCH_NAME}/data/genebass_bmi_exomes.csv'
    print('Downloading genebass_bmi_exomes.csv from GitHub ...')
    import urllib.request; urllib.request.urlretrieve(_gburl, _gb_path)
genebass = pd.read_csv(_gb_path).dropna(subset=['Variant ID']).copy()
_vid = genebass['Variant ID'].str.split('-', expand=True)     # chr-pos-ref-alt (GRCh38)
genebass['chrom']   = _vid[0]
genebass['pos']     = pd.to_numeric(_vid[1], errors='coerce')  # GRCh38 base-pair position
genebass['beta']    = pd.to_numeric(genebass['Beta'], errors='coerce')
genebass['pval']    = pd.to_numeric(genebass['P-Value'], errors='coerce')
genebass = genebass.dropna(subset=['pos', 'beta', 'pval'])
genebass['nlog10p'] = -np.log10(genebass['pval'].clip(lower=1e-300))
print(f"Genebass BMI exome variants: {len(genebass):,}  "
      f"({genebass['CSQ'].nunique()} consequence types)")

# ── Load chr16 gene models + GRCh37->GRCh38-lifted BMI GWAS (for the gene-track challenge) ──
# MANE Select gene/exon/CDS models on chr16 (GRCh38) plus the Pan-UKB BMI GWAS chr16 variants
# lifted to GRCh38 — so the common-variant GWAS, the Genebass exome variants and the gene models
# all share ONE coordinate system (no per-plot fudging). Built by fetch_gene_models.py.
_gm_path = os.path.join(DATA_DIR, 'gene_models_chr16.npz')
if not os.path.exists(_gm_path):
    _gmurl = f'https://media.githubusercontent.com/media/{REPO_SLUG}/{BRANCH_NAME}/data/gene_models_chr16.npz'
    print('Downloading gene_models_chr16.npz from GitHub ...')
    import urllib.request; urllib.request.urlretrieve(_gmurl, _gm_path)
genemodels = dict(np.load(_gm_path, allow_pickle=True))
print(f"chr16 gene models: {len(genemodels['gene_name']):,} MANE genes, "
      f"{len(genemodels['cds_gi']):,} coding exons; "
      f"{len(genemodels['gw_pos38']):,} GWAS variants lifted to GRCh38")

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
# We now use REAL Pan-UKB summary statistics (LDL, CAD, BMI) across all
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
# ── Pleiotropy: signed-beta scatter for LDL vs CAD ────────────────────────────
# A pleiotropy scatter plots the signed effect size (beta) of one trait vs another,
# for variants present in BOTH GWAS. Shared genetic effects show up as a tilted cloud.
# LDL and CAD are causally linked, so we expect their effect sizes to be correlated.

def merge_betas(ta, tb):
    \"\"\"Align two real traits on chrom:pos; return (beta_a, beta_b, sig_a, sig_b).\"\"\"
    da, db = real[ta], real[tb]
    key_a = da['chrom'].astype(np.int64) * 10**9 + da['pos']
    key_b = db['chrom'].astype(np.int64) * 10**9 + db['pos']
    common, ia, ib = np.intersect1d(key_a, key_b, return_indices=True)
    return (da['beta'][ia], db['beta'][ib],
            da['sig'][ia],  db['sig'][ib], len(common))

trait_a, trait_b = 'ldl', 'cad'
beta_a, beta_b, sig_a, sig_b, n_common = merge_betas(trait_a, trait_b)
sig = sig_a | sig_b

fig, ax = plt.subplots(figsize=(6, 6))
ns = np.where(~sig)[0]; ns = ns[_thin(len(ns), k=8000)]          # thin grey cloud for display
ax.scatter(beta_a[ns], beta_b[ns], s=2, alpha=0.1, color='grey', rasterized=True)
si = np.where(sig)[0]; si = si[_thin(len(si), k=6000, seed=2)]   # thin significant points
ax.scatter(beta_a[si], beta_b[si], s=12, alpha=0.6, color='#e15759', zorder=5,
           label=f'sig in either (n={int(sig.sum()):,})')
ax.axhline(0, color='black', lw=0.5); ax.axvline(0, color='black', lw=0.5)
ax.set_xlabel(f'beta — {trait_a}'); ax.set_ylabel(f'beta — {trait_b}')
r = np.corrcoef(beta_a[sig], beta_b[sig])[0, 1] if sig.sum() > 2 else float('nan')
ax.set_title(f'Pleiotropy: {trait_a} vs {trait_b}\\n'
             f'(r={r:.2f} at sig variants, {n_common:,} shared SNPs)')
ax.legend(fontsize=8); plt.tight_layout(); plt.show()
print("Q: LDL and CAD are causally linked — does the LDL–CAD beta cloud tilt positive?")
print("Q: A variant with a large LDL effect but near-zero CAD effect — what does that imply?")
"""

S2_CQ_MD = """\
---

## Challenge Questions
"""

S2_CQ1_MD = """\
### Challenge {N}: QQ plot with 95% confidence interval — ~10 min

Under the null, the $k$-th smallest p-value follows a Beta$(k, n-k+1)$ distribution.
Use `scipy.stats.beta.ppf` to add a 95% confidence band to the QQ plot.

To see what the band is *for*, we draw two QQ plots side by side: a genuinely **near-null** trait
(the shuffled simulated phenotype from the null-QQ cell — its points should sit inside the band),
and the **real BMI** GWAS (a hugely polygenic trait — its points should shoot far above the band).
"""

S2_CQ1_STUDENT = """\
# QQ plot with a 95% confidence band, drawn for a near-null trait and for real BMI.

def qq_plot_ci(nlog10p, ax=None, color='steelblue', ci_alpha=0.05, label=''):
    if ax is None:
        fig, ax = plt.subplots(figsize=(5, 5))
    n   = len(nlog10p)
    observed_sorted = np.sort(nlog10p)     # ascending observed -log10(p)
    ks  = np.arange(n, 0, -1)              # matching order index for ascending obs

    # YOUR CODE HERE
    # Expected quantiles and 95% CI bounds using the Beta distribution.
    # The k-th smallest p-value ~ Beta(k, n-k+1).
    expected_median = ???   # stats.beta.ppf(0.5, ks, n-ks+1)
    ci_lower        = ???   # stats.beta.ppf(ci_alpha/2, ks, n-ks+1)
    ci_upper        = ???   # stats.beta.ppf(1 - ci_alpha/2, ks, n-ks+1)

    ax.fill_between(-np.log10(expected_median),
                    -np.log10(ci_upper),    # upper p-bound → lower -log10
                    -np.log10(ci_lower),
                    alpha=0.2, color=color, label='95% CI')
    ax.scatter(-np.log10(expected_median), observed_sorted, s=2, alpha=0.6, color=color, label=label)
    expected_max = (-np.log10(expected_median)).max()
    ax.plot([0, expected_max], [0, expected_max], 'r--', linewidth=1.0)
    ax.set_xlim(0, expected_max)            # truncate x-axis to max expected
    ax.set_xlabel(r'Expected $-\\log_{10}(p)$'); ax.set_ylabel(r'Observed $-\\log_{10}(p)$')
    ax.legend(fontsize=8)
    return ax

# The near-null trait: the shuffled simulated phenotype (pvals_null from the null-QQ cell).
nlog10p_null = -np.log10(np.clip(pvals_null, 1e-300, 1))

fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))
qq_plot_ci(nlog10p_null, ax=axes[0], color='grey', label='near-null (shuffled)')
axes[0].set_title('Near-null trait — stays inside the band')
qq_plot_ci(real['bmi']['nlog10p'][real['bmi']['rand']], ax=axes[1], color='steelblue', label='BMI')
axes[1].set_title('Real BMI GWAS — escapes the band (polygenic)')
plt.tight_layout(); plt.show()
print("Q: The null trait stays in the band but BMI shoots above it — is BMI's lift-off")
print("   a sign of confounding, or of genuine genome-wide polygenic signal?")
"""

S2_CQ1_SOL = """\
def qq_plot_ci(nlog10p, ax=None, color='steelblue', ci_alpha=0.05, label=''):
    if ax is None:
        fig, ax = plt.subplots(figsize=(5, 5))
    n = len(nlog10p); observed_sorted = np.sort(nlog10p); ks = np.arange(n, 0, -1)
    d = np.unique(np.r_[_thin(n, k=4000), np.where(observed_sorted >= 2)[0]])  # thin bulk, keep tail
    observed_sorted, ks = observed_sorted[d], ks[d]
    expected_median = stats.beta.ppf(0.5, ks, n-ks+1)
    ci_lower        = stats.beta.ppf(ci_alpha/2, ks, n-ks+1)
    ci_upper        = stats.beta.ppf(1 - ci_alpha/2, ks, n-ks+1)
    ax.fill_between(-np.log10(expected_median), -np.log10(ci_upper), -np.log10(ci_lower),
                    alpha=0.2, color=color, label='95% CI')
    ax.scatter(-np.log10(expected_median), observed_sorted, s=2, alpha=0.6, color=color, label=label)
    expected_max = (-np.log10(expected_median)).max()
    ax.plot([0, expected_max], [0, expected_max], 'r--', linewidth=1.0); ax.set_xlim(0, expected_max)
    ax.set_xlabel(r'Expected $-\\log_{10}(p)$'); ax.set_ylabel(r'Observed $-\\log_{10}(p)$')
    ax.legend(fontsize=8); return ax

nlog10p_null = -np.log10(np.clip(pvals_null, 1e-300, 1))

fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))
qq_plot_ci(nlog10p_null, ax=axes[0], color='grey', label='near-null (shuffled)')
axes[0].set_title('Near-null trait — stays inside the band')
qq_plot_ci(real['bmi']['nlog10p'][real['bmi']['rand']], ax=axes[1], color='steelblue', label='BMI')
axes[1].set_title('Real BMI GWAS — escapes the band (polygenic)')
plt.tight_layout(); plt.show()
"""

S2_CQ2_MD = """\
### Challenge {N}: MAF-stratified QQ plot — ~8 min

Are rare variants (low MAF) better or worse behaved than common variants?
Stratify the real BMI variants into MAF bins and overlay QQ plots (random subset).
"""

S2_CQ2_STUDENT = """\
# MAF-stratified QQ plot (real BMI GWAS, random subset).
sub   = real['bmi']['rand']
maf_h = real['bmi']['maf'][sub]
nlp_h = real['bmi']['nlog10p'][sub]

# Provided: QQ coordinates (expected vs observed -log10 p) for a set of p-values.
def qq_coords(nlog10p):
    \"\"\"Return (expected, observed) -log10(p) for a QQ plot, thinned for display.\"\"\"
    obs  = np.sort(nlog10p); m = len(obs)
    exp  = -np.log10(np.arange(m, 0, -1) / (m + 1))
    keep = np.unique(np.r_[_thin(m, k=4000), np.where(obs >= 2)[0]])   # thin bulk, keep tail
    return exp[keep], obs[keep]

# YOUR CODE HERE: choose how to split variants by minor allele frequency, from rare to common.
maf_bins = ???        # list of (low, high) MAF ranges, e.g. [(0.005, 0.01), (0.01, 0.05), ...]
colours  = ['#b07aa1', '#e15759', '#f28e2b', '#4e79a7']

fig, ax = plt.subplots(figsize=(6, 6)); exp_max = 0
for (lo, hi), col in zip(maf_bins, colours):
    mask = (maf_h >= lo) & (maf_h < hi)
    if mask.sum() < 10:
        continue
    exp, obs = qq_coords(nlp_h[mask])
    ax.scatter(exp, obs, s=3, alpha=0.5, color=col, label=f'MAF [{lo:.1%}, {hi:.0%})')
    exp_max = max(exp_max, exp.max())

ax.plot([0, exp_max], [0, exp_max], 'k--', linewidth=1); ax.set_xlim(0, exp_max)
ax.set_title('MAF-stratified QQ plot — real BMI GWAS')
ax.set_xlabel(r'Expected $-\\log_{10}(p)$'); ax.set_ylabel(r'Observed $-\\log_{10}(p)$')
ax.legend(); plt.tight_layout(); plt.show()
print("Q: Common variants carry most of BMI's association signal — does the plot agree?")
"""

S2_CQ2_SOL = """\
sub   = real['bmi']['rand']
maf_h = real['bmi']['maf'][sub]
nlp_h = real['bmi']['nlog10p'][sub]

def qq_coords(nlog10p):
    obs  = np.sort(nlog10p); m = len(obs)
    exp  = -np.log10(np.arange(m, 0, -1) / (m + 1))
    keep = np.unique(np.r_[_thin(m, k=4000), np.where(obs >= 2)[0]])
    return exp[keep], obs[keep]

maf_bins = [(0.005, 0.01), (0.01, 0.05), (0.05, 0.15), (0.15, 0.50)]
colours  = ['#b07aa1', '#e15759', '#f28e2b', '#4e79a7']

fig, ax = plt.subplots(figsize=(6, 6)); exp_max = 0
for (lo, hi), col in zip(maf_bins, colours):
    mask = (maf_h >= lo) & (maf_h < hi)
    if mask.sum() < 10:
        continue
    exp, obs = qq_coords(nlp_h[mask])
    ax.scatter(exp, obs, s=3, alpha=0.5, color=col, label=f'MAF [{lo:.1%}, {hi:.0%})')
    exp_max = max(exp_max, exp.max())
ax.plot([0, exp_max], [0, exp_max], 'k--', linewidth=1); ax.set_xlim(0, exp_max)
ax.set_title('MAF-stratified QQ plot — real BMI GWAS')
ax.set_xlabel(r'Expected $-\\log_{10}(p)$'); ax.set_ylabel(r'Observed $-\\log_{10}(p)$')
ax.legend(); plt.tight_layout(); plt.show()
"""

S2_CQ3_MD = """\
### Challenge {N}: Trumpet plot and power curves — ~12 min

A **trumpet plot** shows the signed effect size ($\\hat{\\beta}$) vs. MAF for each variant.
The "trumpet" shape emerges because the power curves form a ±symmetric band:
variants outside the band are detectable, those inside are below the noise floor.
Points beyond the band on the left (rare variants) require large effects to be detected.

We use the **real LDL GWAS** (Pan-UKB, EUR, N≈400k). Real discovered variants should sit
on or beyond the power band for a study of that size.
"""

S2_CQ3_STUDENT = """\
# Trumpet plot with power curves (signed beta) — real LDL GWAS

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

# Provided: scatter signed beta vs MAF — grey for the random subset, coloured for the GWAS hits.
ax.scatter(maf_l[rnd_l], beta_l[rnd_l], s=3, alpha=0.15, color='grey', rasterized=True)
ax.scatter(maf_l[sig_l], beta_l[sig_l], s=12, alpha=0.6, color='steelblue', label='genome-wide sig')

# A bigger study can detect smaller effects, so its power band is narrower. Overlay the ±power
# curves for a few study sizes — including one close to the real LDL GWAS (N ≈ 400k).
# YOUR CODE HERE: choose the study sizes (number of individuals) to compare.
study_sizes = ???        # e.g. [10_000, 100_000, 400_000]
for n_samples, col in zip(study_sizes, ['#e15759', '#f28e2b', '#59a14f']):
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
### Challenge {N}: Winner's curse — ~15 min

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
# Winner's curse — discovery vs validation across resampled 5k/5k splits
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

S2_MIAMI_MD = """\
### Challenge {N}: GWAS vs exome — a BMI Miami plot — ~12 min

A **Miami plot** stacks two association scans back-to-back on one position axis: here the **Pan-UKB
common-variant GWAS** for BMI points **up**, and **Genebass whole-exome single-variant** results for
BMI point **down**. The common-variant GWAS tags broad regulatory regions; the exome scan pinpoints
**coding** variants in genes.

We zoom into a single chromosome (default **chr16**, home of the *FTO* common-variant peak). Here the
Pan-UKB GWAS has been **lifted from GRCh37 to GRCh38** (see `fetch_gene_models.py`), so it shares **one
coordinate system** with the Genebass exome data and with a **gene / coding-exon track** drawn beneath
the plot. Across chr16 the **exome** hits sit **on coding exons** (that is all the exome assay can see),
whereas the **common-variant** GWAS tags broad regulatory regions. Zooming into *FTO* shows the classic
picture: the GWAS peak sits in *FTO*'s **first intron** with **no coding exon beneath it** — and there
are **no exome/coding hits inside *FTO* at all** (the signal is regulatory, acting at a distance on
*IRX3/IRX5*). After plotting, compare the **effect sizes of significant exome variants by consequence**
(synonymous / missense / pLoF).
"""

S2_MIAMI_STUDENT = """\
# Part A: a BMI Miami plot for chr16 on ONE coordinate system (GRCh38).
# Pan-UKB common-variant GWAS points UP; Genebass exome single-variant results point DOWN.
MIAMI_CHROM = 16

# Pan-UKB common-variant GWAS on chr16, lifted GRCh37->GRCh38 so it lines up with the exome data.
gwas_pos_mb  = genemodels['gw_pos38'] / 1e6
gwas_nlog10p = genemodels['gw_nlog10p']

# Genebass exome single-variant results on chr16 (already GRCh38).
exome_on_chrom = genebass[genebass['chrom'] == str(MIAMI_CHROM)]
exome_pos_mb   = exome_on_chrom['pos'].values / 1e6
exome_nlog10p  = exome_on_chrom['nlog10p'].values

fig, ax = plt.subplots(figsize=(12, 5))
# Top half: GWAS pointing up.
ax.scatter(gwas_pos_mb, gwas_nlog10p, s=8, alpha=0.5, color='steelblue',
           label='Pan-UKB GWAS — common variants (GRCh38, lifted)')
# YOUR CODE HERE: bottom half — plot the Genebass exome variants pointing DOWN
# (i.e. at y = -exome_nlog10p).
ax.scatter(exome_pos_mb, ???, s=16, alpha=0.7, color='#e15759',
           label='Genebass exome — single variants (GRCh38)')

ax.axhline(0, color='black', lw=0.8)
ax.axhline(-np.log10(5e-8), color='grey', ls='--', lw=0.8)   # genome-wide line (top)
ax.axhline( np.log10(5e-8), color='grey', ls='--', lw=0.8)   # genome-wide line (bottom)
ax.set_yticklabels([f'{abs(t):.0f}' for t in ax.get_yticks()])  # show |−log10 p| on both halves
ax.set_xlabel(f'Position on chr{MIAMI_CHROM} (Mb, GRCh38)')
ax.set_ylabel(r'$-\\log_{10}(p)$   (GWAS up / exome down)')
ax.set_title(f'BMI Miami plot, chr{MIAMI_CHROM} (GRCh38): common-variant GWAS vs exome')
ax.legend(fontsize=8, loc='upper right'); plt.tight_layout(); plt.show()

# Provided: zoom into FTO and draw a gene / coding-exon track underneath. Because everything is now
# on GRCh38, we can read off WHICH variants fall in coding sequence.
def gene_track(ax_track, x0_mb, x1_mb):
    \"\"\"chr16 genes in [x0_mb, x1_mb]: thin line = gene body, thick black = coding exon (CDS).\"\"\"
    names = genemodels['gene_name']
    for gi in range(len(names)):
        gs, ge = genemodels['gene_start'][gi] / 1e6, genemodels['gene_end'][gi] / 1e6
        if ge < x0_mb or gs > x1_mb:
            continue
        ax_track.plot([max(gs, x0_mb), min(ge, x1_mb)], [0, 0], color='grey', lw=1, zorder=1)
        if x0_mb <= (gs + ge) / 2 <= x1_mb:
            ax_track.annotate(names[gi], ((gs + ge) / 2, 0), xytext=(0, 7),
                              textcoords='offset points', ha='center', fontsize=7, color='darkred')
    for s, e in zip(genemodels['cds_start'], genemodels['cds_end']):
        if e / 1e6 < x0_mb or s / 1e6 > x1_mb:
            continue
        ax_track.plot([s / 1e6, e / 1e6], [0, 0], color='black', lw=7, solid_capstyle='butt', zorder=2)
    ax_track.set_yticks([]); ax_track.set_ylim(-0.6, 1.0); ax_track.set_xlim(x0_mb, x1_mb)

_fto = list(genemodels['gene_name']).index('FTO')          # FTO gene window (GRCh38), padded
fto_s, fto_e = genemodels['gene_start'][_fto] / 1e6, genemodels['gene_end'][_fto] / 1e6
pad = 0.15 * (fto_e - fto_s); ZOOM = (fto_s - pad, fto_e + pad)

fig, (axz, axg) = plt.subplots(2, 1, figsize=(11, 6), sharex=True,
                               gridspec_kw={'height_ratios': [4, 1]})
zg = (gwas_pos_mb >= ZOOM[0]) & (gwas_pos_mb <= ZOOM[1])
ze = (exome_pos_mb >= ZOOM[0]) & (exome_pos_mb <= ZOOM[1])
axz.scatter(gwas_pos_mb[zg], gwas_nlog10p[zg], s=14, alpha=0.6, color='steelblue', label='GWAS (common)')
axz.scatter(exome_pos_mb[ze], -exome_nlog10p[ze], s=24, alpha=0.8, color='#e15759', label='exome (coding)')
axz.axhline(0, color='black', lw=0.8); axz.axhline(-np.log10(5e-8), color='grey', ls='--', lw=0.8)
axz.set_yticklabels([f'{abs(t):.0f}' for t in axz.get_yticks()])
axz.set_ylabel(r'$-\\log_{10}(p)$'); axz.legend(fontsize=8, loc='upper right')
axz.set_title('Zoom on FTO (chr16, GRCh38): GWAS up / exome down')
gene_track(axg, *ZOOM)
axg.set_xlabel('Position on chr16 (Mb, GRCh38)'); axg.set_ylabel('genes\\n(CDS thick)', fontsize=8)
plt.tight_layout(); plt.show()
print("Q: The GWAS peak sits over FTO intron 1 (no coding exon beneath it), and there are no")
print("   exome/coding hits inside FTO at all. What does this regulatory (non-coding) common-variant")
print("   signal say about how common variants act, vs the coding variants the exome scan can see?")
"""

S2_MIAMI_SOL = """\
MIAMI_CHROM = 16
gwas_pos_mb  = genemodels['gw_pos38'] / 1e6          # GWAS lifted GRCh37->GRCh38
gwas_nlog10p = genemodels['gw_nlog10p']
exome_on_chrom = genebass[genebass['chrom'] == str(MIAMI_CHROM)]
exome_pos_mb   = exome_on_chrom['pos'].values / 1e6
exome_nlog10p  = exome_on_chrom['nlog10p'].values

fig, ax = plt.subplots(figsize=(12, 5))
ax.scatter(gwas_pos_mb, gwas_nlog10p, s=8, alpha=0.5, color='steelblue',
           label='Pan-UKB GWAS — common variants (GRCh38, lifted)')
ax.scatter(exome_pos_mb, -exome_nlog10p, s=16, alpha=0.7, color='#e15759',   # KEY: exome points DOWN
           label='Genebass exome — single variants (GRCh38)')
ax.axhline(0, color='black', lw=0.8)
ax.axhline(-np.log10(5e-8), color='grey', ls='--', lw=0.8)
ax.axhline( np.log10(5e-8), color='grey', ls='--', lw=0.8)
ax.set_yticklabels([f'{abs(t):.0f}' for t in ax.get_yticks()])
ax.set_xlabel(f'Position on chr{MIAMI_CHROM} (Mb, GRCh38)')
ax.set_ylabel(r'$-\\log_{10}(p)$   (GWAS up / exome down)')
ax.set_title(f'BMI Miami plot, chr{MIAMI_CHROM} (GRCh38): common-variant GWAS vs exome')
ax.legend(fontsize=8, loc='upper right'); plt.tight_layout(); plt.show()

def gene_track(ax_track, x0_mb, x1_mb):
    \"\"\"chr16 genes in [x0_mb, x1_mb]: thin line = gene body, thick black = coding exon (CDS).\"\"\"
    names = genemodels['gene_name']
    for gi in range(len(names)):
        gs, ge = genemodels['gene_start'][gi] / 1e6, genemodels['gene_end'][gi] / 1e6
        if ge < x0_mb or gs > x1_mb:
            continue
        ax_track.plot([max(gs, x0_mb), min(ge, x1_mb)], [0, 0], color='grey', lw=1, zorder=1)
        if x0_mb <= (gs + ge) / 2 <= x1_mb:
            ax_track.annotate(names[gi], ((gs + ge) / 2, 0), xytext=(0, 7),
                              textcoords='offset points', ha='center', fontsize=7, color='darkred')
    for s, e in zip(genemodels['cds_start'], genemodels['cds_end']):
        if e / 1e6 < x0_mb or s / 1e6 > x1_mb:
            continue
        ax_track.plot([s / 1e6, e / 1e6], [0, 0], color='black', lw=7, solid_capstyle='butt', zorder=2)
    ax_track.set_yticks([]); ax_track.set_ylim(-0.6, 1.0); ax_track.set_xlim(x0_mb, x1_mb)

_fto = list(genemodels['gene_name']).index('FTO')
fto_s, fto_e = genemodels['gene_start'][_fto] / 1e6, genemodels['gene_end'][_fto] / 1e6
pad = 0.15 * (fto_e - fto_s); ZOOM = (fto_s - pad, fto_e + pad)

fig, (axz, axg) = plt.subplots(2, 1, figsize=(11, 6), sharex=True,
                               gridspec_kw={'height_ratios': [4, 1]})
zg = (gwas_pos_mb >= ZOOM[0]) & (gwas_pos_mb <= ZOOM[1])
ze = (exome_pos_mb >= ZOOM[0]) & (exome_pos_mb <= ZOOM[1])
axz.scatter(gwas_pos_mb[zg], gwas_nlog10p[zg], s=14, alpha=0.6, color='steelblue', label='GWAS (common)')
axz.scatter(exome_pos_mb[ze], -exome_nlog10p[ze], s=24, alpha=0.8, color='#e15759', label='exome (coding)')
axz.axhline(0, color='black', lw=0.8); axz.axhline(-np.log10(5e-8), color='grey', ls='--', lw=0.8)
axz.set_yticklabels([f'{abs(t):.0f}' for t in axz.get_yticks()])
axz.set_ylabel(r'$-\\log_{10}(p)$'); axz.legend(fontsize=8, loc='upper right')
axz.set_title('Zoom on FTO (chr16, GRCh38): GWAS up / exome down')
gene_track(axg, *ZOOM)
axg.set_xlabel('Position on chr16 (Mb, GRCh38)'); axg.set_ylabel('genes\\n(CDS thick)', fontsize=8)
plt.tight_layout(); plt.show()
# FTO intron-1 GWAS peak is regulatory (acts on IRX3/IRX5) with NO coding hits in FTO itself; the
# exome's coding hits sit elsewhere on chr16. Common signals tag regulatory DNA; exomes see only CDS.
"""

S2_CSQ_STUDENT = """\
# Part B: among significant exome variants, do effect sizes differ by consequence?
EXOME_SIG = 1e-6                                  # exome-wide significance threshold
significant_exome = genebass[genebass['pval'] < EXOME_SIG].copy()
print("Consequence (CSQ) types present:", sorted(significant_exome['CSQ'].unique()))

# A variant's consequence (CSQ) describes what it does to the protein. We compare three classes:
# synonymous (silent), missense (one amino-acid change), and pLoF (protein loss-of-function).
# YOUR CODE HERE: which CSQ values are protein-loss-of-function (truncating/disrupting the protein)?
PLOF = ???      # a set of CSQ strings, e.g. {'stop_gained', 'frameshift_variant', ...}

def consequence_group(csq):
    if csq in PLOF:                 return 'pLoF'          # protein loss-of-function
    if csq == 'missense_variant':   return 'missense'
    if csq == 'synonymous_variant': return 'synonymous'
    return 'other'
significant_exome['group'] = significant_exome['CSQ'].map(consequence_group)

groups = ['synonymous', 'missense', 'pLoF']
# Provided: the absolute effect sizes |Beta| for each consequence class.
abs_beta_by_group = [ significant_exome.loc[significant_exome['group'] == g, 'beta'].abs().values
                      for g in groups ]

fig, ax = plt.subplots(figsize=(7, 5))
ax.boxplot(abs_beta_by_group, showfliers=False)
ax.set_xticks(range(1, len(groups) + 1))
ax.set_xticklabels([f'{g}\\n(n={len(a)})' for g, a in zip(groups, abs_beta_by_group)])
for i, a in enumerate(abs_beta_by_group, start=1):              # jittered points
    ax.scatter(np.random.default_rng(i).normal(i, 0.05, len(a)), a, s=10, alpha=0.4, color='grey')
ax.set_ylabel('|effect size|  (|Beta|, IRNT units)')
ax.set_title(f'BMI exome variants (p < {EXOME_SIG:g}): effect size by consequence')
plt.tight_layout(); plt.show()
for g, a in zip(groups, abs_beta_by_group):
    print(f"  {g:11s}: n={len(a):4d}, median |beta| = {np.median(a):.4f}")
print("Q: Which consequence class carries the largest per-allele effects, and why?")
"""

S2_CSQ_SOL = """\
EXOME_SIG = 1e-6
significant_exome = genebass[genebass['pval'] < EXOME_SIG].copy()
PLOF = {'stop_gained', 'frameshift_variant', 'splice_donor_variant',
        'splice_acceptor_variant', 'start_lost'}
def consequence_group(csq):
    if csq in PLOF:                 return 'pLoF'
    if csq == 'missense_variant':   return 'missense'
    if csq == 'synonymous_variant': return 'synonymous'
    return 'other'
significant_exome['group'] = significant_exome['CSQ'].map(consequence_group)

groups = ['synonymous', 'missense', 'pLoF']
abs_beta_by_group = [ significant_exome.loc[significant_exome['group'] == g, 'beta'].abs().values
                      for g in groups ]

fig, ax = plt.subplots(figsize=(7, 5))
ax.boxplot(abs_beta_by_group, showfliers=False)
ax.set_xticks(range(1, len(groups) + 1))
ax.set_xticklabels([f'{g}\\n(n={len(a)})' for g, a in zip(groups, abs_beta_by_group)])
for i, a in enumerate(abs_beta_by_group, start=1):
    ax.scatter(np.random.default_rng(i).normal(i, 0.05, len(a)), a, s=10, alpha=0.4, color='grey')
ax.set_ylabel('|effect size|  (|Beta|, IRNT units)')
ax.set_title(f'BMI exome variants (p < {EXOME_SIG:g}): effect size by consequence')
plt.tight_layout(); plt.show()
for g, a in zip(groups, abs_beta_by_group):
    print(f"  {g:11s}: n={len(a):4d}, median |beta| = {np.median(a):.4f}")
# pLoF > missense > synonymous: disrupting the protein has larger phenotypic consequences than
# a synonymous change, which is (mostly) silent.
"""

S2_SIDAK_MD = """\
### Challenge {N}: Bonferroni vs Šidák — ~8 min

The genome-wide threshold $5\\times10^{-8}$ is essentially a **Bonferroni** correction: for a
family-wise error rate $\\alpha_{FW}$ over $C$ tests, use per-test $\\alpha_{PT}=\\alpha_{FW}/C$
(with $\\alpha_{FW}=0.05$, $C\\approx10^6$ independent common variants).

**Šidák** is exact *under independence*. With $C$ independent tests each at per-test threshold
$\\alpha_{PT}$, the probability of **no** false positive is $(1-\\alpha_{PT})^{C}$, so the
family-wise error rate is

$$\\alpha_{FW} = 1 - (1-\\alpha_{PT})^{C}.$$

**Your job: solve this equation for the per-test threshold $\\alpha_{PT}$** (in terms of
$\\alpha_{FW}$ and $C$), implement it, and compare to Bonferroni ($\\alpha_{FW}/C$) across a range of
$C$ — including the **ratio** of the two thresholds (see Abdi, 2007).
"""

S2_SIDAK_STUDENT = """\
# Bonferroni vs Šidák per-test significance threshold, as a function of the number of tests C.
alpha_familywise = 0.05
C = np.logspace(0, 7, 200)                      # number of tests, from 1 to 10^7

bonferroni_threshold = alpha_familywise / C
# YOUR CODE HERE: solve  alpha_FW = 1 - (1 - alpha_PT)^C  for the per-test threshold alpha_PT,
# then evaluate it for every number of tests C.
sidak_threshold = ???
# YOUR CODE HERE: the ratio of the Šidák to the Bonferroni threshold at each C.
ratio = ???

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
# Left: the two thresholds vs C.
ax1.loglog(C, bonferroni_threshold, color='#4e79a7', label=r'Bonferroni: $\\alpha_{FW}/C$')
ax1.loglog(C, sidak_threshold, '--', color='#e15759', label=r'Šidák: $1-(1-\\alpha_{FW})^{1/C}$')
ax1.axvline(1e6, color='grey', ls=':', label=r'$\\approx$1M independent tests')
ax1.axhline(5e-8, color='black', ls=':', label=r'$5\\times10^{-8}$')
ax1.set_xlabel('number of tests, C'); ax1.set_ylabel('per-test significance threshold')
ax1.set_title('Per-test threshold'); ax1.legend(fontsize=8)
# Right: the Šidák / Bonferroni ratio vs C.
ax2.semilogx(C, ratio, color='#59a14f')
ax2.axhline(1.0, color='grey', ls=':')
ax2.axvline(1e6, color='grey', ls=':')
ax2.set_xlabel('number of tests, C'); ax2.set_ylabel('Šidák / Bonferroni threshold')
ax2.set_title('Ratio of the two thresholds')
plt.tight_layout(); plt.show()

at_1e6 = np.argmin(np.abs(C - 1e6))
print(f"At C=1e6:  Bonferroni={bonferroni_threshold[at_1e6]:.3e}, "
      f"Šidák={sidak_threshold[at_1e6]:.3e}, ratio={ratio[at_1e6]:.4f}")
print("Q: The ratio is largest at small C and tends to ~1 as C grows — why?")
print("Q: Which correction is more conservative, and does the difference ever matter in practice?")
"""

S2_SIDAK_SOL = """\
alpha_familywise = 0.05
C = np.logspace(0, 7, 200)

bonferroni_threshold = alpha_familywise / C
sidak_threshold = 1 - (1 - alpha_familywise)**(1.0 / C)
ratio = sidak_threshold / bonferroni_threshold

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
ax1.loglog(C, bonferroni_threshold, color='#4e79a7', label=r'Bonferroni: $\\alpha_{FW}/C$')
ax1.loglog(C, sidak_threshold, '--', color='#e15759', label=r'Šidák: $1-(1-\\alpha_{FW})^{1/C}$')
ax1.axvline(1e6, color='grey', ls=':', label=r'$\\approx$1M independent tests')
ax1.axhline(5e-8, color='black', ls=':', label=r'$5\\times10^{-8}$')
ax1.set_xlabel('number of tests, C'); ax1.set_ylabel('per-test significance threshold')
ax1.set_title('Per-test threshold'); ax1.legend(fontsize=8)
ax2.semilogx(C, ratio, color='#59a14f')
ax2.axhline(1.0, color='grey', ls=':')
ax2.axvline(1e6, color='grey', ls=':')
ax2.set_xlabel('number of tests, C'); ax2.set_ylabel('Šidák / Bonferroni threshold')
ax2.set_title('Ratio of the two thresholds')
plt.tight_layout(); plt.show()

at_1e6 = np.argmin(np.abs(C - 1e6))
print(f"At C=1e6:  Bonferroni={bonferroni_threshold[at_1e6]:.3e}, "
      f"Šidák={sidak_threshold[at_1e6]:.3e}, ratio={ratio[at_1e6]:.4f}")
# Šidák is always slightly LESS conservative (ratio > 1), most so at small C; as C grows
# 1-(1-a)^(1/C) ≈ a/C, so the ratio tends to 1 and the practical difference vanishes.
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

    _cqn = [0]
    def chmd(src):                      # auto-number a challenge title cell in build order
        _cqn[0] += 1
        return md(src.replace('{N}', str(_cqn[0])))

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
        # Challenge order: encoding → stratification → study design → interpretation
        #                  → capstone
        chmd(S1_CQ1_MD),   *ex(S1_CQ1_STUDENT, S1_CQ1_SOL),
        chmd(S1_CQSEX_MD), *ex(S1_CQSEX_STUDENT, S1_CQSEX_SOL),
        chmd(S1_CQ4_MD),   *ex(S1_CQ4_STUDENT, S1_CQ4_SOL),
        chmd(S1_LZ_MD),    *ex(S1_LZ_STUDENT, S1_LZ_SOL),
        chmd(S1_CQ2_MD),
        code(S1_CQ2_STUDENT) if not run else md(""),
        *ex(S1_CQ2B_STUDENT, S1_CQ2B_SOL),
        *ex(S1_CQ2C_STUDENT, S1_CQ2C_SOL),
        code(S1_CQ2D_STUDENT),                  # Part C is fully provided (book-keeping + plot)
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

    _cqn = [0]
    def chmd(src):
        _cqn[0] += 1
        return md(src.replace('{N}', str(_cqn[0])))

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
        chmd(S2_CQ1_MD), *ex(S2_CQ1_STUDENT, S2_CQ1_SOL),
        chmd(S2_CQ2_MD), *ex(S2_CQ2_STUDENT, S2_CQ2_SOL),
        chmd(S2_CQ3_MD), *ex(S2_CQ3_STUDENT, S2_CQ3_SOL),
        chmd(S2_CQ4_MD), *ex(S2_CQ4_STUDENT, S2_CQ4_SOL),
        chmd(S2_MIAMI_MD), *ex(S2_MIAMI_STUDENT, S2_MIAMI_SOL), *ex(S2_CSQ_STUDENT, S2_CSQ_SOL),
        chmd(S2_SIDAK_MD), *ex(S2_SIDAK_STUDENT, S2_SIDAK_SOL),
    ]
    return notebook(cells)


# ─── Session 3 cells (population structure / PCA / relatedness) ───────────────

S3_TITLE = """\
# Session 3: Complexities of GWAS — Population Structure & Relatedness

**Timing**: ~60 minutes (Parts 1–4 ~45 min; challenges for fast finishers).

**Dataset**: a simulated **"All of Us"-style** diverse cohort (~2,000 participants) plus a small
**"1000 Genomes"-style reference panel** (~75 individuals per continental group, with known labels),
each genotyped at 20,000 independent SNPs.
- `G`        : the diverse cohort we analyse.
- `G_ref`    : the labelled reference panel (superpopulations AFR, AMR, EAS, EUR, SAS).
- `y_strat`  : a cohort phenotype confounded **purely by ancestry** (environmental — no causal variant).
- `y_clean`  : a cohort phenotype with a few **true causal variants** and no confounding.

Everything (PCA, GWAS) is built from scratch with NumPy; the only "black box" we hand you is a
ready-made random-forest classifier for the ancestry-assignment step.
"""

S3_SETUP = """\
# ─────────────────────────────────────────────────────────────────────────────
# SETUP — run once. Downloads the data (on Colab) and defines run_gwas + helpers.
# ─────────────────────────────────────────────────────────────────────────────
import os, numpy as np, pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
import warnings; warnings.filterwarnings('ignore')
plt.rcParams.update({'figure.dpi': 80, 'font.size': 11})

""" + data_bootstrap(['pca_data.npz']) + """
data = np.load(os.path.join(DATA_DIR, 'pca_data.npz'), allow_pickle=True)

# ── The diverse cohort we analyse ("All of Us"-style) ────────────────────────
G              = data['G'].astype(np.float32)   # (N_cohort, M)  genotypes 0/1/2
y_strat        = data['y_strat']                # ancestry-confounded phenotype (no genetics)
y_clean        = data['y_clean']                # phenotype with true causal variants
causal_idx     = data['causal_idx']             # true causal SNP indices for y_clean
rel_pairs      = data['rel_pairs']              # injected sibling pairs (row indices into G)
true_anc       = data['true_anc']               # (N_cohort, 5) TRUE ancestry proportions (answer key)
age            = data['age']
sex            = data['sex']

# ── The labelled reference panel ("1000 Genomes"-style) ──────────────────────
G_ref          = data['G_ref'].astype(np.float32)   # (N_ref, M) genotypes 0/1/2
ref_pop        = data['ref_pop']                    # superpopulation index for each reference person
superpop_names = data['superpop_names']             # ['AFR', 'AMR', 'EAS', 'EUR', 'SAS']

n_cohort = G.shape[0]
n_ref    = G_ref.shape[0]
n_snps   = G.shape[1]
print(f"Cohort:    {n_cohort:,} individuals x {n_snps:,} SNPs")
print(f"Reference: {n_ref:,} individuals, "
      f"{len(superpop_names)} superpopulations ({list(superpop_names)})")

""" + RUN_GWAS_FN + """
def qq_plot(pvals, ax=None, color='steelblue', label=''):
    \"\"\"QQ plot of -log10(p) vs the uniform-null expectation, x-axis truncated.\"\"\"
    if ax is None:
        fig, ax = plt.subplots(figsize=(5, 5))
    n   = len(pvals)
    obs = -np.log10(np.sort(pvals) + 1e-300)
    exp = -np.log10(np.arange(1, n+1) / (n+1))
    ax.scatter(exp, obs, s=4, alpha=0.5, color=color, label=label)
    ax.plot([0, exp.max()], [0, exp.max()], 'r--', lw=1); ax.set_xlim(0, exp.max())
    ax.set_xlabel(r'Expected $-\\log_{10}p$'); ax.set_ylabel(r'Observed $-\\log_{10}p$')
    if label: ax.legend(fontsize=8)
    return ax

def lambda_gc(pvals):
    \"\"\"Genomic-control inflation factor: 1.0 means no inflation.\"\"\"
    chi2 = stats.chi2.isf(np.clip(pvals, 1e-300, 1), df=1)
    return np.median(chi2) / stats.chi2.ppf(0.5, df=1)

print("Setup ready: run_gwas(), qq_plot(), lambda_gc().")
"""

S3_PART1_MD = """\
## Part 1: Seeing population stratification

If allele frequencies differ between ancestral groups **and** the phenotype mean differs between
those groups (for *environmental* reasons), then *every* ancestry-informative variant will look
associated — even with no causal effect at all. This is **confounding by population structure**.

`y_strat` was generated with **no genetic effect** — its mean only differs by ancestry. Run a
GWAS and see what the QQ plot / $\\lambda_{GC}$ look like.
"""

S3_EX1_STUDENT = """\
# Exercise 1.1: GWAS of the ancestry-confounded phenotype, with NO ancestry correction.
#
# run_gwas(phenotype, genotypes, covars=...) regresses the phenotype on each SNP.
# Here we deliberately pass NO covariates, so nothing accounts for ancestry.

# YOUR CODE HERE: run the GWAS of y_strat on the cohort genotypes G (no covariates)
betas, std_errors, pvalues_strat = run_gwas(???, ???)

# Plot the QQ plot and report the genomic-control inflation factor (lambda_GC).
fig, ax = plt.subplots(figsize=(5, 5))
qq_plot(pvalues_strat, ax=ax, label='y_strat, no covariates')
ax.set_title('QQ plot — ancestry-confounded phenotype')
plt.tight_layout(); plt.show()

n_hits = (pvalues_strat < 5e-8).sum()
print(f"lambda_GC = {lambda_gc(pvalues_strat):.2f}   (1.0 would mean no inflation)")
print(f"'Hits' at p < 5e-8: {n_hits:,}  — but NOTHING is truly causal!")
print("Q: Why are there so many associations when no variant affects the trait?")
"""

S3_EX1_SOL = """\
# Run the GWAS with no covariates: every ancestry-informative SNP picks up the ancestry-driven
# difference in phenotype means.
betas, std_errors, pvalues_strat = run_gwas(y_strat, G)

fig, ax = plt.subplots(figsize=(5, 5))
qq_plot(pvalues_strat, ax=ax, label='y_strat, no covariates')
ax.set_title('QQ plot — ancestry-confounded phenotype')
plt.tight_layout(); plt.show()

n_hits = (pvalues_strat < 5e-8).sum()
print(f"lambda_GC = {lambda_gc(pvalues_strat):.2f}")
print(f"'Hits' at p < 5e-8: {n_hits:,}  — all spurious (ancestry confounding).")
"""

S3_PART2_MD = """\
## Part 2: Principal Component Analysis (PCA) from scratch

PCA finds the axes of greatest genetic variation across individuals — which, for structured
samples, are the **ancestry axes**. We compute it directly from the standardised genotype matrix
with a singular value decomposition (SVD); no specialised package needed.

If $Z$ is the (individuals × SNPs) standardised genotype matrix, then $Z = U S V^\\top$ and the
**principal-component scores** are the columns of $U S$. The top few capture ancestry.

We stack the **reference panel on top of the cohort** and run PCA on everyone together, so that
both live in the *same* PC space. We then colour the reference individuals by their known
superpopulation and draw the cohort in grey — exactly the kind of plot you see for All of Us.
"""

S3_EX2_STUDENT = """\
# Exercise 2.1: PCA of the reference panel + cohort, in one shared coordinate system.

# Step 1. Stack reference panel (first) on top of the cohort (after).
genotypes_all = np.vstack([G_ref, G])               # shape (n_ref + n_cohort, n_snps)

# Step 2. Standardise each SNP to mean 0, sd 1.
genotype_mean = genotypes_all.mean(axis=0)
genotype_sd   = genotypes_all.std(axis=0) + 1e-8
Z = (genotypes_all - genotype_mean) / genotype_sd

# Step 3. Economy SVD of Z / sqrt(n_total); the PC scores are the columns of U * S.
n_total = genotypes_all.shape[0]
# YOUR CODE HERE: fill in the matrix to decompose, and the PC-score formula.
U, S, Vt = np.linalg.svd(???, full_matrices=False)
PC = ???                                            # (n_total, k) PC scores = U * S

# Step 4. Split the PC scores back into reference rows (first) and cohort rows (after).
PC_ref    = PC[:n_ref]
PC_cohort = PC[n_ref:]

# Step 5. Plot PC1 vs PC2: cohort in grey, reference coloured by superpopulation.
fig, ax = plt.subplots(figsize=(7, 5.5))
ax.scatter(PC_cohort[:, 0], PC_cohort[:, 1], s=8, alpha=0.4, color='lightgrey', label='cohort')
for group_index, group_name in enumerate(superpop_names):
    is_this_group = ref_pop == group_index
    ax.scatter(PC_ref[is_this_group, 0], PC_ref[is_this_group, 1],
               s=45, edgecolor='black', lw=0.3, label=f'ref: {group_name}')
ax.set_xlabel('PC1'); ax.set_ylabel('PC2')
ax.set_title('Genotype PCA — cohort anchored by a labelled reference panel')
ax.legend(fontsize=8, markerscale=1.2); plt.tight_layout(); plt.show()
print("Q: Which reference cluster does the bulk of the cohort sit near?")
print("Q: Some cohort individuals fall BETWEEN reference clusters — who might they be?")
"""

S3_EX2_SOL = """\
genotypes_all = np.vstack([G_ref, G])
genotype_mean = genotypes_all.mean(axis=0)
genotype_sd   = genotypes_all.std(axis=0) + 1e-8
Z = (genotypes_all - genotype_mean) / genotype_sd

n_total = genotypes_all.shape[0]
U, S, Vt = np.linalg.svd(Z / np.sqrt(n_total), full_matrices=False)
PC = U * S

PC_ref    = PC[:n_ref]
PC_cohort = PC[n_ref:]

fig, ax = plt.subplots(figsize=(7, 5.5))
ax.scatter(PC_cohort[:, 0], PC_cohort[:, 1], s=8, alpha=0.4, color='lightgrey', label='cohort')
for group_index, group_name in enumerate(superpop_names):
    is_this_group = ref_pop == group_index
    ax.scatter(PC_ref[is_this_group, 0], PC_ref[is_this_group, 1],
               s=45, edgecolor='black', lw=0.3, label=f'ref: {group_name}')
ax.set_xlabel('PC1'); ax.set_ylabel('PC2')
ax.set_title('Genotype PCA — cohort anchored by a labelled reference panel')
ax.legend(fontsize=8, markerscale=1.2); plt.tight_layout(); plt.show()
"""

S3_PART3_MD = """\
## Part 3: Assigning continental ancestry with a reference panel

In a biobank we rarely *know* anyone's ancestry — we only have the genotypes. The standard trick is
exactly what we set up in Part 2: project everyone into PC space, then **use the labelled reference
panel to name the clusters**.

A convenient way to turn "near which reference cluster?" into an automatic label is a
**classifier**. We train a **random forest** on the reference individuals (features = their top PCs,
labels = their superpopulation), then ask it to predict a superpopulation for each cohort member.

Crucially, the classifier also reports a **probability** for each prediction. Admixed individuals
sit *between* clusters, so no single superpopulation gets a high probability — we use a
**confidence threshold** to leave those people **unassigned** rather than forcing a wrong label.
"""

S3_EX3_STUDENT = """\
# Exercise 3.1: train a random forest on the reference panel, then label the cohort.
from sklearn.ensemble import RandomForestClassifier

n_pc = 10                                   # number of PCs to use as features

# Training data = the REFERENCE individuals (we know their superpopulation labels).
reference_features = PC_ref[:, :n_pc]       # (n_ref, n_pc)
reference_labels   = ref_pop                # known superpopulation index per reference person

# Build and train the classifier.
classifier = RandomForestClassifier(n_estimators=300, random_state=0)
classifier.fit(reference_features, reference_labels)

# Ask the classifier to label each COHORT individual from their PCs.
cohort_features        = PC_cohort[:, :n_pc]
predicted_probability  = classifier.predict_proba(cohort_features)   # (n_cohort, 5)
best_probability       = predicted_probability.max(axis=1)           # confidence of top guess
predicted_group_index  = predicted_probability.argmax(axis=1)        # which superpopulation

# Only keep a label if the classifier is confident enough; otherwise call it 'Unassigned'.
CONFIDENCE_THRESHOLD = 0.80                  # try changing this and see what happens
# YOUR CODE HERE: boolean array, True where best_probability passes the threshold.
is_confident = ???

assigned_label = np.where(is_confident,
                          superpop_names[predicted_group_index],
                          'Unassigned')

# Report how many cohort members got each label.
labels_unique, label_counts = np.unique(assigned_label, return_counts=True)
print(f"Confidence threshold = {CONFIDENCE_THRESHOLD}")
for label, count in zip(labels_unique, label_counts):
    print(f"  {label:11s}: {count:5d}  ({100*count/n_cohort:4.1f}%)")
print("Q: Which groups get assigned confidently, and which get left 'Unassigned'?")
"""

S3_EX3_SOL = """\
from sklearn.ensemble import RandomForestClassifier

n_pc = 10
reference_features = PC_ref[:, :n_pc]
reference_labels   = ref_pop

classifier = RandomForestClassifier(n_estimators=300, random_state=0)
classifier.fit(reference_features, reference_labels)

cohort_features        = PC_cohort[:, :n_pc]
predicted_probability  = classifier.predict_proba(cohort_features)
best_probability       = predicted_probability.max(axis=1)
predicted_group_index  = predicted_probability.argmax(axis=1)

CONFIDENCE_THRESHOLD = 0.80
is_confident = best_probability >= CONFIDENCE_THRESHOLD

assigned_label = np.where(is_confident,
                          superpop_names[predicted_group_index],
                          'Unassigned')

labels_unique, label_counts = np.unique(assigned_label, return_counts=True)
print(f"Confidence threshold = {CONFIDENCE_THRESHOLD}")
for label, count in zip(labels_unique, label_counts):
    print(f"  {label:11s}: {count:5d}  ({100*count/n_cohort:4.1f}%)")
"""

S3_PART3B_MD = """\
Now **re-draw the PCA, colouring each cohort member by its assigned superpopulation** (and leaving
the unassigned individuals grey). This is the figure a biobank publishes to describe its cohort.
"""

S3_EX3B_STUDENT = """\
# Exercise 3.2: re-plot the cohort coloured by assigned superpopulation.
fig, ax = plt.subplots(figsize=(7, 5.5))

# Unassigned cohort members first, in grey.
unassigned = assigned_label == 'Unassigned'
ax.scatter(PC_cohort[unassigned, 0], PC_cohort[unassigned, 1],
           s=8, alpha=0.4, color='lightgrey', label='Unassigned')

# YOUR CODE HERE: one coloured scatter per superpopulation (assigned cohort members only).
for group_name in superpop_names:
    in_group = assigned_label == group_name
    ax.scatter(PC_cohort[in_group, 0], PC_cohort[in_group, 1],
               s=12, alpha=0.7, label=???)

ax.set_xlabel('PC1'); ax.set_ylabel('PC2')
ax.set_title('Cohort coloured by assigned continental ancestry')
ax.legend(fontsize=8, markerscale=1.5); plt.tight_layout(); plt.show()
print("Q: The 'Unassigned' points cluster between groups — what does that tell you about them?")
"""

S3_EX3B_SOL = """\
fig, ax = plt.subplots(figsize=(7, 5.5))
unassigned = assigned_label == 'Unassigned'
ax.scatter(PC_cohort[unassigned, 0], PC_cohort[unassigned, 1],
           s=8, alpha=0.4, color='lightgrey', label='Unassigned')
for group_name in superpop_names:
    in_group = assigned_label == group_name
    ax.scatter(PC_cohort[in_group, 0], PC_cohort[in_group, 1],
               s=12, alpha=0.7, label=str(group_name))
ax.set_xlabel('PC1'); ax.set_ylabel('PC2')
ax.set_title('Cohort coloured by assigned continental ancestry')
ax.legend(fontsize=8, markerscale=1.5); plt.tight_layout(); plt.show()
"""

S3_PART4_MD = """\
## Part 4: Correcting stratification with PCs

Including the top PCs as **covariates** soaks up the ancestry signal, so the spurious associations
disappear and $\\lambda_{GC}$ returns to ~1. Crucially, **true** signal survives — re-run the
clean phenotype `y_clean` (which has real causal variants) and check its hits are still there.

(We use `PC_cohort` — the cohort's own rows of the PCA — as the covariates.)
"""

S3_EX4_STUDENT = """\
# Exercise 4.1: re-run the GWAS, this time correcting for ancestry with PC covariates.
n_pc = 10
ancestry_covariates = PC_cohort[:, :n_pc]     # the cohort's top PCs

# YOUR CODE HERE: re-run the y_strat GWAS, passing the PCs as covariates.
_, _, pvalues_strat_pc = run_gwas(y_strat, G, covars=???)

# Compare the QQ plots before and after PC correction.
fig, axes = plt.subplots(1, 2, figsize=(11, 5))
qq_plot(pvalues_strat, ax=axes[0])
axes[0].set_title(f'No PCs  (lambda = {lambda_gc(pvalues_strat):.2f})')
qq_plot(pvalues_strat_pc, ax=axes[1], color='#59a14f')
axes[1].set_title(f'With {n_pc} PCs  (lambda = {lambda_gc(pvalues_strat_pc):.2f})')
plt.tight_layout(); plt.show()

# A real signal must SURVIVE the correction: GWAS of y_clean (true causal variants) with PCs.
_, _, pvalues_clean = run_gwas(y_clean, G, covars=ancestry_covariates)
n_recovered = sum(pvalues_clean[c] < 5e-8 for c in causal_idx)
print(f"y_clean hits at p < 5e-8 (with PCs): {(pvalues_clean < 5e-8).sum()}")
print(f"  true causal SNPs recovered: {n_recovered}/{len(causal_idx)}")
print("Q: PCs removed the false signal but kept the real one — why?")
"""

S3_EX4_SOL = """\
n_pc = 10
ancestry_covariates = PC_cohort[:, :n_pc]

_, _, pvalues_strat_pc = run_gwas(y_strat, G, covars=ancestry_covariates)

fig, axes = plt.subplots(1, 2, figsize=(11, 5))
qq_plot(pvalues_strat, ax=axes[0])
axes[0].set_title(f'No PCs  (lambda = {lambda_gc(pvalues_strat):.2f})')
qq_plot(pvalues_strat_pc, ax=axes[1], color='#59a14f')
axes[1].set_title(f'With {n_pc} PCs  (lambda = {lambda_gc(pvalues_strat_pc):.2f})')
plt.tight_layout(); plt.show()

_, _, pvalues_clean = run_gwas(y_clean, G, covars=ancestry_covariates)
n_recovered = sum(pvalues_clean[c] < 5e-8 for c in causal_idx)
print(f"y_clean hits at p < 5e-8 (with PCs): {(pvalues_clean < 5e-8).sum()}")
print(f"  true causal SNPs recovered: {n_recovered}/{len(causal_idx)}")
"""

S3_CQ_MD = """\
---

## Challenge Questions
"""

S3_CQ1_MD = """\
### Challenge {N}: Who gets left "Unassigned"? — ~12 min

The classifier left some cohort members unlabelled. Are these really the **admixed** individuals?
We can check, because the simulation stored each person's **true ancestry proportions** (`true_anc`,
columns AFR, AMR, EAS, EUR, SAS). A person dominated by one ancestry has a *max* proportion near 1;
an admixed person's largest proportion is much smaller. Then explore how the **threshold** trades
off coverage against confidence.
"""

S3_CQ1_STUDENT = """\
# Challenge, Part A: are the 'Unassigned' people the admixed ones?
# 'dominant ancestry fraction' = the largest of a person's 5 true ancestry proportions.
dominant_ancestry_fraction = true_anc.max(axis=1)

fig, ax = plt.subplots(figsize=(7, 4))
ax.hist(dominant_ancestry_fraction[~unassigned], bins=30, alpha=0.6, label='assigned', color='#4e79a7')
ax.hist(dominant_ancestry_fraction[ unassigned], bins=30, alpha=0.6, label='Unassigned', color='grey')
ax.set_xlabel('largest true ancestry proportion (1.0 = single-ancestry)')
ax.set_ylabel('cohort members'); ax.legend()
ax.set_title('Unassigned individuals are the most admixed'); plt.tight_layout(); plt.show()
print(f"Mean dominant ancestry — assigned: {dominant_ancestry_fraction[~unassigned].mean():.2f}, "
      f"unassigned: {dominant_ancestry_fraction[unassigned].mean():.2f}")

# Challenge, Part B: how does the threshold trade coverage against confidence?
# YOUR CODE HERE: for each candidate threshold, what fraction of the cohort is assigned?
thresholds = np.linspace(0.4, 1.0, 25)
fraction_assigned = [ (best_probability >= ???).mean() for t in thresholds ]

fig, ax = plt.subplots(figsize=(7, 4))
ax.plot(thresholds, fraction_assigned, marker='o', ms=4)
ax.set_xlabel('confidence threshold'); ax.set_ylabel('fraction of cohort assigned')
ax.set_title('Stricter threshold → fewer (but more confident) assignments')
plt.tight_layout(); plt.show()
print("Q: A stricter threshold drops more people. Who is dropped first, and is that fair?")
"""

S3_CQ1_SOL = """\
dominant_ancestry_fraction = true_anc.max(axis=1)
fig, ax = plt.subplots(figsize=(7, 4))
ax.hist(dominant_ancestry_fraction[~unassigned], bins=30, alpha=0.6, label='assigned', color='#4e79a7')
ax.hist(dominant_ancestry_fraction[ unassigned], bins=30, alpha=0.6, label='Unassigned', color='grey')
ax.set_xlabel('largest true ancestry proportion (1.0 = single-ancestry)')
ax.set_ylabel('cohort members'); ax.legend()
ax.set_title('Unassigned individuals are the most admixed'); plt.tight_layout(); plt.show()
print(f"Mean dominant ancestry — assigned: {dominant_ancestry_fraction[~unassigned].mean():.2f}, "
      f"unassigned: {dominant_ancestry_fraction[unassigned].mean():.2f}")

thresholds = np.linspace(0.4, 1.0, 25)
fraction_assigned = [ (best_probability >= t).mean() for t in thresholds ]
fig, ax = plt.subplots(figsize=(7, 4))
ax.plot(thresholds, fraction_assigned, marker='o', ms=4)
ax.set_xlabel('confidence threshold'); ax.set_ylabel('fraction of cohort assigned')
ax.set_title('Stricter threshold → fewer (but more confident) assignments')
plt.tight_layout(); plt.show()
# Admixed people are dropped first — a real cost, since ancestry-restricted analyses then
# systematically exclude them.
"""

S3_CQ2_MD = """\
### Challenge {N}: Relatedness and the GRM — ~12 min

Close relatives share long genomic segments, which (like ancestry) violates the independence GWAS
assumes. The **genetic relatedness matrix** $\\text{GRM} = \\frac{1}{M} Z Z^\\top$ quantifies it.
A few **sibling pairs** were hidden in the cohort — find them, and think about why mixed models exist.
"""

S3_CQ2_STUDENT = """\
# Challenge: build the genetic relatedness matrix (GRM) of the cohort and find the relatives.
# Standardise the cohort genotypes (mean 0, sd 1 per SNP).
Z_cohort = (G - G.mean(axis=0)) / (G.std(axis=0) + 1e-8)

# YOUR CODE HERE: GRM = Z Zᵀ / (number of SNPs).
#   diagonal ≈ 1   (each person with themselves)
#   off-diagonal ≈ 2 × kinship  (≈ 0.5 for full siblings/parent-child, ≈ 0 for strangers)
GRM = ???

# Histogram of all off-diagonal (between-person) relatedness values.
off_diagonal = GRM[np.triu_indices(n_cohort, k=1)]
fig, ax = plt.subplots(figsize=(7, 3))
ax.hist(off_diagonal, bins=120, color='steelblue'); ax.set_yscale('log')
ax.axvline(0.5, color='red', ls='--', label='sib / parent-child ≈ 0.5')
ax.set_xlabel('GRM off-diagonal (relatedness)'); ax.set_ylabel('pairs (log scale)'); ax.legend()
ax.set_title('Most pairs ≈ 0; a few related pairs stand out'); plt.tight_layout(); plt.show()

print("Injected sibling pairs (relatedness should be ≈ 0.5):")
for person_i, person_j in rel_pairs:
    print(f"  rows {person_i},{person_j}: GRM = {GRM[person_i, person_j]:.2f}")
print("Q: Why does ignoring relatedness inflate test statistics, and how do mixed models help?")
"""

S3_CQ2_SOL = """\
Z_cohort = (G - G.mean(axis=0)) / (G.std(axis=0) + 1e-8)
GRM = Z_cohort @ Z_cohort.T / n_snps

off_diagonal = GRM[np.triu_indices(n_cohort, k=1)]
fig, ax = plt.subplots(figsize=(7, 3))
ax.hist(off_diagonal, bins=120, color='steelblue'); ax.set_yscale('log')
ax.axvline(0.5, color='red', ls='--', label='sib / parent-child ≈ 0.5')
ax.set_xlabel('GRM off-diagonal (relatedness)'); ax.set_ylabel('pairs (log scale)'); ax.legend()
ax.set_title('Most pairs ≈ 0; a few related pairs stand out'); plt.tight_layout(); plt.show()
print("Injected sibling pairs (relatedness should be ≈ 0.5):")
for person_i, person_j in rel_pairs:
    print(f"  rows {person_i},{person_j}: GRM = {GRM[person_i, person_j]:.2f}")
# Relatives share segments → correlated residuals → inflated statistics. A linear mixed model
# puts the GRM in the covariance of a random effect, accounting for this structure.
"""


def build_session3(answers=False, run=False, nb_path=None):
    def ex(student_src, sol_src):
        if run:
            return [code(sol_src)]
        cells = [code(student_src)]
        if answers:
            cells.append(solution(sol_src))
        return cells

    _cqn = [0]
    def chmd(src):
        _cqn[0] += 1
        return md(src.replace('{N}', str(_cqn[0])))

    title = (colab_badge(nb_path) + "\n\n" + S3_TITLE) if nb_path else S3_TITLE
    cells = [
        md(title),
        code(S3_SETUP),
        md(S3_PART1_MD),
        *ex(S3_EX1_STUDENT, S3_EX1_SOL),
        md(S3_PART2_MD),
        *ex(S3_EX2_STUDENT, S3_EX2_SOL),
        md(S3_PART3_MD),
        *ex(S3_EX3_STUDENT, S3_EX3_SOL),
        md(S3_PART3B_MD),
        *ex(S3_EX3B_STUDENT, S3_EX3B_SOL),
        md(S3_PART4_MD),
        *ex(S3_EX4_STUDENT, S3_EX4_SOL),
        md(S3_CQ_MD),
        chmd(S3_CQ1_MD), *ex(S3_CQ1_STUDENT, S3_CQ1_SOL),
        chmd(S3_CQ2_MD), *ex(S3_CQ2_STUDENT, S3_CQ2_SOL),
    ]
    return notebook(cells)


# ─── Session 4 cells (statistical fine-mapping) ───────────────────────────────

S4_TITLE = """\
# Session 4: Fine-mapping

**Timing**: ~60 minutes (Parts 1–3 ~35 min; challenges for fast finishers).

**Dataset**: two simulated GWAS **loci** (~400 variants each, N=5,000) with realistic LD.
- Locus A has **one** causal variant hidden in a cluster of near-perfectly-correlated SNPs.
- Locus B has **two** causal variants.

The goal of fine-mapping: go from "dozens of significant variants in a peak" to a short list of
**credible** causal candidates. Everything is built from regression + a one-line Bayes factor.
"""

S4_SETUP = """\
# ─────────────────────────────────────────────────────────────────────────────
# SETUP — run once. Downloads the data (on Colab) and defines run_gwas + helpers.
# ─────────────────────────────────────────────────────────────────────────────
import os, numpy as np, pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
import warnings; warnings.filterwarnings('ignore')
plt.rcParams.update({'figure.dpi': 80, 'font.size': 11})

""" + data_bootstrap(['finemap_data.npz']) + """
d = np.load(os.path.join(DATA_DIR, 'finemap_data.npz'), allow_pickle=True)
GA = d['GA'].astype(np.float32); yA = d['yA']; posA = d['posA']; causalA = int(d['causalA'][0])
GB = d['GB'].astype(np.float32); yB = d['yB']; posB = d['posB']; causalB = d['causalB']
print(f"Locus A: {GA.shape[1]} variants, N={GA.shape[0]:,}, 1 causal")
print(f"Locus B: {GB.shape[1]} variants, N={GB.shape[0]:,}, 2 causal")

""" + RUN_GWAS_FN + """
def r2_with(G, j):
    \"\"\"r² between variant j and every variant in the locus.\"\"\"
    g = G[:, j]
    return np.array([np.corrcoef(g, G[:, k])[0, 1]**2 for k in range(G.shape[1])])

print("Setup ready: run_gwas(), r2_with().")
"""

S4_PART1_MD = """\
## Part 1: Why a single GWAS peak isn't enough

Inside an associated locus, **many** variants are genome-wide significant — not because they are
all causal, but because they are in **LD** with the causal one. A LocusZoom-style plot (coloured
by $r^2$ with the lead SNP) shows the tell-tale triangular peak. The marginal **lead** SNP is often
just a *tag*, not the causal variant.
"""

S4_EX1_STUDENT = """\
# Exercise 1.1: regional GWAS + LocusZoom of Locus A.
# YOUR CODE HERE: run the GWAS of phenotype yA on the locus genotypes GA.
betas, std_errors, pvalues = run_gwas(???, ???)

lead_variant = int(np.argmin(pvalues))          # most significant ("lead") variant
r2_to_lead   = r2_with(GA, lead_variant)         # LD (r²) of every variant with the lead

print(f"Genome-wide significant variants: {(pvalues < 5e-8).sum()} of {len(pvalues)}")
print(f"Marginal lead = variant {lead_variant} (pos {posA[lead_variant]} kb);  "
      f"true causal = variant {causalA}")

fig, ax = plt.subplots(figsize=(11, 4))
sc = ax.scatter(posA, -np.log10(pvalues + 1e-300), c=r2_to_lead, cmap='RdYlBu_r',
                vmin=0, vmax=1, s=18)
ax.scatter(posA[causalA], -np.log10(pvalues[causalA] + 1e-300), marker='*', s=320,
           color='lime', edgecolor='black', zorder=5, label='true causal')
plt.colorbar(sc, ax=ax, label='$r^2$ with lead'); ax.axhline(-np.log10(5e-8), color='red', ls='--')
ax.set_xlabel('Position (kb)'); ax.set_ylabel(r'$-\\log_{10}p$')
ax.set_title('Locus A — many significant variants in one LD peak'); ax.legend()
plt.tight_layout(); plt.show()
print("Q: How many variants would you have to follow up if you took every significant one?")
"""

S4_EX1_SOL = """\
betas, std_errors, pvalues = run_gwas(yA, GA)

lead_variant = int(np.argmin(pvalues))
r2_to_lead   = r2_with(GA, lead_variant)

print(f"Genome-wide significant variants: {(pvalues < 5e-8).sum()} of {len(pvalues)}")
print(f"Marginal lead = variant {lead_variant} (pos {posA[lead_variant]} kb);  "
      f"true causal = variant {causalA}")

fig, ax = plt.subplots(figsize=(11, 4))
sc = ax.scatter(posA, -np.log10(pvalues + 1e-300), c=r2_to_lead, cmap='RdYlBu_r',
                vmin=0, vmax=1, s=18)
ax.scatter(posA[causalA], -np.log10(pvalues[causalA] + 1e-300), marker='*', s=320,
           color='lime', edgecolor='black', zorder=5, label='true causal')
plt.colorbar(sc, ax=ax, label='$r^2$ with lead'); ax.axhline(-np.log10(5e-8), color='red', ls='--')
ax.set_xlabel('Position (kb)'); ax.set_ylabel(r'$-\\log_{10}p$')
ax.set_title('Locus A — many significant variants in one LD peak'); ax.legend()
plt.tight_layout(); plt.show()
"""

S4_PART2_MD = """\
## Part 2: "Poor man's" fine-mapping — conditional analysis

A simple way to count **independent** signals: take the lead variant, add it as a **covariate**,
and re-run the GWAS. If a signal remains, there's a second independent association; repeat until
nothing is significant. (This isn't a credible set, but it builds the right intuition.)

We'll **re-draw the LocusZoom after every round**, so you can watch the peak collapse once its
causal variant is conditioned out.
"""

S4_EX2_STUDENT = """\
# Exercise 2.1: iterative conditional analysis on Locus A, with a LocusZoom after each round.

# A small helper that draws one LocusZoom-style plot (−log10 p, coloured by r² with the round's lead).
def plot_locuszoom(pvalues, lead_variant, title):
    r2_to_lead = r2_with(GA, lead_variant)
    fig, ax = plt.subplots(figsize=(11, 3.2))
    sc = ax.scatter(posA, -np.log10(pvalues + 1e-300), c=r2_to_lead,
                    cmap='RdYlBu_r', vmin=0, vmax=1, s=16)
    ax.scatter(posA[causalA], -np.log10(pvalues[causalA] + 1e-300), marker='*', s=280,
               color='lime', edgecolor='black', zorder=5, label='true causal')
    ax.axvline(posA[lead_variant], color='grey', ls=':', label=f'round lead (variant {lead_variant})')
    ax.axhline(-np.log10(5e-8), color='red', ls='--')
    plt.colorbar(sc, ax=ax, label='r² with round lead')
    ax.set_xlabel('Position (kb)'); ax.set_ylabel(r'$-\\log_{10}p$')
    ax.set_title(title); ax.legend(fontsize=8); plt.tight_layout(); plt.show()

significance_threshold = 5e-8
conditioned_on = []                         # variants we have added as covariates so far

# Round 0: the ordinary GWAS, with nothing conditioned out yet.
_, _, pvalues = run_gwas(yA, GA)
round_number = 0

while round_number < 6:                     # safety cap on the number of rounds
    lead_variant = int(np.argmin(pvalues))
    conditioning_text = conditioned_on if conditioned_on else 'nothing'
    plot_locuszoom(pvalues, lead_variant,
                   f'Round {round_number}: conditioning on {conditioning_text}  '
                   f'(lead variant {lead_variant}, p = {pvalues[lead_variant]:.1e})')

    # Stop once the strongest remaining signal is no longer genome-wide significant.
    if pvalues[lead_variant] >= significance_threshold:
        print("No genome-wide-significant signal remains — stop.")
        break

    # Otherwise, record this lead and condition on ALL signals found so far.
    conditioned_on.append(lead_variant)
    signal_genotypes = GA[:, conditioned_on]    # genotype columns of every signal so far
    # YOUR CODE HERE: re-run the GWAS, passing the signal genotypes as covariates.
    _, _, pvalues = run_gwas(yA, GA, covars=???)
    round_number += 1

print(f"Independent signals found: {len(conditioned_on)}  → variants {conditioned_on}")
print("(Locus A has 1 causal variant — do you recover a single independent signal?)")
print("Q: After conditioning on the lead, why do all the other peak variants drop out?")
"""

S4_EX2_SOL = """\
def plot_locuszoom(pvalues, lead_variant, title):
    r2_to_lead = r2_with(GA, lead_variant)
    fig, ax = plt.subplots(figsize=(11, 3.2))
    sc = ax.scatter(posA, -np.log10(pvalues + 1e-300), c=r2_to_lead,
                    cmap='RdYlBu_r', vmin=0, vmax=1, s=16)
    ax.scatter(posA[causalA], -np.log10(pvalues[causalA] + 1e-300), marker='*', s=280,
               color='lime', edgecolor='black', zorder=5, label='true causal')
    ax.axvline(posA[lead_variant], color='grey', ls=':', label=f'round lead (variant {lead_variant})')
    ax.axhline(-np.log10(5e-8), color='red', ls='--')
    plt.colorbar(sc, ax=ax, label='r² with round lead')
    ax.set_xlabel('Position (kb)'); ax.set_ylabel(r'$-\\log_{10}p$')
    ax.set_title(title); ax.legend(fontsize=8); plt.tight_layout(); plt.show()

significance_threshold = 5e-8
conditioned_on = []
_, _, pvalues = run_gwas(yA, GA)
round_number = 0
while round_number < 6:
    lead_variant = int(np.argmin(pvalues))
    conditioning_text = conditioned_on if conditioned_on else 'nothing'
    plot_locuszoom(pvalues, lead_variant,
                   f'Round {round_number}: conditioning on {conditioning_text}  '
                   f'(lead variant {lead_variant}, p = {pvalues[lead_variant]:.1e})')
    if pvalues[lead_variant] >= significance_threshold:
        print("No genome-wide-significant signal remains — stop.")
        break
    conditioned_on.append(lead_variant)
    signal_genotypes = GA[:, conditioned_on]
    _, _, pvalues = run_gwas(yA, GA, covars=signal_genotypes)
    round_number += 1
print(f"Independent signals found: {len(conditioned_on)}  → variants {conditioned_on}")
# Conditioning on the lead absorbs the shared LD signal, so its tag SNPs are no longer significant.
"""

S4_PART3_MD = """\
## Part 3: Posterior probabilities and the credible set

Bayesian fine-mapping assigns each variant a **posterior inclusion probability (PIP)** — the
probability it is the causal one. Assuming a single causal variant in the locus, Wakefield's
**approximate Bayes factor (ABF)** for variant $j$ uses only its $z = \\hat\\beta/\\text{se}$:

$$\\log\\text{ABF}_j = \\tfrac{1}{2}\\log(1-r) + \\tfrac{1}{2} z_j^2\\, r, \\qquad r = \\frac{W}{V_j + W}$$

where $V_j=\\text{se}_j^2$ and $W$ is the prior variance of the effect. Normalising the ABFs gives
PIPs; the **95% credible set** is the smallest set of variants whose PIPs sum to ≥ 0.95.
"""

S4_EX3_STUDENT = """\
# Exercise 3.1: PIP and the 95% credible set for Locus A.
betas, std_errors, pvalues = run_gwas(yA, GA)
z = betas / std_errors

# Wakefield approximate Bayes factor, one variant at a time.
prior_variance    = 0.04                     # W: prior variance of the true effect size
sampling_variance = std_errors**2            # V: squared standard error of each estimate
r = prior_variance / (sampling_variance + prior_variance)

# YOUR CODE HERE: the log approximate Bayes factor = 0.5*log(1 - r) + 0.5 * z^2 * r
log_abf = ???
abf     = np.exp(log_abf - log_abf.max())    # exponentiate (subtract max for numerical stability)
# YOUR CODE HERE: posterior inclusion probabilities = abf normalised to sum to 1
pip     = ???

# 95% credible set: the smallest set of top-PIP variants whose PIPs sum to >= 0.95.
order          = np.argsort(pip)[::-1]                 # variants from highest to lowest PIP
cumulative_pip = np.cumsum(pip[order])
credible_set   = order[:np.searchsorted(cumulative_pip, 0.95) + 1]

print(f"95% credible set: {sorted(credible_set.tolist())}  ({len(credible_set)} variants)")
print(f"Contains the true causal ({causalA})? {causalA in credible_set.tolist()}")
print(f"PIP of the true causal: {pip[causalA]:.2f}   (max PIP in locus: {pip.max():.2f})")
print("Q: Fine-mapping cut the significant peak down to how few variants?")
"""

S4_EX3_SOL = """\
betas, std_errors, pvalues = run_gwas(yA, GA)
z = betas / std_errors

prior_variance    = 0.04
sampling_variance = std_errors**2
r = prior_variance / (sampling_variance + prior_variance)

log_abf = 0.5*np.log(1 - r) + 0.5*z**2*r
abf     = np.exp(log_abf - log_abf.max())
pip     = abf / abf.sum()

order          = np.argsort(pip)[::-1]
cumulative_pip = np.cumsum(pip[order])
credible_set   = order[:np.searchsorted(cumulative_pip, 0.95) + 1]

print(f"95% credible set: {sorted(credible_set.tolist())}  ({len(credible_set)} variants)")
print(f"Contains the true causal ({causalA})? {causalA in credible_set.tolist()}")
print(f"PIP of the true causal: {pip[causalA]:.2f}   (max PIP in locus: {pip.max():.2f})")
"""

S4_CQ_MD = """\
---

## Challenge Questions
"""

S4_CQ1_MD = """\
### Challenge {N}: A PIP LocusZoom — ~10 min

Re-draw the locus with each variant's height = PIP (instead of −log10 p). The credible set should
stand out as the handful of high-PIP variants. Mark the true causal.
"""

S4_CQ1_STUDENT = """\
# Challenge: PIP LocusZoom for Locus A  (reuse pip, credible_set from Part 3).
is_in_credible_set = np.zeros(len(pip), bool)
is_in_credible_set[credible_set] = True

fig, ax = plt.subplots(figsize=(11, 4))
ax.scatter(posA[~is_in_credible_set], pip[~is_in_credible_set], s=15, color='lightgrey',
           label='not in credible set')
ax.scatter(posA[is_in_credible_set], pip[is_in_credible_set], s=40, color='#e15759', zorder=4,
           label='95% credible set')
ax.scatter(posA[causalA], pip[causalA], marker='*', s=320, color='lime',
           edgecolor='black', zorder=5, label='true causal')
ax.set_xlabel('Position (kb)'); ax.set_ylabel('PIP'); ax.set_title('Fine-mapping: PIP by position')
ax.legend(); plt.tight_layout(); plt.show()
print(f"Sum of PIP over the credible set: {pip[credible_set].sum():.2f}")
print("Q: The whole significant peak collapses to a few high-PIP variants — why those?")
"""

S4_CQ1_SOL = """\
is_in_credible_set = np.zeros(len(pip), bool)
is_in_credible_set[credible_set] = True

fig, ax = plt.subplots(figsize=(11, 4))
ax.scatter(posA[~is_in_credible_set], pip[~is_in_credible_set], s=15, color='lightgrey',
           label='not in credible set')
ax.scatter(posA[is_in_credible_set], pip[is_in_credible_set], s=40, color='#e15759', zorder=4,
           label='95% credible set')
ax.scatter(posA[causalA], pip[causalA], marker='*', s=320, color='lime',
           edgecolor='black', zorder=5, label='true causal')
ax.set_xlabel('Position (kb)'); ax.set_ylabel('PIP'); ax.set_title('Fine-mapping: PIP by position')
ax.legend(); plt.tight_layout(); plt.show()
print(f"Sum of PIP over the credible set: {pip[credible_set].sum():.2f}")
"""

S4_CQ2_MD = """\
### Challenge {N}: Resolution depends on power — ~10 min

Fine-mapping resolution improves with sample size. Recompute the credible set on random subsets of
the individuals (N/2, N/4) and watch it **grow** as power falls.
"""

S4_CQ2_STUDENT = """\
# Challenge: credible-set size vs sample size.
def compute_credible_set(phenotype, genotypes, prior_variance=0.04):
    \"\"\"Return the 95% credible set (array of variant indices) for a locus.\"\"\"
    betas, std_errors, _ = run_gwas(phenotype, genotypes)
    z = betas / std_errors
    sampling_variance = std_errors**2
    r = prior_variance / (sampling_variance + prior_variance)
    log_abf = 0.5*np.log(1 - r) + 0.5*z**2*r
    pip = np.exp(log_abf - log_abf.max()); pip /= pip.sum()
    order = np.argsort(pip)[::-1]
    cumulative_pip = np.cumsum(pip[order])
    return order[:np.searchsorted(cumulative_pip, 0.95) + 1]

rng = np.random.default_rng(0)
for fraction in [1.0, 0.5, 0.25]:
    n = int(fraction * len(yA))
    subsample = rng.choice(len(yA), n, replace=False)
    # YOUR CODE HERE: compute the credible set on this subsample of individuals.
    cs = compute_credible_set(???, ???)
    print(f"N={n:5d}: credible-set size = {len(cs):3d}, contains causal = {causalA in cs.tolist()}")
print("Q: Why does a smaller sample give a larger (less useful) credible set?")
"""

S4_CQ2_SOL = """\
def compute_credible_set(phenotype, genotypes, prior_variance=0.04):
    betas, std_errors, _ = run_gwas(phenotype, genotypes)
    z = betas / std_errors
    sampling_variance = std_errors**2
    r = prior_variance / (sampling_variance + prior_variance)
    log_abf = 0.5*np.log(1 - r) + 0.5*z**2*r
    pip = np.exp(log_abf - log_abf.max()); pip /= pip.sum()
    order = np.argsort(pip)[::-1]
    cumulative_pip = np.cumsum(pip[order])
    return order[:np.searchsorted(cumulative_pip, 0.95) + 1]

rng = np.random.default_rng(0)
for fraction in [1.0, 0.5, 0.25]:
    n = int(fraction * len(yA))
    subsample = rng.choice(len(yA), n, replace=False)
    cs = compute_credible_set(yA[subsample], GA[subsample])
    print(f"N={n:5d}: credible-set size = {len(cs):3d}, contains causal = {causalA in cs.tolist()}")
"""

S4_CQ3_MD = """\
### Challenge {N}: When one causal variant isn't enough — ~12 min

The single-causal credible set assumes exactly **one** causal variant per locus. **Locus B** has
two. Build the single-causal credible set for Locus B and check whether it captures **both**
causals — then use conditional analysis to recover the second signal (the idea behind SuSiE).
"""

S4_CQ3_STUDENT = """\
# Challenge: two causal variants (Locus B).
# Step 1. Build the single-causal credible set, exactly as in Part 3.
betas, std_errors, pvalues = run_gwas(yB, GB)
z = betas / std_errors
prior_variance = 0.04
sampling_variance = std_errors**2
r = prior_variance / (sampling_variance + prior_variance)
log_abf = 0.5*np.log(1 - r) + 0.5*z**2*r
abf = np.exp(log_abf - log_abf.max())
pip = abf / abf.sum()

order = np.argsort(pip)[::-1]
cumulative_pip = np.cumsum(pip[order])
credible_set = order[:np.searchsorted(cumulative_pip, 0.95) + 1]

print(f"True causals: {causalB.tolist()}")
print(f"Single-causal 95% credible set: {sorted(credible_set.tolist())}")
print(f"  captures BOTH causals? {all(c in credible_set.tolist() for c in causalB)}")

# Step 2. Condition on the lead variant, then refit — does the SECOND causal now stand out?
lead_variant   = int(np.argmin(pvalues))
lead_genotype  = GB[:, [lead_variant]]          # genotype of the lead, as a single covariate column
# YOUR CODE HERE: re-run the GWAS, passing the lead genotype as a covariate.
_, _, pvalues_conditional = run_gwas(yB, GB, covars=???)

new_lead = int(np.argmin(pvalues_conditional))
print(f"After conditioning on variant {lead_variant}: new lead = {new_lead} "
      f"(near the other causal {causalB.tolist()})")
print("Q: Why does a single-causal credible set fail when there are two signals?")
"""

S4_CQ3_SOL = """\
betas, std_errors, pvalues = run_gwas(yB, GB)
z = betas / std_errors
prior_variance = 0.04
sampling_variance = std_errors**2
r = prior_variance / (sampling_variance + prior_variance)
log_abf = 0.5*np.log(1 - r) + 0.5*z**2*r
abf = np.exp(log_abf - log_abf.max())
pip = abf / abf.sum()

order = np.argsort(pip)[::-1]
cumulative_pip = np.cumsum(pip[order])
credible_set = order[:np.searchsorted(cumulative_pip, 0.95) + 1]

print(f"True causals: {causalB.tolist()}")
print(f"Single-causal 95% credible set: {sorted(credible_set.tolist())}")
print(f"  captures BOTH causals? {all(c in credible_set.tolist() for c in causalB)}")

lead_variant   = int(np.argmin(pvalues))
lead_genotype  = GB[:, [lead_variant]]
_, _, pvalues_conditional = run_gwas(yB, GB, covars=lead_genotype)

new_lead = int(np.argmin(pvalues_conditional))
print(f"After conditioning on variant {lead_variant}: new lead = {new_lead} "
      f"(near the other causal {causalB.tolist()})")
# A single-causal model puts all PIP near one signal; SuSiE fits several single-effect components
# iteratively (much like conditioning) to give one credible set per causal variant.
"""


def build_session4(answers=False, run=False, nb_path=None):
    def ex(student_src, sol_src):
        if run:
            return [code(sol_src)]
        cells = [code(student_src)]
        if answers:
            cells.append(solution(sol_src))
        return cells

    _cqn = [0]
    def chmd(src):
        _cqn[0] += 1
        return md(src.replace('{N}', str(_cqn[0])))

    title = (colab_badge(nb_path) + "\n\n" + S4_TITLE) if nb_path else S4_TITLE
    cells = [
        md(title),
        code(S4_SETUP),
        md(S4_PART1_MD),
        *ex(S4_EX1_STUDENT, S4_EX1_SOL),
        md(S4_PART2_MD),
        *ex(S4_EX2_STUDENT, S4_EX2_SOL),
        md(S4_PART3_MD),
        *ex(S4_EX3_STUDENT, S4_EX3_SOL),
        md(S4_CQ_MD),
        chmd(S4_CQ1_MD), *ex(S4_CQ1_STUDENT, S4_CQ1_SOL),
        chmd(S4_CQ2_MD), *ex(S4_CQ2_STUDENT, S4_CQ2_SOL),
        chmd(S4_CQ3_MD), *ex(S4_CQ3_STUDENT, S4_CQ3_SOL),
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
    save(build_session3(answers=False, nb_path="session3/practical.ipynb"), os.path.join(BASE, "session3", "practical.ipynb"))
    save(build_session3(answers=True,  nb_path="session3/answers.ipynb"),   os.path.join(BASE, "session3", "answers.ipynb"))
    save(build_session3(run=True,      nb_path="session3/run.ipynb"),       os.path.join(BASE, "session3", "run.ipynb"))
    save(build_session4(answers=False, nb_path="session4/practical.ipynb"), os.path.join(BASE, "session4", "practical.ipynb"))
    save(build_session4(answers=True,  nb_path="session4/answers.ipynb"),   os.path.join(BASE, "session4", "answers.ipynb"))
    save(build_session4(run=True,      nb_path="session4/run.ipynb"),       os.path.join(BASE, "session4", "run.ipynb"))
    print("Done. 12 notebooks written.")
