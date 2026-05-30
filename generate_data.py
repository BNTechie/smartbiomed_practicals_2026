#!/usr/bin/env python3
"""
Instructor data-generation script for SMARTbiomed GWAS practicals.

Produces:
  data/gwas_data.npz  — GWAS dataset (genotypes, phenotypes, variant info)
  data/fly_data.csv   — Drosophila cross dataset

APPROACH
--------
Session 1 uses fully simulated data, giving instructors complete control:
  1. Generate 10,000 diploid genomes with block-LD structure (chr1, 0–30 Mb).
  2. Inject QC-failing variants (HWE violations, high missingness) and a
     realistic tail of low-level missingness.
  3. Assign spike-and-slab causal effects and simulate phenotypes (h²=0.25).
  4. Save outputs.

No downloads required. Runtime: < 2 minutes.

DEPENDENCIES
  pip install scipy numpy pandas
"""

import numpy as np
import pandas as pd
from scipy import stats
from pathlib import Path

SEED     = 2026
CHROM    = 1        # chromosome label (any non-weird autosome; chr6 excluded due to HLA)
N        = 10_000   # synthetic individuals
M_RAW    = 52_500   # variants before QC (~50k post-QC; keeps gwas_data.npz < 100 MB)
START_KBP =  1_000  # region start (kbp)
END_KBP   = 250_000 # region end   (kbp)  — chr1 is ~248 Mb

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(exist_ok=True)


# ─── Genotype generation ──────────────────────────────────────────────────────

def make_genotypes(n_samples=N, n_vars=M_RAW, chrom=CHROM,
                   start_kbp=START_KBP, end_kbp=END_KBP,
                   block_size=50, ld_alpha=0.80, seed=SEED):
    """
    Generate a synthetic diploid genotype matrix with block-LD structure.

    Within each block of `block_size` variants, haplotypes share a common
    ancestor allele with probability `ld_alpha`, producing pairwise r² ≈ ld_alpha².

    Returns
    -------
    bim : DataFrame  variant info (chrom, snp, cm, pos, a1, a2); pos in bp
    G   : (n_samples, n_vars) int8  diploid dosage (0/1/2)
    """
    rng = np.random.default_rng(seed)

    # Evenly spaced positions across the region
    spacing_kbp = (end_kbp - start_kbp) / n_vars
    pos_kbp = start_kbp + np.arange(n_vars) * spacing_kbp
    pos_bp  = (pos_kbp * 1000).astype(int)

    # EUR-like MAF distribution: right-skewed toward lower MAF, min 5% for block-LD variants
    maf = rng.beta(0.5, 2.0, n_vars) * 0.45 + 0.05

    # Generate 2*n_samples haplotypes with block-LD
    n_haps = 2 * n_samples
    H = np.zeros((n_haps, n_vars), dtype=np.int8)

    for b in range((n_vars - 1) // block_size + 1):
        s = b * block_size
        e = min(s + block_size, n_vars)
        avg_maf = maf[s:e].mean()
        anc = (rng.random(n_haps) < avg_maf).astype(np.int8)   # shared block ancestor
        mix  = rng.random((n_haps, e - s)) < ld_alpha
        indp = (rng.random((n_haps, e - s)) < maf[None, s:e]).astype(np.int8)
        H[:, s:e] = np.where(mix, anc[:, None], indp)

    # Combine haplotype pairs into diploid dosages
    G = (H[:n_samples] + H[n_samples:]).astype(np.int8)

    rsids = np.array([f"rs{1_000_000 + i}" for i in range(n_vars)])
    bim = pd.DataFrame({
        "chrom": chrom,
        "snp":   rsids,
        "cm":    0.0,
        "pos":   pos_bp,
        "a1":    "A",
        "a2":    "G",
    })
    print(f"  Generated: {n_samples:,} samples × {n_vars:,} variants  "
          f"(chr{chrom}:{start_kbp:,}–{end_kbp:,} kbp, "
          f"block-LD α={ld_alpha}, block_size={block_size})")
    return bim, G


# ─── QC failure injection ─────────────────────────────────────────────────────

def inject_qc_failures(G, maf, seed=SEED, n_hwe_bad=500, n_miss_bad=300, n_low_miss=2500,
                       n_rare=3_000):
    """
    Inject realistic QC-failing variants into the genotype matrix.

    Tiers of injected failures:
      - n_rare    variants replaced with rare-allele binomial draws (MAF 0.1–1%,
                  fail the >1% MAF filter); no LD, mimics low-frequency array SNPs
      - n_hwe_bad variants get excess homozygosity (fail HWE)
      - n_miss_bad variants get 6–20% missingness (fail >5% threshold)
      - n_low_miss variants get 0.5–4.5% missingness (pass, fills histogram tail)

    Returns G_raw : (N, M) float array with NaN for missing genotypes.
    """
    rng = np.random.default_rng(seed)
    N, M = G.shape
    G_raw = G.astype(float)

    # Tier 1: Replace columns with rare variants (MAF 0.1–1%) — fail MAF QC
    rare_cols = rng.choice(M, min(n_rare, M), replace=False)
    for j in rare_cols:
        rare_af = rng.uniform(0.001, 0.01)
        G_raw[:, j] = rng.binomial(2, rare_af, N).astype(float)

    # Operate HWE and missingness only on non-rare common variants (MAF > 10%)
    non_rare = np.setdiff1d(np.arange(M), rare_cols)
    common_idx = non_rare[maf[non_rare] > 0.1]

    # Tier 2: HWE violation — swap a fraction of heterozygotes to homozygotes
    hwe_bad = rng.choice(common_idx, min(n_hwe_bad, len(common_idx)), replace=False)
    for j in hwe_bad:
        het_mask = G_raw[:, j] == 1
        if het_mask.sum() > 10:
            to_swap = rng.choice(np.where(het_mask)[0],
                                  int(0.6 * het_mask.sum()), replace=False)
            G_raw[to_swap, j] = rng.choice([0.0, 2.0], len(to_swap))

    # Tier 3: High missingness — fail QC (>5% threshold)
    remaining = np.setdiff1d(non_rare, hwe_bad)
    miss_bad = rng.choice(remaining, min(n_miss_bad, len(remaining)), replace=False)
    for j in miss_bad:
        n_miss = rng.integers(int(0.06 * N), int(0.20 * N))   # 6–20%
        G_raw[rng.choice(N, n_miss, replace=False), j] = np.nan

    # Tier 4: Low-level missingness — pass QC, give the distribution a realistic tail
    remaining2 = np.setdiff1d(remaining, miss_bad)
    low_miss_vars = rng.choice(remaining2, min(n_low_miss, len(remaining2)), replace=False)
    for j in low_miss_vars:
        n_miss = rng.integers(int(0.005 * N), int(0.045 * N))  # 0.5–4.5%
        G_raw[rng.choice(N, n_miss, replace=False), j] = np.nan

    print(f"  Injected rare variants:    {len(rare_cols):,} (MAF 0.1–1%, fail MAF QC)")
    print(f"  Injected HWE failures:     {len(hwe_bad):,} variants")
    print(f"  Injected high missingness: {len(miss_bad):,} variants (6–20%)")
    print(f"  Injected low missingness:  {len(low_miss_vars):,} variants (0.5–4.5%)")
    return G_raw


# ─── Effect sizes ─────────────────────────────────────────────────────────────

def make_spike_slab_betas(maf, p_causal=0.0002, seed=SEED, arch_alpha=0.75):
    """
    Spike-and-slab prior with MAF-dependent effect sizes.

    Effect SD ∝ [2f(1-f)]^{-arch_alpha/2}: rare variants are allowed larger effects,
    mimicking natural selection against large-effect common alleles.
    arch_alpha=0.75 is the typical LDSC estimate for complex traits.
    """
    rng = np.random.default_rng(seed)
    n_vars   = len(maf)
    n_causal = max(1, int(n_vars * p_causal))
    causal_idx = rng.choice(n_vars, n_causal, replace=False)

    het     = 2 * maf[causal_idx] * (1 - maf[causal_idx])
    sigma_j = (het + 1e-8) ** (-arch_alpha / 2)
    sigma_j = sigma_j / sigma_j.mean()          # normalise to mean 1

    true_betas = np.zeros(n_vars)
    true_betas[causal_idx] = rng.normal(0, 0.3 * sigma_j)
    print(f"  Spike-and-slab: {n_causal:,} causal variants "
          f"({100*p_causal:.3f}% of {n_vars:,}, arch_alpha={arch_alpha})")
    return true_betas


# ─── Phenotype simulation ─────────────────────────────────────────────────────

def simulate_phenotypes(G_qc, true_betas, seed=SEED):
    """
    Simulate continuous and binary phenotypes from QC-passed genotypes.

    Continuous: liability model, h²≈0.025 (genetic) + covariate effects + noise.
    Non-additive effects are injected at fixed loci (one dominant, one recessive),
    and one causal variant carries a sex-specific effect (sign-flipped between
    sexes) for the sex-stratified GWAS challenge.
    Binary: liability threshold at the 90th percentile (~10% cases).

    Returns y_cont, y_poly, y_bin, age, sex, dom_idx, rec_idx, sexspec_idx.
    """
    rng = np.random.default_rng(seed)
    N, M = G_qc.shape

    # Covariates first (sex is needed for the sex-specific genetic effect below).
    # Age: UKB age-at-recruitment distribution (Instance 0; range 37–73, median 58,
    # mean ≈ 56.5, sd ≈ 8.1). Quantile interpolation through the observed decile values.
    _p   = [0, .10, .20, .30, .40, .50, .60, .70, .80, .90, 1.0]
    _age = [37, 44,  48,  52,  55,  58,  60,  62,  64,  67,  73]
    age  = np.interp(rng.uniform(0, 1, N), _p, _age)
    sex = rng.binomial(1, 0.5, N).astype(np.int8)

    G_std = (G_qc - G_qc.mean(0)) / (G_qc.std(0) + 1e-8)
    genetic = G_std @ true_betas
    if genetic.std() > 0:
        genetic = genetic / genetic.std() * np.sqrt(0.025)  # h²≈0.025, 3 causal variants

    # Inject dominant, recessive and sex-specific loci at moderately frequent variants.
    # dominant: effect present for any copy (AB = BB); recessive: only BB
    af_qc = G_qc.mean(0) / 2   # approximate AF (0-imputed, slightly deflated)
    lo = np.percentile(af_qc, 70)   # use upper 30% by AF — enough BB homozygotes
    hi = np.percentile(af_qc, 95)   # avoid top 5% (very high AF, little BB contrast)
    common_idx = np.where((af_qc >= lo) & (af_qc <= hi))[0]
    rng2 = np.random.default_rng(seed + 1)
    chosen = rng2.choice(common_idx, 3, replace=False)
    dom_idx, rec_idx, sexspec_idx = int(chosen[0]), int(chosen[1]), int(chosen[2])

    dom_effect = np.where(G_qc[:, dom_idx] >= 1, 1.0, 0.0)   # dominant
    rec_effect = np.where(G_qc[:, rec_idx] == 2, 1.0, 0.0)   # recessive
    dom_effect = (dom_effect - dom_effect.mean()) / (dom_effect.std() + 1e-8) * np.sqrt(0.005)
    rec_effect = (rec_effect - rec_effect.mean()) / (rec_effect.std() + 1e-8) * np.sqrt(0.005)

    # Sex-specific effect: a moderate-frequency variant whose per-allele effect is SIGN-FLIPPED
    # between sexes (+ in males, − in females). A pooled GWAS averages the two ≈ to zero (looks
    # null); a sex-stratified GWAS recovers strong opposite-sign effects. Variance 0.01 → per-sex
    # R²≈0.01, easily genome-wide significant at N≈5,000 per stratum.
    sex_sign = np.where(sex == 1, 1.0, -1.0)
    sexspec_effect = G_std[:, sexspec_idx] * sex_sign
    sexspec_effect = sexspec_effect / (sexspec_effect.std() + 1e-8) * np.sqrt(0.01)

    noise_var = max(1 - 0.025 - 0.005 - 0.005 - 0.01, 0.01)  # genetic + dom + rec + sexspec
    y_cont = (genetic
              + dom_effect + rec_effect + sexspec_effect
              + 0.15 * (sex - 0.5)
              + 0.10 * (age - 57) / 8
              + rng.normal(0, np.sqrt(noise_var), N))
    y_cont = (y_cont - y_cont.mean()) / y_cont.std()

    threshold = np.percentile(y_cont, 90)   # top 10% → ~10% cases
    y_bin = (y_cont >= threshold).astype(np.int8)

    # Third phenotype: low h² (≈0.02), fully polygenic, independent of y_cont.
    # Betas are drawn as pure omnigenic background with no shared signal.
    rng3 = np.random.default_rng(seed + 2)
    betas_bg   = rng3.normal(0, 1, M) / np.sqrt(M)
    betas_poly = betas_bg / (np.linalg.norm(betas_bg) + 1e-8)

    y_poly_genetic_raw = G_std @ betas_poly
    if y_poly_genetic_raw.std() > 0:
        y_poly_genetic = y_poly_genetic_raw / y_poly_genetic_raw.std() * np.sqrt(0.02)
    else:
        y_poly_genetic = y_poly_genetic_raw

    noise_var_poly = max(1 - 0.02, 0.01)
    y_poly = y_poly_genetic + rng3.normal(0, np.sqrt(noise_var_poly), N)
    y_poly = (y_poly - y_poly.mean()) / y_poly.std()

    rg = np.corrcoef(y_poly_genetic, genetic / (genetic.std() + 1e-8))[0, 1]
    print(f"  Continuous trait: mean={y_cont.mean():.3f}, std={y_cont.std():.3f}")
    print(f"  Polygenic trait:  h²≈0.02, genetic correlation with y_cont ≈ {rg:.2f}  (should be ≈ 0)")
    print(f"  Binary trait: {y_bin.sum():,} cases "
          f"({100*y_bin.mean():.1f}%, liability threshold)")
    print(f"  Non-additive loci: dominant at G_qc col {dom_idx}, "
          f"recessive at G_qc col {rec_idx}")
    print(f"  Sex-specific locus: G_qc col {sexspec_idx} "
          f"(MAF {min(af_qc[sexspec_idx], 1-af_qc[sexspec_idx]):.2f}, "
          f"effect sign-flipped between sexes)")
    return y_cont, y_poly, y_bin, age, sex, dom_idx, rec_idx, sexspec_idx


# ─── Drosophila cross dataset ─────────────────────────────────────────────────

def make_fly_data(seed=SEED):
    """
    Simulate 2,000 offspring from a Drosophila test cross.
    6 X-linked genes at positions [0, 5, 15, 30, 35, 50] cM + 2 autosomal traits.
    Trait names are phenotype-descriptive and interleaved so students must
    analyse the data to identify which are X-linked.
    """
    rng = np.random.default_rng(seed)
    n_fly    = 2_000
    x_pos_cM = np.array([0.0, 5.0, 15.0, 30.0, 35.0, 50.0])
    # Column names: X-linked and autosomal interleaved; names don't reveal linkage
    # X-linked (indices 0-5 in x_phenos): eye, wing, leg, notch, vein, scute
    # Autosomal (indices 0-1 in auto_tr): thorax, band
    x_col_names   = ["trait_eye", "trait_wing", "trait_leg",
                      "trait_notch", "trait_vein", "trait_scute"]
    auto_col_names = ["trait_thorax", "trait_band"]
    n_male = n_fly // 2

    male_haps = np.zeros((n_male, 6), dtype=np.int8)
    for i in range(n_male):
        curr = rng.integers(0, 2)
        male_haps[i, 0] = curr
        for j in range(1, 6):
            dist_M = (x_pos_cM[j] - x_pos_cM[j-1]) / 100
            curr = (curr + rng.poisson(dist_M)) % 2
            male_haps[i, j] = curr

    female_haps = np.zeros((n_fly - n_male, 6), dtype=np.int8)
    fly_sex  = np.array([1] * n_male + [0] * (n_fly - n_male))
    x_phenos = np.vstack([male_haps, female_haps])
    auto_tr  = np.column_stack([rng.binomial(1, 0.35, n_fly),
                                 rng.binomial(1, 0.48, n_fly)])
    perm = rng.permutation(n_fly)

    # Interleave X-linked and autosomal columns so order doesn't give it away
    fly_df = pd.DataFrame({
        "sex":            fly_sex[perm],
        x_col_names[0]:   x_phenos[perm, 0],   # X-linked
        auto_col_names[0]:auto_tr[perm, 0],     # autosomal
        x_col_names[1]:   x_phenos[perm, 1],   # X-linked
        x_col_names[2]:   x_phenos[perm, 2],   # X-linked
        x_col_names[3]:   x_phenos[perm, 3],   # X-linked
        x_col_names[4]:   x_phenos[perm, 4],   # X-linked
        x_col_names[5]:   x_phenos[perm, 5],   # X-linked
        auto_col_names[1]:auto_tr[perm, 1],     # autosomal
    })
    true_positions = {n.replace("trait_", ""): p
                      for n, p in zip(x_col_names, x_pos_cM)}
    print(f"  Fly dataset: {n_fly:,} offspring "
          f"({n_male:,} males, {n_fly-n_male:,} females), "
          f"6 X-linked + 2 autosomal traits")
    print(f"  True gene order: "
          f"{' — '.join(f'{g}({p:.0f}cM)' for g, p in true_positions.items())}")
    return fly_df


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    print("=" * 65)
    print("SMARTbiomed GWAS Practical — Data Generation")
    print(f"  chr{CHROM}, {START_KBP:,}–{END_KBP:,} kbp  |  "
          f"N={N:,}  |  M={M_RAW:,} variants (pre-QC)")
    print(f"  MAF range: 0.1%–50%  |  binary trait prevalence: ~10%")
    print("=" * 65)

    # ── 1. Generate synthetic genotypes ──────────────────────────────────────
    print(f"\n[1/4] Generating block-LD genotypes (fully simulated)")
    bim, G = make_genotypes(n_samples=N, n_vars=M_RAW, chrom=CHROM,
                            start_kbp=START_KBP, end_kbp=END_KBP, seed=SEED)
    maf_raw = np.minimum(G.mean(0).astype(float) / 2,
                         1 - G.mean(0).astype(float) / 2)

    # ── 2. Inject QC failures ────────────────────────────────────────────────
    print(f"\n[2/4] Injecting QC-failing variants")
    G_raw = inject_qc_failures(G, maf_raw, seed=SEED)

    # ── 3. Simulate phenotypes ───────────────────────────────────────────────
    print(f"\n[3/4] Simulating phenotypes")

    # Apply QC to get a clean matrix for phenotype simulation
    miss_rate = np.isnan(G_raw).mean(0)
    maf       = np.minimum(np.nanmean(G_raw, 0) / 2, 1 - np.nanmean(G_raw, 0) / 2)
    G_int  = np.where(np.isnan(G_raw), -1, G_raw).astype(int)
    n_samp = (~np.isnan(G_raw)).sum(0).astype(float)
    n_AA   = (G_int == 0).sum(0).astype(float)
    n_AB   = (G_int == 1).sum(0).astype(float)
    n_BB   = (G_int == 2).sum(0).astype(float)
    p_h    = (2*n_BB + n_AB) / (2*n_samp + 1e-8)   # ALT allele frequency
    e_AA   = n_samp*(1-p_h)**2   # expected homozygous REF (0 ALT alleles)
    e_AB   = n_samp*2*p_h*(1-p_h)
    e_BB   = n_samp*p_h**2        # expected homozygous ALT (2 ALT alleles)
    hwe_c  = ((n_AA-e_AA)**2/(e_AA+1e-8) + (n_AB-e_AB)**2/(e_AB+1e-8)
               + (n_BB-e_BB)**2/(e_BB+1e-8))
    hwe_p  = stats.chi2.sf(hwe_c, df=1)
    qc_pass = (miss_rate < 0.05) & (maf > 0.01) & (hwe_p > 1e-6)
    print(f"  Post-QC: {qc_pass.sum():,} / {M_RAW:,} variants pass")

    G_qc = np.where(np.isnan(G_raw[:, qc_pass]), 0, G_raw[:, qc_pass])
    true_betas_raw = make_spike_slab_betas(maf_raw, p_causal=3/M_RAW, seed=SEED)
    true_betas_qc  = true_betas_raw[qc_pass]
    y_cont, y_poly, y_bin, age, sex, dom_idx_qc, rec_idx_qc, sexspec_idx_qc = \
        simulate_phenotypes(G_qc, true_betas_qc, seed=SEED)

    # ── 4. Save ──────────────────────────────────────────────────────────────
    print(f"\n[4/4] Saving")
    pos_kbp = bim["pos"].values // 1000
    rsids   = bim["snp"].values

    # Encode as int8 (-1=missing) to keep file size manageable (~150 MB compressed)
    G_save = np.where(np.isnan(G_raw), -1, G_raw).astype(np.int8)

    out_path = DATA_DIR / "gwas_data.npz"
    np.savez_compressed(
        out_path,
        G_raw       = G_save,          # int8: 0/1/2, -1=missing
        pos         = pos_kbp,
        rsids       = rsids,
        age         = age,
        sex         = sex,
        y_cont      = y_cont,          # continuous trait, h²=0.25, few causal variants
        y_poly      = y_poly,          # fully polygenic trait, h²≈0.02, uncorrelated with y_cont
        y_bin       = y_bin,           # binary trait, ~10% cases, liability threshold
        true_betas  = true_betas_raw,
        dom_idx_qc  = np.array([dom_idx_qc]),   # post-QC index of dominant locus
        rec_idx_qc  = np.array([rec_idx_qc]),   # post-QC index of recessive locus
        sexspec_idx_qc = np.array([sexspec_idx_qc]),  # post-QC index of sex-specific locus
    )
    print(f"  Saved: {out_path}  ({out_path.stat().st_size // 1e6:.0f} MB)")

    fly_df   = make_fly_data(seed=SEED)
    fly_path = DATA_DIR / "fly_data.csv"
    fly_df.to_csv(fly_path, index=False)
    print(f"  Saved: {fly_path}")

    print(f"\n✓ All data generated successfully.")
    print(f"  Output directory: {DATA_DIR}")


if __name__ == "__main__":
    main()
