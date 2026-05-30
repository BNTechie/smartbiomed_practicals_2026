#!/usr/bin/env python3
"""
Instructor data generation for Session 3 (population structure / PCA / ancestry inference).

Produces data/pca_data.npz — an "All of Us"-style diverse cohort plus a small
"1000 Genomes"-style reference panel with known continental superpopulation labels:

  - 5 ancestral components on a population tree (AFR diverged first; EUR/EAS/SAS/AMR
    descend from an out-of-Africa node) → realistic continental allele-frequency divergence.
  - REFERENCE panel: ~50 unadmixed individuals per superpopulation (AFR, AMR, EAS,
    EUR, SAS) with ground-truth labels — the "1000 Genomes" anchor used to *label* clusters.
  - TARGET cohort: ~2,000 diverse participants drawn as admixtures of the components
    (including African-American and Hispanic/Latino clines) → a continuous "All of Us" PCA.
  - a few injected sibling pairs (relatedness / GRM challenge).
  - y_strat: a phenotype confounded purely by continental ancestry (no genetics).
  - y_clean: a phenotype with a few true causal variants (no confounding).

No downloads. Runtime ~1 minute.  DEPENDENCIES: numpy
"""
import numpy as np
from pathlib import Path

SEED        = 2026
M           = 20_000      # independent SNPs (no LD needed for PCA)
N_REF       = 75          # reference individuals per superpopulation
N_SIB_PAIRS = 6
DATA_DIR    = Path(__file__).resolve().parent / "data"
DATA_DIR.mkdir(exist_ok=True)

SUPERPOPS = ['AFR', 'AMR', 'EAS', 'EUR', 'SAS']            # index 0..4
EUR = SUPERPOPS.index('EUR')


def drift(p, fst, rng):
    """Balding–Nichols allele-frequency drift from parent freqs p by Wright's Fst."""
    a = p * (1 - fst) / fst
    b = (1 - p) * (1 - fst) / fst
    return np.clip(rng.beta(a, b), 1e-4, 1 - 1e-4)


def draw_admixed(Q, F, rng):
    """Per-locus admixture: each of the two alleles is drawn from a component chosen
    with probability given by the individual's ancestry proportions Q[i]."""
    n = Q.shape[0]
    G = np.zeros((n, M), dtype=np.int8)
    cols = np.arange(M)
    for i in range(n):
        for _ in range(2):                                # two alleles per genotype
            src = rng.choice(len(SUPERPOPS), size=M, p=Q[i])
            G[i] += (rng.random(M) < F[src, cols]).astype(np.int8)
    return G


def grp(rng, mean, conc, n):
    """n individuals with Dirichlet ancestry proportions concentrated near `mean`."""
    return rng.dirichlet(conc * np.asarray(mean, float), n)


def main():
    print("=" * 64)
    print("Session 3 data — 'All of Us'-style cohort + '1000 Genomes' reference")
    print("=" * 64)
    rng = np.random.default_rng(SEED)

    # ── Population tree → per-superpopulation allele frequencies ────────────────
    anc = rng.uniform(0.05, 0.95, M)                      # root (ancestral) freqs
    ooa = drift(anc, 0.06, rng)                           # out-of-Africa node
    freq = {
        'AFR': drift(anc, 0.10, rng),                     # diverged first; most diverse
        'EUR': drift(ooa, 0.05, rng),
        'EAS': drift(ooa, 0.08, rng),
        'SAS': drift(ooa, 0.07, rng),
        'AMR': drift(ooa, 0.12, rng),                     # extra drift → spreads out
    }
    F = np.vstack([freq[s] for s in SUPERPOPS])           # (5, M)

    # ── Reference panel ("1000 Genomes"): unadmixed, labelled by superpopulation ─
    G_ref, ref_pop = [], []
    for k in range(len(SUPERPOPS)):
        G_ref.append(rng.binomial(2, F[k][None, :], size=(N_REF, M)))
        ref_pop += [k] * N_REF
    G_ref  = np.vstack(G_ref).astype(np.int8)
    ref_pop = np.array(ref_pop, np.int8)

    # ── Target cohort ("All of Us"): admixed recruitment groups ─────────────────
    #          ancestry mean over [AFR, AMR, EAS, EUR, SAS], concentration, n
    specs = [
        ('European',  [0.02, 0.02, 0.02, 0.92, 0.02], 120, 700),
        ('Afr-Amer',  [0.80, 0.03, 0.02, 0.13, 0.02],  25, 350),  # AFR–EUR cline
        ('Hispanic',  [0.10, 0.45, 0.02, 0.40, 0.03],   8, 400),  # AMR–EUR–AFR cline
        ('East Asian',[0.02, 0.03, 0.90, 0.03, 0.02], 100, 200),
        ('South Asian',[0.02,0.03, 0.06, 0.07, 0.82],  60, 250),
        ('Admixed',   [0.20, 0.20, 0.20, 0.20, 0.20],   4, 100),  # broadly admixed
    ]
    Q_parts, group = [], []
    for name, mean, conc, n in specs:
        Q_parts.append(grp(rng, mean, conc, n))
        group += [name] * n
    Q = np.vstack(Q_parts)
    G_target = draw_admixed(Q, F, rng)

    # ── Injected sibling pairs (European), appended to the cohort ───────────────
    sib_rows, rel_pairs = [], []
    base = G_target.shape[0]
    cols = np.arange(M)
    for s in range(N_SIB_PAIRS):
        ph = rng.random((4, M)) < F[EUR][None, :]          # 4 parental haplotypes
        for _ in range(2):                                 # two siblings, free recombination
            m = ph[rng.integers(0, 2, M),     cols]
            f = ph[2 + rng.integers(0, 2, M), cols]
            sib_rows.append((m.astype(np.int8) + f.astype(np.int8)))
        rel_pairs.append([base + 2*s, base + 2*s + 1])
    G_target = np.vstack([G_target, np.array(sib_rows, np.int8)])
    sib_Q = np.zeros((2*N_SIB_PAIRS, len(SUPERPOPS))); sib_Q[:, EUR] = 1.0
    Q = np.vstack([Q, sib_Q]); group += ['European'] * (2*N_SIB_PAIRS)

    Q = Q.astype(np.float32)
    group = np.array(group)
    rel_pairs = np.array(rel_pairs)
    N = G_target.shape[0]

    # ── Drop SNPs that are monomorphic in the cohort or the reference panel ─────
    # (a fixed SNP carries no information and would make run_gwas divide by zero).
    keep = (G_target.std(0) > 0) & (G_ref.std(0) > 0)
    G_target = G_target[:, keep]
    G_ref    = G_ref[:, keep]
    anc      = anc[keep]
    M_kept   = int(keep.sum())

    # ── Phenotypes (defined on the target cohort that gets the GWAS) ────────────
    G_std = (G_target - G_target.mean(0)) / (G_target.std(0) + 1e-8)

    # y_strat: environmental shift that differs by continental ancestry — NO genetics.
    comp_env = np.array([1.4, 0.8, -0.5, 0.0, 0.4])        # mean per [AFR,AMR,EAS,EUR,SAS]
    y_strat = Q @ comp_env + rng.normal(0, 1.0, N)
    y_strat = (y_strat - y_strat.mean()) / y_strat.std()

    # y_clean: a few strong, equal-magnitude true causal variants at common SNPs.
    n_causal = 3
    common = np.where(np.minimum(anc, 1 - anc) > 0.2)[0]
    causal_idx = rng.choice(common, n_causal, replace=False)
    betas = np.zeros(M_kept); betas[causal_idx] = rng.choice([-1.0, 1.0], n_causal)
    g = G_std @ betas
    g = g / g.std() * np.sqrt(0.30)
    y_clean = g + rng.normal(0, np.sqrt(0.70), N)
    y_clean = (y_clean - y_clean.mean()) / y_clean.std()

    age = np.clip(rng.normal(50, 8, N), 30, 70)
    sex = rng.binomial(1, 0.5, N).astype(np.int8)

    out = DATA_DIR / "pca_data.npz"
    np.savez_compressed(
        out,
        G=G_target, G_ref=G_ref, ref_pop=ref_pop,
        superpop_names=np.array(SUPERPOPS),
        true_anc=Q, group=group,
        y_strat=y_strat.astype(np.float32), y_clean=y_clean.astype(np.float32),
        causal_idx=causal_idx.astype(np.int32), rel_pairs=rel_pairs.astype(np.int32),
        age=age.astype(np.float32), sex=sex,
    )

    # ── Sanity check: PCA separation of the reference panel ─────────────────────
    G_all = np.vstack([G_ref, G_target]).astype(np.float32)
    Z = (G_all - G_all.mean(0)) / (G_all.std(0) + 1e-8)
    U, S, _ = np.linalg.svd(Z / np.sqrt(G_all.shape[0]), full_matrices=False)
    PC = (U * S)[:, :2]
    print(f"  cohort N={N:,}  +  reference N={len(ref_pop)}  ({N_REF}/superpop)   "
          f"M={M_kept:,} (dropped {M - M_kept} monomorphic)")
    print(f"  recruitment groups: "
          f"{', '.join(f'{n}×{nm}' for nm, _, _, n in specs)} + {2*N_SIB_PAIRS} sibs")
    print("  reference mean PC1/PC2 per superpopulation:")
    for k, s in enumerate(SUPERPOPS):
        m = ref_pop == k
        print(f"    {s}: PC1={PC[:len(ref_pop)][m,0].mean():+.2f}  "
              f"PC2={PC[:len(ref_pop)][m,1].mean():+.2f}")
    print(f"  causal variants for y_clean: {sorted(causal_idx.tolist())}")
    print(f"  sibling pairs (rows): {rel_pairs.tolist()}")
    print(f"  Saved: {out}  ({out.stat().st_size/1e6:.1f} MB)")


if __name__ == "__main__":
    main()
