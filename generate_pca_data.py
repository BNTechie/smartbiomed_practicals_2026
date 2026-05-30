#!/usr/bin/env python3
"""
Instructor data generation for Session 3 (population structure / PCA / relatedness).

Produces data/pca_data.npz — a structured-population genotype dataset with:
  - K=3 discrete populations (Balding–Nichols model, increasing Fst/diversity)
  - a set of admixed individuals (Dirichlet ancestry proportions)
  - a few injected sibling pairs (for the relatedness / GRM challenge)
  - y_strat : a phenotype confounded purely by population (no genetic effect)
  - y_clean : a phenotype with a handful of true causal variants (no confounding)

No downloads. Runtime < 1 minute.  DEPENDENCIES: numpy
"""
import numpy as np
from pathlib import Path

SEED      = 2026
M         = 20_000      # independent SNPs (no LD needed for PCA)
N_PER_POP = 300
N_ADMIX   = 150
N_SIB_PAIRS = 6
FST       = [0.05, 0.08, 0.12]   # pop 3 is the most diverged / diverse
DATA_DIR  = Path(__file__).resolve().parent / "data"
DATA_DIR.mkdir(exist_ok=True)


def bn_freqs(anc, fst, rng):
    """Balding–Nichols per-population allele frequencies given ancestral freqs."""
    a = anc * (1 - fst) / fst
    b = (1 - anc) * (1 - fst) / fst
    return np.clip(rng.beta(a, b), 1e-4, 1 - 1e-4)


def main():
    print("=" * 60)
    print("Session 3 data — structured populations (PCA / relatedness)")
    print(f"  {len(FST)} populations × {N_PER_POP} + {N_ADMIX} admixed, M={M:,} SNPs")
    print("=" * 60)
    rng = np.random.default_rng(SEED)
    K = len(FST)

    anc = rng.uniform(0.05, 0.95, M)                      # ancestral allele freqs
    popfreq = np.vstack([bn_freqs(anc, f, rng) for f in FST])   # (K, M)

    # ── Discrete populations ──────────────────────────────────────────────────
    G_blocks, pop = [], []
    for k in range(K):
        G_blocks.append(rng.binomial(2, popfreq[k][None, :], size=(N_PER_POP, M)))
        pop += [k] * N_PER_POP

    # ── Admixed individuals: per-allele population drawn from Dirichlet proportions ──
    adm_prop = rng.dirichlet(np.ones(K), N_ADMIX)         # (N_ADMIX, K)
    G_adm = np.zeros((N_ADMIX, M), dtype=np.int16)
    for i in range(N_ADMIX):
        for _ in range(2):                                # two alleles per genotype
            src = rng.choice(K, size=M, p=adm_prop[i])
            G_adm[i] += (rng.random(M) < popfreq[src, np.arange(M)]).astype(np.int16)
    G_blocks.append(G_adm); pop += [K] * N_ADMIX          # label K = "admixed"

    # ── Sibling pairs (population 0): transmit haplotypes from 2 parents ──────
    sib_rows, rel_pairs = [], []
    base = sum(b.shape[0] for b in G_blocks)
    for s in range(N_SIB_PAIRS):
        # parent haplotypes ~ population-0 frequencies
        ph = rng.random((4, M)) < popfreq[0][None, :]     # 4 haplotypes (2 per parent)
        cols = np.arange(M)
        for _ in range(2):                                # two siblings
            # per-locus transmission (free recombination) → siblings share ~50% IBD
            m = ph[rng.integers(0, 2, M),     cols]       # maternal allele per locus
            f = ph[2 + rng.integers(0, 2, M), cols]       # paternal allele per locus
            sib_rows.append((m.astype(np.int16) + f.astype(np.int16)))
        rel_pairs.append([base + 2*s, base + 2*s + 1])
    G_blocks.append(np.array(sib_rows, dtype=np.int16)); pop += [0] * (2*N_SIB_PAIRS)

    G   = np.vstack(G_blocks).astype(np.int8)
    pop = np.array(pop, dtype=np.int8)
    N   = G.shape[0]
    rel_pairs = np.array(rel_pairs)

    # ancestry proportions for everyone (one-hot for pure pops; Dirichlet for admixed)
    admix_frac = np.zeros((N, K))
    idx = 0
    for k in range(K):
        admix_frac[idx:idx+N_PER_POP, k] = 1.0; idx += N_PER_POP
    admix_frac[idx:idx+N_ADMIX] = adm_prop; idx += N_ADMIX
    admix_frac[idx:, 0] = 1.0                              # siblings are pop 0

    # ── Phenotypes ────────────────────────────────────────────────────────────
    G_std = (G - G.mean(0)) / (G.std(0) + 1e-8)

    # y_strat: environmental shift that differs by ancestry — NO genetic effect.
    pop_env = np.array([0.0, 0.7, 1.4])                    # mean per discrete population
    env = admix_frac[:, :K] @ pop_env                     # admixed = weighted average
    y_strat = env + rng.normal(0, 1.0, N)
    y_strat = (y_strat - y_strat.mean()) / y_strat.std()

    # y_clean: a few strong true causal variants (equal-magnitude effects at common SNPs so each
    # is clearly powered), no ancestry confounding.
    n_causal = 3
    common = np.where(np.minimum(anc, 1 - anc) > 0.2)[0]
    causal_idx = rng.choice(common, n_causal, replace=False)
    betas = np.zeros(M); betas[causal_idx] = rng.choice([-1.0, 1.0], n_causal)  # equal magnitude
    g = G_std @ betas
    g = g / g.std() * np.sqrt(0.30)
    y_clean = g + rng.normal(0, np.sqrt(0.70), N)
    y_clean = (y_clean - y_clean.mean()) / y_clean.std()

    age = np.clip(rng.normal(50, 8, N), 30, 70)
    sex = rng.binomial(1, 0.5, N).astype(np.int8)

    out = DATA_DIR / "pca_data.npz"
    np.savez_compressed(
        out,
        G=G, pop=pop, admix_frac=admix_frac.astype(np.float32),
        y_strat=y_strat.astype(np.float32), y_clean=y_clean.astype(np.float32),
        causal_idx=causal_idx.astype(np.int32), rel_pairs=rel_pairs.astype(np.int32),
        age=age.astype(np.float32), sex=sex,
        pop_names=np.array(['Pop1', 'Pop2', 'Pop3', 'Admixed']),
    )
    print(f"  N={N:,}  (pops {N_PER_POP}×{K} + {N_ADMIX} admixed + {2*N_SIB_PAIRS} sibs)")
    print(f"  causal variants for y_clean: {sorted(causal_idx.tolist())}")
    print(f"  sibling pairs (rows): {rel_pairs.tolist()}")
    print(f"  Saved: {out}  ({out.stat().st_size/1e6:.1f} MB)")


if __name__ == "__main__":
    main()
