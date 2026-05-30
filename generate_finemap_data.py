#!/usr/bin/env python3
"""
Instructor data generation for Session 4 (statistical fine-mapping).

Produces data/finemap_data.npz — two single-locus datasets with realistic LD:
  - Locus A: ONE causal variant sitting inside a tight cluster of near-perfectly-LD tag SNPs.
    The marginal lead SNP is usually a TAG, not the causal one → motivates fine-mapping. A 95%
    credible set narrows the ~significant peak to a handful of variants (incl. the causal).
  - Locus B: TWO causal variants (each with its own LD cluster) → breaks the single-causal
    assumption behind the simple credible set (SuSiE teaser).

LD model: an AR(1) latent process (smoothly decaying LD along the locus); each causal is then
surrounded by a few near-copy "tag" SNPs (allele flipped with small probability) to create the
high-LD cluster that makes fine-mapping non-trivial.

No downloads. Runtime < 1 minute.  DEPENDENCIES: numpy, scipy
"""
import numpy as np
from scipy.special import erfinv
from pathlib import Path

SEED     = 2026
N        = 5_000
RHO      = 0.90      # AR(1) LD decay of the background locus
P_FLIP   = 0.005     # tag SNPs are near-perfect copies of their causal (r² ≈ 0.98)
N_TAG    = 7         # tag SNPs per causal
DATA_DIR = Path(__file__).resolve().parent / "data"
DATA_DIR.mkdir(exist_ok=True)


def make_locus(M, causals, rng):
    """AR(1)-LD locus; each causal gets a cluster of near-copy tag SNPs around it. Returns G."""
    maf = rng.uniform(0.10, 0.50, M)
    thr = np.sqrt(2) * erfinv(2 * maf - 1)          # latent thresholds for target MAF
    nh  = 2 * N

    def haplotypes():
        z = np.empty((nh, M), dtype=np.float32)
        z[:, 0] = rng.standard_normal(nh)
        s = np.sqrt(1 - RHO**2)
        for j in range(1, M):
            z[:, j] = RHO * z[:, j-1] + s * rng.standard_normal(nh)
        H = (z < thr).astype(np.int8)
        for c in causals:                            # tight LD cluster of near-copies
            for col in [c-3, c-2, c-1, c+1, c+2, c+3, c+4][:N_TAG]:
                H[:, col] = H[:, c] ^ (rng.random(nh) < P_FLIP).astype(np.int8)
        return H

    return (haplotypes()[:N] + haplotypes()[N:]).astype(np.int8)


def pheno(G, causals, var_each, rng):
    Gs = (G - G.mean(0)) / (G.std(0) + 1e-8)
    g = Gs[:, causals] @ (np.sqrt(var_each) * rng.choice([-1.0, 1.0], len(causals)))
    y = g + rng.normal(0, np.sqrt(max(1 - var_each * len(causals), 0.01)), N)
    return ((y - y.mean()) / y.std()).astype(np.float32)


def _lead(G, y):
    Gc = G.astype(float) - G.mean(0); ys = y - y.mean()
    ss = (Gc**2).sum(0); b = Gc.T @ ys / ss
    n = N - 2; se = np.sqrt(((ys@ys) - b**2*ss) / n / ss)
    return np.abs(b/se)


def main():
    print("=" * 60)
    print("Session 4 data — fine-mapping loci")
    print("=" * 60)
    rng = np.random.default_rng(SEED)

    MA, causalA = 400, 200
    GA = make_locus(MA, [causalA], rng)
    yA = pheno(GA, [causalA], var_each=0.03, rng=rng)
    posA = (np.arange(MA) * 2.0).astype(np.int32)            # kbp

    MB, causalB = 400, [120, 280]
    GB = make_locus(MB, causalB, rng)
    yB = pheno(GB, causalB, var_each=0.025, rng=rng)
    posB = (np.arange(MB) * 2.0).astype(np.int32)

    out = DATA_DIR / "finemap_data.npz"
    np.savez_compressed(
        out,
        GA=GA, posA=posA, yA=yA, causalA=np.array([causalA], np.int32),
        GB=GB, posB=posB, yB=yB, causalB=np.array(causalB, np.int32),
    )
    zA = _lead(GA, yA); lead = int(np.argmax(zA))
    print(f"  Locus A: causal={causalA}, marginal lead={lead} "
          f"({'SAME' if lead == causalA else 'a TAG SNP'}), max|z|={zA.max():.1f}")
    print(f"  Locus B: causals={causalB}")
    print(f"  Saved: {out}  ({out.stat().st_size/1e6:.1f} MB)")


if __name__ == "__main__":
    main()
