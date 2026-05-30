#!/usr/bin/env python3
"""
Fetch + thin real Pan-UKB GWAS summary statistics for the Session 2 practical.

Downloads three Pan-UKB (EUR) genome-wide summary-statistic flat files, thins them
to a small bundled artefact, and writes data/sumstats_real.npz.

Traits (parallel to the simulated y_cont / y_bin / y_poly):
  - ldl    : LDL direct           (continuous biomarker, field 30780)
  - cad    : I25 chronic ischaemic heart disease (binary, ICD10)
  - bmi    : body mass index      (highly polygenic continuous, field 21001)

The Pan-UKB flat files are ~2-2.7 GB each. Rather than storing them, we STREAM
(curl | gzip -dc | awk) and thin on the fly:
  - keep every genome-wide-significant variant (p < 5e-8) so Manhattan towers survive
  - keep a uniform random ~RAND_PROB subset (unbiased for QQ / lambda_GC)
  - restrict to EUR, low_confidence==false, non-missing beta/se/p, MAF >= 0.5%
p-values are kept in -log10 units (neglog10_pval) to avoid float underflow at the tail.

No private data: these are public Pan-UKB downloads. Runtime: a few minutes per file
(network-bound). Re-run only when refreshing the bundled file.

DEPENDENCIES: numpy  (+ standard curl, gzip, awk)
"""
import csv
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(exist_ok=True)
TMP_DIR = Path("/tmp")

# Pan-UKB S3 flat files (EUR meta columns used downstream)
TRAITS = {
    "ldl":    "https://pan-ukb-us-east-1.s3.amazonaws.com/sumstats_flat_files/biomarkers-30780-both_sexes-irnt.tsv.bgz",
    "cad":    "https://pan-ukb-us-east-1.s3.amazonaws.com/sumstats_flat_files/icd10-I25-both_sexes.tsv.bgz",
    "bmi":    "https://pan-ukb-us-east-1.s3.amazonaws.com/sumstats_flat_files/continuous-21001-both_sexes-irnt.tsv.bgz",
}

POP        = "EUR"      # homogeneous-ancestry analysis (matches lecture framing)
MAF_MIN    = 0.005      # drop ultra-rare
SIG_NLOG10 = 7.30103    # -log10(5e-8): force-keep genome-wide-significant variants
RAND_PROB  = 0.012      # uniform-subset keep prob → ~150k SNPs after QC filtering
SEED       = 42


def header_columns(url):
    """Stream just the header line and return {column_name: 1-based index}."""
    cmd = f'curl -sL --max-time 120 "{url}" | gzip -dc 2>/dev/null | head -1'
    out = subprocess.check_output(cmd, shell=True, text=True)
    names = out.rstrip("\n").split("\t")
    return {name: i + 1 for i, name in enumerate(names)}


def thin_trait(name, url):
    """Stream the Pan-UKB file and write a thinned TSV; return its path."""
    out_path = TMP_DIR / f"sumstats_thin_{name}.tsv"
    if out_path.exists() and out_path.stat().st_size > 0:
        print(f"[{name}] reusing cached {out_path} "
              f"({out_path.stat().st_size/1e6:.1f} MB)", flush=True)
        return out_path

    cols = header_columns(url)
    # Continuous traits expose af_{POP}; binary (case/control) traits expose
    # af_controls_{POP} / af_cases_{POP} instead — use control AF as the population proxy.
    if f"af_{POP}" in cols:
        af_col = f"af_{POP}"
    elif f"af_controls_{POP}" in cols:
        af_col = f"af_controls_{POP}"
    else:
        raise KeyError(f"[{name}] no af_{POP} or af_controls_{POP} "
                       f"(have e.g. {list(cols)[:8]} ...)")
    need = {
        "chr": "chr", "pos": "pos",
        "af":  af_col,       "beta": f"beta_{POP}",
        "se":  f"se_{POP}",  "nlp":  f"neglog10_pval_{POP}",
        "lc":  f"low_confidence_{POP}",
    }
    idx = {}
    for k, colname in need.items():
        if colname not in cols:
            raise KeyError(f"[{name}] expected column '{colname}' not found "
                           f"(have e.g. {list(cols)[:8]} ...)")
        idx[k] = cols[colname]
    awk = (
        "awk -F'\\t' "
        f"-v c={idx['chr']} -v p={idx['pos']} -v af={idx['af']} -v b={idx['beta']} "
        f"-v se={idx['se']} -v nlp={idx['nlp']} -v lc={idx['lc']} "
        f"-v prob={RAND_PROB} -v sig={SIG_NLOG10} -v mafmin={MAF_MIN} "
        "'BEGIN{srand(" + str(SEED) + ")} "
        "NR==1{next} "
        "{"
        " afv=$af; if(afv==\"NA\")next;"
        " if($lc!=\"false\")next;"
        " bv=$b; sev=$se; pv=$nlp;"
        " if(bv==\"NA\"||sev==\"NA\"||pv==\"NA\")next;"
        " maf=afv+0; if(maf>0.5)maf=1-maf;"
        " if(maf<mafmin)next;"
        " issig=(pv+0>=sig)?1:0;"
        " isrand=(rand()<prob)?1:0;"
        " if(issig||isrand) print $c\"\\t\"$p\"\\t\"maf\"\\t\"bv\"\\t\"sev\"\\t\"pv\"\\t\"issig\"\\t\"isrand;"
        "}'"
    )
    cmd = f'curl -sL --max-time 540 "{url}" | gzip -dc 2>/dev/null | {awk} > "{out_path}"'
    print(f"[{name}] streaming + thinning (EUR cols "
          f"af={idx['af']},beta={idx['beta']},p={idx['nlp']}) ...", flush=True)
    subprocess.run(cmd, shell=True, check=True)
    n = sum(1 for _ in open(out_path))
    print(f"[{name}] thinned rows: {n:,}  → {out_path}", flush=True)
    return out_path


def load_thinned(path):
    """Load a thinned TSV into typed numpy arrays."""
    # chr may be 'X' — keep as string then map; here Pan-UKB uses 1-22, X, Y
    chrom, pos, maf, beta, se, nlp, sig, rand = [], [], [], [], [], [], [], []
    with open(path) as f:
        for line in f:
            a = line.rstrip("\n").split("\t")
            c = a[0]
            c = 23 if c == "X" else 24 if c == "Y" else (25 if c in ("XY", "MT") else int(c))
            chrom.append(c); pos.append(int(a[1])); maf.append(float(a[2]))
            beta.append(float(a[3])); se.append(float(a[4])); nlp.append(float(a[5]))
            sig.append(int(a[6])); rand.append(int(a[7]))
    return (np.array(chrom, np.int16), np.array(pos, np.int64),
            np.array(maf, np.float32), np.array(beta, np.float32),
            np.array(se, np.float32), np.array(nlp, np.float32),
            np.array(sig, bool), np.array(rand, bool))


def lambda_gc_from_nlog10(nlog10p):
    """lambda_GC from -log10(p): convert to chi2(1) via survival function inverse."""
    from scipy import stats
    p = np.power(10.0, -np.clip(nlog10p, 0, 300))
    chi2 = stats.chi2.isf(np.clip(p, 1e-300, 1), df=1)
    return float(np.median(chi2) / stats.chi2.ppf(0.5, df=1))


def main():
    print("=" * 65)
    print("Pan-UKB real summary statistics — fetch + thin (EUR)")
    print(f"  traits: {', '.join(TRAITS)}  |  MAF>={MAF_MIN}  |  rand_prob={RAND_PROB}")
    print("=" * 65)

    arrays = {}
    summary = {}
    for name, url in TRAITS.items():
        path = thin_trait(name, url)
        chrom, pos, maf, beta, se, nlp, sig, rand = load_thinned(path)
        arrays[f"{name}_chrom"] = chrom
        arrays[f"{name}_pos"]   = pos
        arrays[f"{name}_maf"]   = maf
        arrays[f"{name}_beta"]  = beta
        arrays[f"{name}_se"]    = se
        arrays[f"{name}_nlog10p"] = nlp     # p in -log10 units (avoids underflow)
        arrays[f"{name}_sig"]   = sig       # p < 5e-8
        arrays[f"{name}_rand"]  = rand      # member of unbiased random subset
        lam = lambda_gc_from_nlog10(nlp[rand])   # lambda on the unbiased subset
        summary[name] = dict(n=int(len(chrom)), n_sig=int(sig.sum()),
                             n_rand=int(rand.sum()), lambda_gc=round(lam, 3))
        print(f"[{name}] total={len(chrom):,}  sig={int(sig.sum()):,}  "
              f"rand={int(rand.sum()):,}  lambda_GC(rand)={lam:.3f}")

    out = DATA_DIR / "sumstats_real.npz"
    np.savez_compressed(out, meta=json.dumps(summary), **arrays)
    print(f"\nSaved: {out}  ({out.stat().st_size/1e6:.1f} MB)")
    print("Per-trait summary:")
    for name, s in summary.items():
        print(f"  {name:6s}: {s['n']:>8,} SNPs  ({s['n_sig']:,} sig, "
              f"{s['n_rand']:,} in random subset)  lambda_GC={s['lambda_gc']}")
    print("\n✓ Real sumstats bundled.")


if __name__ == "__main__":
    sys.exit(main())
