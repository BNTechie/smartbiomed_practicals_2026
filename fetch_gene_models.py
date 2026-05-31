#!/usr/bin/env python3
"""
Instructor data prep for the Session 2 "GWAS vs exome" gene-map challenge.

Bundles, for chromosome 16 (GRCh38):
  - canonical **MANE Select** gene models (gene body, exons, and coding CDS) for every
    protein-coding gene, parsed from the NCBI MANE flat file; and
  - the Pan-UKB BMI GWAS variants on chr16 **lifted over from GRCh37 → GRCh38** (via pyliftover),
    so the common-variant GWAS lines up exactly with the GRCh38 gene models and the GRCh38
    Genebass exome variants — no per-plot build fudging needed.

Output: data/gene_models_chr16.npz

This lets the notebook (a) draw a coding-exon track across chr16 and quantify the fraction of GWAS
vs exome variants that fall in coding sequence, and (b) zoom into FTO (GWAS peak in intron 1, no
coding hits) and 16p11.2 (exome coding hits sitting on exons), all on one coordinate system.

No runtime API calls in the notebook. DEPENDENCIES (instructor only): numpy, pyliftover (+ curl, gzip).
"""
import gzip
import re
import subprocess
from pathlib import Path

import numpy as np

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(exist_ok=True)
TMP = Path("/tmp")

MANE_URL = ("https://ftp.ncbi.nlm.nih.gov/refseq/MANE/MANE_human/current/"
            "MANE.GRCh38.v1.5.ensembl_genomic.gff.gz")
CHROM = "chr16"
_NAME = re.compile(r"gene_name=([^;]+)")


def download_mane():
    path = TMP / "MANE.GRCh38.ensembl_genomic.gff.gz"
    if path.exists() and path.stat().st_size > 1_000_000:
        print(f"  reusing cached {path} ({path.stat().st_size/1e6:.1f} MB)")
        return path
    print(f"  downloading {MANE_URL} ...")
    subprocess.run(f'curl -sL --max-time 300 "{MANE_URL}" -o "{path}"', shell=True, check=True)
    return path


def parse_chr16(path):
    """Return per-gene MANE Select models on chr16 (GRCh38)."""
    genes = {}            # name -> dict(start, end, strand, exons=[], cds=[])
    with gzip.open(path, "rt") as f:
        for line in f:
            if line.startswith("#"):
                continue
            c = line.rstrip("\n").split("\t")
            if len(c) < 9 or c[0] != CHROM:
                continue
            feat, start, end, strand, attr = c[2], int(c[3]), int(c[4]), c[6], c[8]
            m = _NAME.search(attr)
            if not m:
                continue
            name = m.group(1)
            g = genes.setdefault(name, dict(start=start, end=end, strand=strand,
                                            exons=[], cds=[]))
            if feat == "gene":
                g.update(start=start, end=end, strand=strand)
            elif feat == "exon":
                g["exons"].append((start, end))
            elif feat == "CDS":
                g["cds"].append((start, end))
    return genes


def liftover_chr16_gwas():
    """Lift the chr16 Pan-UKB BMI GWAS variants from GRCh37 to GRCh38."""
    from pyliftover import LiftOver
    lo = LiftOver("hg19", "hg38")
    rs = np.load(DATA_DIR / "sumstats_real.npz", allow_pickle=True)
    m = rs["bmi_chrom"] == 16
    pos37 = rs["bmi_pos"][m].astype(np.int64)
    nlp = rs["bmi_nlog10p"][m]
    pos38 = np.full(len(pos37), -1, np.int64)
    for i, p in enumerate(pos37):
        r = lo.convert_coordinate("chr16", int(p))
        if r:
            pos38[i] = r[0][1]
    ok = pos38 > 0
    print(f"  GWAS chr16: lifted {ok.sum():,}/{len(ok):,} variants 37->38")
    return pos37[ok], pos38[ok], nlp[ok]


def main():
    print("=" * 64)
    print("Session 2 gene models — MANE Select chr16 + lifted GWAS")
    print("=" * 64)
    genes = parse_chr16(download_mane())
    print(f"  parsed {len(genes):,} chr16 MANE genes")

    names = sorted(genes)
    gene_name = np.array(names)
    gene_start = np.array([genes[n]["start"] for n in names], np.int32)
    gene_end = np.array([genes[n]["end"] for n in names], np.int32)
    gene_strand = np.array([genes[n]["strand"] for n in names])

    ex_gi, ex_s, ex_e, cds_gi, cds_s, cds_e = [], [], [], [], [], []
    for gi, n in enumerate(names):
        for s, e in genes[n]["exons"]:
            ex_gi.append(gi); ex_s.append(s); ex_e.append(e)
        for s, e in genes[n]["cds"]:
            cds_gi.append(gi); cds_s.append(s); cds_e.append(e)

    gw_pos37, gw_pos38, gw_nlp = liftover_chr16_gwas()

    out = DATA_DIR / "gene_models_chr16.npz"
    np.savez_compressed(
        out,
        gene_name=gene_name, gene_start=gene_start, gene_end=gene_end, gene_strand=gene_strand,
        ex_gi=np.array(ex_gi, np.int32), ex_start=np.array(ex_s, np.int32), ex_end=np.array(ex_e, np.int32),
        cds_gi=np.array(cds_gi, np.int32), cds_start=np.array(cds_s, np.int32), cds_end=np.array(cds_e, np.int32),
        gw_pos37=gw_pos37.astype(np.int64), gw_pos38=gw_pos38.astype(np.int64),
        gw_nlog10p=gw_nlp.astype(np.float32),
    )
    fto = names.index("FTO")
    print(f"  FTO (MANE, GRCh38): {gene_start[fto]:,}-{gene_end[fto]:,}  "
          f"({(np.array(ex_gi) == fto).sum()} exons)")
    print(f"  total exons={len(ex_gi):,}  CDS={len(cds_gi):,}")
    print(f"  Saved: {out}  ({out.stat().st_size/1e6:.2f} MB)")


if __name__ == "__main__":
    main()
