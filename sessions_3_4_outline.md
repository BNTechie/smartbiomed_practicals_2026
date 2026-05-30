# Afternoon practicals (Pier) — Sessions 3 & 4: question design

Intuition-first practicals built from scratch with NumPy/SciPy/Matplotlib — no specialised genetics
packages and no niche Python. Both follow the morning's pattern: skeleton `practical.ipynb` with
`???` blanks, an `answers.ipynb` with collapsible solutions, an executed `run.ipynb`, Colab badges,
and runtime data download. Data is simulated by `generate_pca_data.py` and `generate_finemap_data.py`.

---

## Session 3 — Complexities of GWAS: population structure & relatedness

**Data** (`data/pca_data.npz`): an **"All of Us"-style** diverse cohort (~2,000 participants drawn as
admixtures over 5 continental components on a population tree, incl. African-American and
Hispanic/Latino clines) **plus a "1000 Genomes"-style reference panel** (75 unadmixed individuals per
superpopulation AFR/AMR/EAS/EUR/SAS, with known labels). ~20,000 independent SNPs; a few sibling
pairs hidden in the cohort. `y_strat` = ancestry-confounded phenotype (no genetics); `y_clean` = a
few true causal variants. `true_anc` holds each cohort member's real ancestry proportions (answer key).

| # | Question | What it teaches | The "aha" |
|---|----------|-----------------|-----------|
| Part 1 | GWAS of `y_strat` (no covariates) → QQ + λ_GC | Confounding by ancestry | Thousands of "hits" with **zero** causal variants → λ_GC ≈ 11 |
| Part 2 | **PCA from scratch** (SVD) on reference + cohort together; colour reference by superpop, cohort grey | PCs are the ancestry axes; a labelled panel names the clusters | Cohort forms a continuous spread anchored by the reference clusters — the All of Us picture |
| Part 3 | Train a **random forest** on the reference PCs; predict each cohort member's superpop; apply a **confidence threshold**; re-plot coloured by assigned ancestry | How biobanks actually assign continental ancestry | Clearly-defined groups get labelled; **admixed individuals fall below threshold → "Unassigned"** |
| Part 4 | Re-run GWAS of `y_strat` **with top PCs as covariates**; re-run `y_clean` too | PCA corrects stratification | λ_GC → 1, false hits vanish, **true** hits (3/3) survive |
| Challenge 1 (~12 min) | Confirm "Unassigned" = the admixed (via `true_anc`); sweep the threshold (coverage vs confidence) | The cost of a hard ancestry cutoff | Unassigned people have the lowest *dominant* ancestry fraction; a stricter threshold drops the admixed first |
| Challenge 2 (~12 min) | Build the GRM = ZZᵀ/M; find the hidden sibling pairs | Relatedness violates independence | Sib pairs show GRM ≈ 0.5 → motivates mixed models |

Helpers reused: `run_gwas` (from the morning). New: `qq_plot`, `lambda_gc` (in the setup cell).
Standard linear algebra (`np.linalg.svd`), matrix products, regression; the only ready-made tool is
`sklearn.ensemble.RandomForestClassifier` for the ancestry-assignment step.

---

## Session 4 — Fine-mapping

**Data** (`data/finemap_data.npz`): two GWAS loci (~400 variants, N=5,000) with realistic LD.
Locus A = 1 causal variant inside a tight cluster of near-perfect-LD tags; Locus B = 2 causal.

| # | Question | What it teaches | The "aha" |
|---|----------|-----------------|-----------|
| Part 1 | Regional GWAS + LocusZoom (colour by r² with lead) | LD makes a whole peak significant | ~8 significant variants; the marginal lead is often a **tag**, not the causal |
| Part 2 | **Conditional analysis** — add the lead as a covariate, re-run, repeat, **re-drawing the LocusZoom each round** | Counting independent signals | Watch the peak collapse: conditioning on the lead makes the rest of the peak vanish (1 signal at Locus A) |
| Part 3 | **PIP & 95% credible set** via Wakefield ABF from z = β̂/se | Posterior prob. of being causal | The significant peak collapses to a ~6-variant credible set containing the causal |
| Challenge 1 (~10 min) | PIP LocusZoom (height = PIP) | Visualising the credible set | A handful of high-PIP variants carry essentially all the posterior |
| Challenge 2 (~10 min) | Recompute the credible set at N/2, N/4 | Resolution scales with power (and LD) | The set grows as power falls; LD ultimately caps resolution |
| Challenge 3 (~12 min) | Locus B (2 causal): single-causal credible set, then condition | Single-causal assumption breaks | One credible set can't capture two signals → conditioning/SuSiE recovers the second |

Helpers reused: `run_gwas`. New: `r2_with` (locus LD). The Bayes factor is one closed-form line
(`0.5*log(1-r) + 0.5*z²*r`); credible set = sort PIP, cumulative sum ≥ 0.95. No niche functions.

---

### Notes
- Both datasets are simulated locally (no external downloads); bundled in `data/` via Git LFS and
  auto-downloaded by the notebooks on Colab.
- Timings are rough; Parts 1–3 ≈ 35 min, challenges for fast finishers.
- Possible extensions: LDSC-intercept vs λ_GC (stratification vs polygenicity); a tiny variational
  / spike-and-slab inference toy (Session 3 computation theme); a from-scratch SuSiE single-effect
  iteration (Session 4).
