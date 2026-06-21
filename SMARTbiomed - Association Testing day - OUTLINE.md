PARAMETERS

20-40 students per class

Practicals:

- Create GitHub repo with jupyter notebooks for practicals, run on Google Colab  
- We want lectures and practicals with a strong focus on understanding what is going on under the hood \- not just plug and play with existing software.  
- We want each practical to start simple (skeleton code with blanks to fill in) but include several open-ended challenging questions at the end, requiring writing code from scratch, in case there are students who progress rapidly 
- Provide the basic functions (i.e. linear regression) for students. 
- Create answer sheet for students to check answers after class (or during practical, if stuck). The answer sheet should be the same notebook, with hidden/commented-out solution cells. Include hints in the worksheet notebook if students are stuck. If they are really stuck, they can simply refer to the answer sheet notebook.
- The primary input data for the practicals should be pre-generated. 
- Library preferences: Generally, try to keep things lightweight, unless using a larger package makes code significantly more readable. Store large matrices in numpy. Prefer matplotlib over seaborn, unless absolutely necessary. scipy.stats over statsmodels, unless there are really simple one-liner functions in statsmodels that make the code more readable. Avoid using plink or regenie - we want to build intuition for what those packages are doing under the hood.
- Where possible, streamline the coding to emphasise building intuition rather than relying on knowing special functions or having skill in python. Metaphorically, students should be given the lego pieces and asked to understand how to put them together. Make it easy for them to understand the logic behind everything.*-
  - Implemented accordingly: in the **challenge** questions the numpy/pandas plumbing is provided (helpers like `recode`, `r2_with`, `recombination_frequency`, `qq_coords`), so each blank is a *genetic/conceptual* choice (the recoding table, which traits are X-linked, what a recombinant is, which consequences are loss-of-function, how to bin by MAF). The morning's main from-scratch exercises (missingness/MAF/`run_gwas`/QQ/λ_GC/Manhattan) and the afternoon core methods (PCA via SVD, the GRM, the Wakefield Bayes factor) keep their under-the-hood implementations.
- Student profile: Assume students have at least an introductory course-level of statistics background. They should be familiar with standard Python packages (or be able to use AI tools to help them). These are early career researchers, likely graduate students and postdocs.

SCHEDULE

Morning: 0900-1200 (3hrs)  
2 x (45min lecture \+ 45 min practical)

Afternoon: 1300-1700 (4hrs)  
2 x (1h lecture \+ 1h practical)

OUTLINE OF MATERIAL

MORNING (Nik)

**Session 1: Intro to association testing**

- History of association testing; what has enabled modern-day GWAS?  
  - Pre-GWAS:   
    - Linkage analysis  
    - LOD scores  
  - Accurate, scalable genotyping:  
    - Genotype array  
    - Imputation  
  - Biobanks (100k+: UKB, FinnGen, AoU, BBJ, deCODE, MVP, CKB, TPMI, MCPS, PRECISE…)  
  - Consortia (WTCCC, GIANT, GBMI,...)  
  - Accurate/deep phenotyping (Standardization, e.g. ICD codes; integration of health records)  
  - Computation (infrastructure and software)  
- Core concept of GWAS: Regression of phenotype on genotype for single variants genome-wide  
  - Introduce some simple notation here: y \~ beta\*x \+ e  
    - What is x?  
    - What is y? (continuous \-\> linear reg; dichotomous \-\> logistic)  
    - What’s the difference between marginal (beta) and true (alpha) effect sizes? \- tease finemapping and PGS  
    - “Hypothesis free”  
- What variants to include?  
  - Standard filters: Missingness, MAF, HWE  
  - Do we have power for rare variants (Generally, no \- tease burden testing)  
    - Show ‘trumpet plots’ of variant effect size and MAF to show where we are powered  
  - Why do we care about HWE? \- excess homozygosity  
  - Differential missingness  
  - (maybe) How to deal with chrX dosage?  
- (briefly) What individuals to include?  
  - Classic: Unrelated, homogenous ancestry  
  - Ascertainment bias  
  - Case/control balance  
  - Population stratification  
  - Ancestry-specific effects  
- (briefly) What covariates to include?  
  - Age, sex, ancestry (tease PCA; mention Ancestry Components as alternative to PCs?), location/assessment centre, genotyping platform?

- Practical (implemented in `session1/`)
  - Run simple regression on a pre-generated simulated toy dataset (~50k post-QC variants on chr1, 10k samples of homogeneous ancestry; provided to students as `data/gwas_data.npz`).
    - Genotypes simulated with block-LD structure (not a 1000G matrix); MAF 0.5–50% with ~3k injected rare variants (0.1–1%) so students must do MAF QC; plus injected HWE failures and missingness tiers.
    - Continuous trait `y_cont`: low h² (≈0.035), spike-and-slab with only **3** causal variants, MAF-dependent effect sizes (no large-effect common variants), plus one injected **dominant** and one **recessive** locus. Flip effect in one causal SNP between sexes, to simulate sex-specific effect.
    - Dichotomous trait `y_bin`: liability-threshold model on `y_cont`, ~10% prevalence (90th percentile).
    - Third trait `y_poly`: fully polygenic, very low h² (≈0.02), **uncorrelated** with `y_cont` (used in Session 2 null-QQ contrast).
    - Covariates: UKB-like age distribution (decile-interpolated, ~37–73) and sex. **Age also drives a genome-wide allele-frequency gradient** (mild age-correlated stratification), so an unadjusted GWAS is genomically inflated — adjusting for age corrects it.
  - Main exercises: plot phenotype/covariate distributions; missingness/MAF/HWE QC (a HWE mid-p test is provided); run linear & logistic GWAS with/without covariates (**λ_GC ≈ 1.10 unadjusted → ≈ 1.0 once age/sex are included**, showing covariates control stratification); interpret effect sizes.
  - Challenge questions (auto-numbered, time-tagged; order = study design → interpretation → capstone):
    1. **Sex-stratified GWAS** (~10 min): run GWAS separately in males and females; find the sign-flipped sex-specific locus.
    2. **Ascertainment by age of onset** (~10 min): turn all `y_bin` cases younger than 60 into controls and watch the hits change.
    3. **Manual LocusZoom** (~12 min): r²-coloured regional plot on the simulated cohort (where individual genotypes allow LD colouring).
    4. **Drosophila linkage analysis** (~25 min): as Sturtevant (1913) — determine which traits are X-linked (sex-frequency difference), then order 6 genes by recombination frequency. Fly columns are deliberately renamed so the linkage isn't given away; `data/fly_data.csv`. Provide ground-truth cM, the fiddly helpers (`unravel_index`, `fill_diagonal`), the ends-of-sequence hint, Haldane's map function, and a provided function to plot the **estimated vs true gene order**.
    - *(The PGS / genetic-component challenge was dropped — the low-h², 3-causal data is too sparse for an instructive prediction exercise. The HWE chi-squared-vs-mid-p challenge was also removed; HWE QC stays in the main exercises. The dominant/recessive-encoding challenge was dropped as well — the data still contains the two injected non-additive loci, just no longer surfaced as an exercise.)*
    

**Session 2: Interpreting GWAS**

- Manhattan plots  
  - Why are there towers? (LD \- tease finemapping)  
- Significance and multiple testing  
  - Bonferroni  
    - Why is 5e-8 used as a threshold?   
    - Assumes independence of tests. More stringent than FDR.  
    - Family-wise error rate vs. FDR.   
  - Mention winner’s curse? (subsequent studies of an observed GWAS hit are more likely to have lower effect size)  
- What is GWAS good for?  
  - Relatively inexpensive way to get power from large sample size  
  - By definition, good for looking at genome-wide signal  
  - Common variant signal  
  - Simple model \- (relatively) easy to run and interpret  
- What is GWAS not good for?  
  - Difficult to link non-coding variant to actual gene  
  - Rare variants  
- QQ plots  
  - Core principle: Obs vs. exp  
  - What does inflation mean? What does it look like?  
  - What does an underpowered QQ plot look like?  
  - QQ vs. PP plots  
  - Lambda GC  
    - When is lift off due to pop strat vs. polygenicity vs. model miscalibration?  
    - How do we test for model miscalibration? e.g. simulated null trait, synonymous variants…  
- Web resources for GWAS  
  - GWAS Catalog: [https://www.ebi.ac.uk/gwas/](https://www.ebi.ac.uk/gwas/)   
  - ClinVar: [https://www.ncbi.nlm.nih.gov/clinvar/](https://www.ncbi.nlm.nih.gov/clinvar/)   
  - Open Targets: [https://platform.opentargets.org/](https://platform.opentargets.org/)   
  - gnomAD: [https://gnomad.broadinstitute.org/](https://gnomad.broadinstitute.org/)   
  - Pan-UKB/Neale lab sumstats: [https://pan.ukbb.broadinstitute.org/downloads](https://pan.ukbb.broadinstitute.org/downloads)   
  - Genebass  
  - All by All  
- Future of GWAS  
  - (As high-throughput sequencing becomes more popular…) Is GWAS still relevant?  
  - How will GWAS improve/change in the near-/long-term future?

- Practical (implemented in `session2/`; hybrid real + simulated data)
  - Use **real genome-wide Pan-UKB summary statistics** (EUR) for three traits, bundled pre-thinned in `data/sumstats_real.npz` (generated by `fetch_sumstats.py`): **LDL** (continuous, field 30780), **CAD / I25 chronic ischaemic heart disease** (binary, ICD10), **BMI** (highly polygenic, field 21001). Thinning keeps every genome-wide-significant SNP (for Manhattan towers) plus an unbiased ~130k random subset per trait (for QQ / lambda_GC); p-values stored as −log10(p) to avoid underflow.
  - Also bundle **Genebass whole-exome single-variant** results for BMI in `data/genebass_bmi_exomes.csv` (Variant ID in GRCh38, plus `CSQ`, `Beta`, `P-Value`), for the Miami-plot challenge.
  - Make Manhattan plots for all three traits, genome-wide across the 22 autosomes (exclude p>1e-2 for plotting). Real data → polygenic architecture, no single dominant SNP.
  - Make QQ plots + lambda_GC on the unbiased subset; **x-axis truncated to the max expected −log10(p)**. Discuss inflation from true polygenicity vs stratification. A separate **null-QQ contrast** uses the shuffled simulated phenotype (λ_GC ≈ 1).
  - Pleiotropy: signed-beta scatter at shared SNPs for **LDL↔CAD** (causally linked → tilted cloud).
  - The setup also re-runs the Session 1 **simulated** GWAS, used for the exercises that need individual-level genotypes (below).
  - Challenge questions (auto-numbered, time-tagged):
    1. **QQ with 95% CI band** (~10 min): Beta(k, n−k+1) confidence band; drawn for a **near-null** trait (shuffled simulated phenotype — stays in the band) *and* real BMI (escapes the band → genuine polygenic signal).
    2. **MAF-stratified QQ** (~8 min): stratify the real BMI variants into MAF bins and overlay QQ plots.
    3. **Trumpet plot** (~12 min): signed beta vs MAF (log-x) from real LDL, with ±power curves for N=10k/100k/400k.
    4. **Winner's curse** (~15 min): resample a **5k discovery** cohort until 3 replicates each have a hit, validate those hits in a **disjoint 5k** sample, scatter discovery vs validation effect sizes (regression to the mean).
    5. **GWAS × exome Miami plot** (~12 min): single-chromosome Miami (default chr16) of the Pan-UKB common-variant GWAS (up) vs the Genebass exome single-variant results (down) for BMI — noting the **GRCh37 vs GRCh38** build mismatch (no liftover) — then compare effect sizes of significant exome variants by **consequence** (synonymous < missense < pLoF).
    6. **Bonferroni vs Šidák** (~8 min): solve per-test α from the family-wise rate for C tests, Šidák `1−(1−α)^(1/C)` vs Bonferroni `α/C`, and plot their **ratio across a range of C** (largest at small C, → 1 as C grows). (Abdi 2007.)
  - Other ideas: Show false positive rates as a function of number of tests for independent tests, and test with varying levels of correlation. Also show Bonferroni threshold relative to that. Let students test different levels of correlation to see that higher correlation makes the threshold more conservative.
  - *(The Manual LocusZoom plot moved to Session 1, where individual genotypes allow r² colouring. The GWAS-Catalog REST-API challenge was retired in favour of the Genebass exome challenge above.)*

AFTERNOON (Pier)

**Session 3: ‘Complexities’ of GWAS** (Population stratification, relatedness, computational and statistical aspects)

- Population stratification / structure  
  - What does population stratification look like in a GWAS? (inflated p-values, lambda GC \[Nik will also mention, but worth mentioning again?\], tease LDSC intercept?)  
- Relatedness  
  - Genomic control, PCA, mixed models  
- Computational and statistical aspects  
  - Power vs speed → spike and slab prior, variational inference  
- Practical (implemented in `session3/`)
  - **Data** (`data/pca_data.npz`, via `generate_pca_data.py`): an **"All of Us"-style** diverse cohort (~2,000 participants drawn as admixtures over 5 continental components on a population tree, including African-American and Hispanic/Latino clines) **plus a "1000 Genomes"-style reference panel** (75 unadmixed individuals per superpopulation AFR/AMR/EAS/EUR/SAS, with known labels). ~20k independent SNPs; a few sibling pairs hidden in the cohort; `y_strat` = ancestry-confounded phenotype (no genetics); `y_clean` = a few true causal variants; `true_anc` = each cohort member's real ancestry proportions (answer key).
  - **Part 1** — *See* stratification: GWAS of `y_strat` with no covariates → inflated QQ / λ_GC ≈ 11 although nothing is causal.
  - **Part 2** — **PCA from scratch** (SVD of standardised genotypes) on reference + cohort together; colour the reference by superpopulation, cohort in grey → the continuous "All of Us" picture anchored by reference clusters (AFR most dispersed).
  - **Part 3** — **Assign continental ancestry**: train a **random-forest classifier** on the reference PCs, predict each cohort member's superpopulation with a **confidence threshold** (admixed individuals fall out as "Unassigned"); re-plot the cohort coloured by assigned ancestry.
  - **Part 4** — Correct stratification: re-run `y_strat` with top PCs as covariates → λ_GC → 1; `y_clean`'s true hits (3/3) survive.
  - Challenges: (1, ~12 min) confirm the "Unassigned" individuals are the most admixed (via `true_anc`) and sweep the threshold (coverage vs confidence); (2, ~12 min) build the GRM = ZZᵀ/M and find the hidden sibling pairs (≈0.5) → motivates mixed models.
  - *(A variational-inference / spike-and-slab toy remains a possible extension; not yet implemented.)*

**Session 4: Finemapping**

- LD  
- Functional  
- Define PIP  
- Mention current methods (e.g. FINEMAP, SuSiE)  
- Computational aspects?  
- Practical (implemented in `session4/`)
  - **Data** (`data/finemap_data.npz`, via `generate_finemap_data.py`): two simulated GWAS loci (~400 variants, N=5,000) with realistic LD. Locus A = 1 causal variant inside a tight cluster of near-perfect-LD tags (the marginal lead is usually a tag, not the causal); Locus B = 2 causal variants.
  - **Part 1** — Regional GWAS + **LocusZoom** coloured by r² with the lead → a whole peak is significant; the lead is often just a tag.
  - **Part 2** — **"Poor man's" finemapping** (conditional analysis): add the lead as a covariate and re-run, repeating until nothing is significant — **re-drawing the LocusZoom after each round** so the peak is seen to collapse (1 independent signal at Locus A).
  - **Part 3** — **PIP & 95% credible set** via Wakefield approximate Bayes factor from z = β̂/se → the significant peak collapses to a ~6-variant credible set containing the causal.
  - Challenges: (1, ~10 min) PIP LocusZoom (height = PIP); (2, ~10 min) credible-set size vs sample size (N/2, N/4); (3, ~12 min) the 2-causal Locus B, where a single-causal credible set fails and conditioning recovers the second signal (the SuSiE idea).

\----------------------

OLD MATERIAL

Introduction: History of association testing \- what has enabled GWAS?

- Pre-GWAS:   
  - Chromosome walking and jumping  
  - Linkage analysis  
  - LOD scores  
- Accurate, scalable genotyping:  
  - Genotype array  
  - Imputation  
- Biobanks (100k+: UKB, FinnGen, AoU, BBJ, deCODE, MVP, CKB, TPMI, MCPS, PRECISE…)  
- Consortia (WTCCC, GIANT, GBMI,...)  
- Accurate/deep phenotyping (Standardization, e.g. ICD codes; integration of health records)  
- Computation (infrastructure and software)

What is GWAS?

- Input and output: Phenotype and genotype in, variant-level association out  
- Core concept:   
  - Regression of \`phenotype \~ genotype\` for single variants across the genome  
  - Typically additive model (can reference Duncan’s Science paper as a counterexample)  
  - Introduce some simple notation here: y \~ beta\*x \+ e  
    - What is x?  
    - What is y? (continuous \-\> linear reg; dichotomous \-\> logistic)  
    - What’s the difference between marginal (beta) and true (alpha) effect sizes? \- tease finemapping and PGS  
- What variants to include?  
  - Do we have power for rare variants (Generally, no \- tease burden testing)  
    - Show ‘trumpet plots’ of variant effect size and MAF to show where we are powered  
  - Why do we care about HWE? \- excess homozygosity  
  - Differential missingness  
  - (maybe) How to deal with chrX dosage?  
- What individuals to include?  
  - Ascertainment bias  
  - Case/control balance  
  - Population stratification  
  - Ancestry-specific effects  
- What covariates to include?  
  - Age, sex, ancestry (PCA), location/assessment centre, genotyping platform?  
- (maybe briefly) Mention other flavours of GWAS  
  - Age of onset  
  - eQTLs  
  - vQTLS  
- (maybe) How would you design a GWAS study for a trait of interest?  
  - Examples of unusual traits:  
    - Super-rare late-onset disease (must have sufficient cases; must also ascertain solely for individuals past typical age of diagnosis)  
    - Likelihood of participating in a study (unless you force participation, this study will be affected by ascertainment bias)  
- Practical:  
  - Run PCA from scratch?  
  - Run simple regression on a simulated toy dataset (10k variants, 10k samples?)  
    - Continuous trait  
    - Dichotomous trait  
  - Challenge questions:  
    - How would you test for dominant or recessive effects? (A: Encode genotype differently)

Interpreting GWAS

- Manhattan plots  
  - Why are there towers? (LD \- tease finemapping)  
  - Sense check: What do traits look like under the lens of GWAS?  
    - Polygenic (many peaks) vs. oligogenic (few peaks) vs. monogenic (mabye one peak, if signal is in common variants)  
    - Dichotomous vs. continuous phenotypes  
  - Other version: Trumpet plot, Miami plot, Fuji plot  
- Significance and multiple testing  
  - Bonferroni  
    - Why is 5e-8 used as a threshold?   
    - Assumes independence of tests. More stringent than FDR.  
    - Family-wise error rate vs. FDR.   
  - Mention winner’s curse? (subsequent studies of an observed GWAS hit are more likely to have lower effect size)  
  - (maybe discuss) Why not use FDR correction/Benjamini-Hochberg?  
- What is GWAS good for?  
  - Relatively inexpensive way to get power from large sample size  
  - By definition, good for looking at genome-wide signal  
  - Common variant signal  
  - Simple model \- (relatively) easy to run and interpret  
- What is GWAS not good for?  
  - Difficult to link non-coding variant to actual gene  
  - Rare variants  
- QQ plots  
  - What does inflation mean? What does it look like?  
  - What does an underpowered QQ plot look like?  
  - Lambda GC  
- Practical:  
  - Make Manhattan plot (for multiple traits?)  
  - Make QQ plot (challenge: include 95% CI?)  
    - Stratify by MAF?  
  - Challenge questions:  
    - Make trumpet plot  
      - Make power curves \- what effect size are we powered to detect at a given MAF?  
      - (maybe \- need to refine this idea) Download real variant-level results from studies of different sample sizes. Plot them all together in a trumpet plot. How do the ‘discovered’ variants compare? Does this align with the expected power gained by higher sample size?  
    - Code an example of winner’s curse  
      - Start with a downsampled version of the data. Run GWAS. Now resample the data and rerun GWAS. How many of the significant variants decreased in effect size? Keep resampling and running GWAS. For each significant variant, show the distribution of effect sizes of the empirical null distribution of effect sizes and the original observed effect size.  
    - Use marginal betas to calculate expected genetic component of phenotype (teasing PGS)  
      - How would you refine these betas to better approximate the true causal effects? (pruning and thresholding? maybe a penalized regression?)

Causal finemapping

- (background) \[Functional?\] Finemapping vs. Statistical Finemappings  
- Statistical basis for finemapping  
- What is PIP?  
- (maybe) Current methods: FINEMAP, SuSiE  
- Practical  
  - Finemap a locus  
    - Code an actual lightweight example of statistical finemapping  
    - Manual example (poor man’s finemapping)  
      - Identify the variant with lowest p-value in a significant locus. Include that variant as a covariate in the GWAS, then rerun GWAS. Continue in this fashion until there are no remaining significant variants. (Note: this certainly isn’t guaranteed the same end result as a credible set, but perhaps still instructive?)  
  - Challenge:  
    - Make LocusZoom plot

Future of GWAS

- (As high-throughput sequencing becomes more popular…) Is GWAS still relevant?  
- How will GWAS improve/change in the near-/long-term future?

Misc things to mention:

- Relevant resources:  
  - GWAS Catalog: [https://www.ebi.ac.uk/gwas/](https://www.ebi.ac.uk/gwas/)   
  - ClinVar: [https://www.ncbi.nlm.nih.gov/clinvar/](https://www.ncbi.nlm.nih.gov/clinvar/)   
  - Open Targets: [https://platform.opentargets.org/](https://platform.opentargets.org/)   
  - gnomAD: [https://gnomad.broadinstitute.org/](https://gnomad.broadinstitute.org/)   
  - Pan-UKB/Neale lab sumstats: [https://pan.ukbb.broadinstitute.org/downloads](https://pan.ukbb.broadinstitute.org/downloads)   
  - Tease for later sessions:  
    - Genebass  
    - All by All  
- 

~~SCHEDULE \- OPTION 2~~

~~Morning: 0900-1200 (3hrs)~~  
~~2 x (45min lecture \+ 45 min practical)~~

~~Afternoon: 1300-1700 (4hrs)~~  
~~1300-1400 Group work (1h)~~  
~~Lit review~~

- ~~Find the most cited GWAS papers and explain their strengths and weaknesses. Perhaps assign groups of students to specific phenotype groups (e.g. height, BMI, psychiatric disorders, gastrointestinal disorders, cardiovascular disorders) and show the progression in sample size, loci identified, etc.~~ 

  ~~1400-1700~~ 

  ~~2 x (45min lecture \+ 45 min practical)~~

~~SCHEDULE \- OPTION 3~~

~~Morning: 0900-1200 (3hrs)~~  
~~0900-1030 Lectures (1.5h)~~  
~~2 x 45 min lectures~~  
~~1030-1200 Practical (1.5h)~~

~~Afternoon: 1300-1700 (4hrs)~~  
~~1300-1400 Group work (1h)~~  
~~Lit review (see “SCHEDULE \- OPTION 2”)~~  
~~1400-1530 Lectures (1.5h)~~  
~~2 x 45 min lectures~~  
~~1530-1700 Practical (1.5h)~~