RAW DATA BY FIGURE
==================

The numbers plotted in every panel of the revised figures R1-R8, one CSV per panel.

PROVENANCE
----------
All files derive from two GEO accessions and nothing else:
  GSE165816  single-cell RNA-seq, 54 libraries (44 skin, 10 PBMC); the discovery atlas
  GSE134431  bulk RNA-seq, 21 samples; 7 healing and 6 non-healing foot ulcers
  GSE143735  bulk RNA-seq, 13 samples; 5 healing and 4 non-healing forearm ulcers

Processing, in order: per-library quality control (200-5000 genes per cell, < 10%
mitochondrial reads); normalisation to 10,000 counts per cell and log1p transform;
3,000 highly variable genes; regression of mitochondrial fraction and cell-cycle
scores; 50 principal components; Harmony integration over library; Leiden
clustering; annotation of clusters by canonical marker score. The myeloid
compartment was re-clustered on its own and scored against thirteen functional
programmes. Bulk cohorts were collapsed to gene symbols, restricted to
protein-coding genes, and analysed with PyDESeq2 on raw counts.

CONVENTIONS
-----------
  * 'group' is the clinical group of the library or bulk sample:
    Healthy, Diabetic (diabetic without ulcer), Healing, Nonhealing.
  * 'gid' is the GEO library label (G1-G50); 'gsm' the GEO sample accession.
  * Programme scores are scanpy score_genes values: the mean expression of the
    programme's genes minus the mean of an expression-matched control set, so zero
    means 'no higher than expected by chance' and the values are comparable
    between programmes but are not expression units.
  * log2FoldChange is non-healing relative to healing throughout; a positive value
    means higher in non-healing ulcers.
  * P values are two-sided. 'padj' is Benjamini-Hochberg.
  * Where both exist, patient-level statistics are the ones to quote: cell-level
    values are provided only to show the size of the pseudoreplication artefact.
  * Random seed 123 throughout.

FILES
-----
FigureR10a_interaction_counts.csv
    Number of significant ligand-receptor interactions from each sender population (rows) to each receiver population (columns), at FDR < 0.05 against a label-permutation null.
    columns: sender, B/Plasma cell, Dendritic cell, Endothelial, Erythrocyte, Fibroblast, Keratinocyte, Lymphatic EC, Macrophage: inflammatory, Macrophage: lipid-assoc., Macrophage: resident, Mast cell, Melanocyte, NK cell ...

FigureR10d_macrophage_contrast_patient_level.csv
    Macrophage-centred ligand-receptor interactions scored separately in each library and compared between healing groups. direction distinguishes signals the macrophage receives (incoming) from those it sends (outgoing).
    columns: pair, direction, n_nonhealing, n_healing, mean_nonhealing, mean_healing, log2_ratio, rank_biserial, p, padj

FigureR11b_pathway_enrichment.csv
    Enrichr over-representation results for genes nominally higher in healing and in non-healing ulcers, across GO Biological Process and KEGG.
    columns: Gene_set, Term, Overlap, P-value, Adjusted P-value, Old P-value, Old Adjusted P-value, Odds Ratio, Combined Score, Genes, direction

FigureR11d_representative_genes.csv
    Bulk expression of representative myeloid-activation, resident/complement and lipid-metabolism genes, with the log2 fold change between healing groups.
    columns: panel, gene, mean_nonhealing, mean_healing, log2FC, p

FigureR12a_bulk_module_trait.csv
    Correlation of each bulk co-expression module eigengene with non-healing status, with module size and unadjusted P.
    columns: module, n_genes, pearson_r, p, padj

FigureR12b_sc_module_trait.csv
    The same for modules built from single-cell macrophage pseudobulk.
    columns: module, n_genes, pearson_r, p, padj

FigureR12c_module_overlap.csv
    Every bulk-module by single-cell-module pair, with the size of each module, their overlap, Jaccard index, and a hypergeometric P value against chance overlap.
    columns: bulk_module, sc_module, n_bulk, n_sc, overlap, jaccard, p, padj

FigureR12d_consensus_panel_annotated.csv
    The 129-gene consensus panel with, for each gene, the number of original analyses supporting it and what the full-scale re-analysis now shows: pseudobulk and bulk effect sizes and P values, trajectory association, network module membership, and whether the gene was recovered by any full-scale analysis.
    columns: gene, P3_sc_macroDEG, P4_trajectory, P5_LRpairs, P6_bulkDEG, P7_ML_top30, P8_hubs, n_analyses, sc_pseudobulk_p, sc_pseudobulk_log2FC, bulk_p, bulk_log2FC, trajectory_padj, in_bulk_module ...

FigureR13a_bulk_ml_performance.csv
    Apparent (resubstitution), 5-fold cross-validated and leave-one-out AUC for every model and feature set on the bulk cohort. Any feature selection is performed inside each training fold.
    columns: feature_set, model, n_features, apparent_auc, cv5_auc, leaveoneout_auc

FigureR13c_external_transfer.csv
    Internal leave-one-out AUC on GSE134431 against external AUC on GSE143735 for a classifier trained on GSE134431 and applied without retraining. The gap for the consensus panel is the size of the circularity introduced by having derived that panel partly from the internal cohort.
    columns: feature_set, n_shared_genes, internal_loo_auc, external_auc

FigureR13d_bulk_feature_importance.csv
    Random-forest importance for each consensus-panel gene, with its L1 coefficient and whether L1 retained it, so the sparsity of the penalised fit is visible.
    columns: gene, rf_importance, l1_coefficient, importance, in_consensus_panel, in_canonical_set, l1_nonzero

FigureR14a_prioritised_by_state.csv
    Mean expression of the prioritised genes in each macrophage state.
    columns: state, S100A9, HIF1A, GK, COL3A1, COL1A1, CD99, APOD, EGFL7, TIMP1, AREG, SRGN, TGFB1

FigureR14b_prioritised_per_patient.csv
    Mean expression of each prioritised gene within each library, the unit used for the group comparison.
    columns: gid, group, S100A9, HIF1A, GK, COL3A1, COL1A1, CD99, APOD, EGFL7, TIMP1, AREG, SRGN, TGFB1

FigureR14c_prioritised_gene_tests.csv
    For each prioritised gene, the healing comparison performed at cell level and at patient level, with Benjamini-Hochberg adjustment of the patient-level values.
    columns: gene, mean_nonhealing, mean_healing, log2FC, p_patient_level, p_cell_level, padj_patient_level

FigureR15_perturbation_summary.csv
    In-silico perturbation of the macrophage classifier. ko_mean/oe_mean, mean change in predicted non-healing probability on knockout and over-expression across 25 classifiers trained on different patient-grouped splits; *_sd, spread across those models; *_sign_consistency, fraction of models agreeing on the direction; stable, whether both exceed 0.80. Only stable genes are ranked.
    columns: gene, ko_mean, ko_sd, oe_mean, oe_sd, n, ko_sign_consistency, oe_sign_consistency, stable, oe_p, oe_padj, direction

FigureR16a_inferred_sex.csv
    XIST and mean Y-chromosome gene expression for every library, summed over all its cells, with the inferred sex and the clinical group.
    columns: gid, XIST, Y_genes, inferred_sex, group

FigureR16c_sexlinked_genes.csv
    Effect size and P value of each sex-linked gene in the macrophage pseudobulk contrast between healing groups.
    columns: gene, log2FoldChange, pvalue, padj

FigureR16d_technical_covariates.csv
    Cell number, total counts and genes detected per library, compared between healing groups to confirm that technical depth is not confounded.
    columns: covariate, median_nonhealing, median_healing, p

FigureR1a_persample_QC.csv
    One row per skin library. gid/gsm, GEO identifiers; tissue and clinical group; n_cells_raw and n_cells_pass, cells before and after quality control; median nFeature/nCount/pctMT, per-library medians of genes, transcripts and mitochondrial percentage per cell.
    columns: gsm, gid, tissue, condition, group, n_cells_raw, n_cells_pass, n_genes_detected, median_nFeature, median_nCount, median_pctMT, secs

FigureR1b_QC_threshold_sensitivity.csv
    Cells surviving each gene-count rule, summed over the 44 skin libraries. rule, the minimum-maximum genes per cell (mitochondrial content always < 10%); n_pass, surviving cells; pct, percentage relative to the published 200-5000 rule.
    columns: rule, n_pass, pct

FigureR1c_myeloid_recovery.csv
    Myeloid representation in the published sub-sampled atlas and the full atlas. atlas_cells, cells analysed; myeloid_cells, cells annotated myeloid; pct, their percentage.
    columns: analysis, atlas_cells, myeloid_cells, pct

FigureR2a_celltype_counts.csv
    Cells assigned to each lineage in the full atlas, from Leiden clusters scored against canonical marker sets.
    columns: cell_type, n_cells

FigureR2c_composition_by_group.csv
    Percentage of cells of each lineage within each clinical group; rows sum to 100.
    columns: group, Fibroblast, Keratinocyte, Smooth muscle, Endothelial, Myeloid, Pericyte, T cell, Erythrocyte, NK cell, B/Plasma cell, Lymphatic EC, Mast cell, Melanocyte ...

FigureR3b_myeloid_composition.csv
    Percentage of myeloid cells assigned to each myeloid population within each clinical group; rows sum to 100.
    columns: group, Inflammatory monocyte-macrophage, LC/Langerhans, Lipid-associated macrophage, Mature/migratory DC, Proliferating myeloid, Resident M2-like macrophage, cDC1, cDC2, pDC

FigureR3c_marker_mean_expression.csv
    Mean log-normalised expression of canonical myeloid markers in each myeloid population.
    columns: state, FOLR2, LYVE1, C1QA, CD163, SELENOP, S100A8, S100A9, FCN1, VCAN, TREM2, APOE, GPNMB, CD9 ...

FigureR4a_programme_by_subtype.csv
    Mean score of each of the thirteen functional programmes per myeloid population, z-scored across populations. Scores are scanpy score_genes values: mean expression of the programme genes minus that of an expression-matched control set.
    columns: myeloid_subtype, ms_Resident_tissue, ms_Lipid_associated_LAM, ms_Inflammatory_mono, ms_Classical_M1, ms_Alternative_M2, ms_Angiogenic, ms_Efferocytotic, ms_Antigen_presenting, ms_Complement, ms_Interferon_stim, ms_Matrix_remodelling, ms_Proliferating, ms_Phagocytic_lysosomal

FigureR4b_M1_M2_scores.csv
    Per-cell classical M1 and alternative M2 scores for macrophages in healing and non-healing ulcers, with resident and inflammatory scores for reference.
    columns: cell, ms_Classical_M1, ms_Alternative_M2, ms_Resident_tissue, ms_Inflammatory_mono, group

FigureR4c_patient_level_programme_scores.csv
    Programme scores averaged within each library. One row per library; this is the unit of analysis for every group comparison in the revision.
    columns: gid, group, ms_Resident_tissue, ms_Lipid_associated_LAM, ms_Inflammatory_mono, ms_Classical_M1, ms_Alternative_M2, ms_Angiogenic, ms_Efferocytotic, ms_Antigen_presenting, ms_Complement, ms_Interferon_stim, ms_Matrix_remodelling, ms_Proliferating ...

FigureR5a_patient_vs_cell_level_P.csv
    For each programme, the group comparison performed twice: p_cell_level treats cells as replicates, p_patient_level treats libraries as replicates. Also rank_biserial and Hodges-Lehmann effect sizes and Benjamini-Hochberg adjusted values. The gap between the two P columns is the pseudoreplication effect.
    columns: programme, n_nonhealing, n_healing, mean_nonhealing, mean_healing, hodges_lehmann, rank_biserial, p_patient_level, p_cell_level, n_cells_nonhealing, n_cells_healing, padj_patient_level, padj_cell_level, log10_p_inflation

FigureR5b_pseudobulk_volcano.csv
    PyDESeq2 results for macrophage pseudobulk, non-healing versus healing (n = 19 libraries). log2FoldChange, lfcSE, stat, pvalue, padj as returned by DESeq2's Wald test on library-summed counts.
    columns: gene, baseMean, log2FoldChange, lfcSE, stat, pvalue, padj, y

FigureR5c_state_composition_per_patient.csv
    Percentage of each macrophage state within each library, plus the library's clinical group. Rows sum to 100 across the state columns.
    columns: gid, Inflammatory monocyte-macrophage, Lipid-associated macrophage, Resident M2-like macrophage, group

FigureR6b_lambda_benchmark.csv
    Deconvolution accuracy against 200 synthetic mixtures of known composition, at each ridge penalty, under a matched-platform and a cross-platform benchmark. rmse/mae/pearson are over all cell types; mac_rmse and mac_pearson are for the macrophage fraction alone.
    columns: lambda, rmse, mae, pearson, mac_rmse, mac_pearson, benchmark

FigureR6c_reference_size_benchmark.csv
    The same benchmark for the full-atlas reference and for a reference down-sampled to the original 300-cells-per-library design; n_macrophages records the size of each reference.
    columns: reference, n_cells, n_macrophages, rmse, mae, pearson, mac_rmse, mac_pearson

FigureR6d_macrophage_abundance.csv
    Per-sample deconvolution fractions from both references, the reference-free macrophage marker score (mean z-score of 17 canonical macrophage genes across samples), and clinical group.
    columns: index, full::B/Plasma cell, full::Dendritic cell, full::Endothelial, full::Erythrocyte, full::Fibroblast, full::Keratinocyte, full::Lymphatic EC, full::Macrophage, full::Mast cell, full::Melanocyte, full::NK cell, full::Pericyte, full::Schwann cell ...

FigureR7a_concordance_by_evidence.csv
    Replication of discovery effects in GSE143735 by evidence level. directional_concordance_pct, genes with the same sign, with a 95% binomial confidence interval and an exact binomial test against 50%; statistically_replicated_pct, genes with the same sign AND P < 0.05 in the validation cohort.
    columns: set, n_genes, directional_concordance_pct, concordance_ci_low_pct, concordance_ci_high_pct, binomial_p_vs_chance, statistically_replicated_n, statistically_replicated_pct, spearman_rho, spearman_p

FigureR7c_variance_site_vs_healing.csv
    For each of the 4000 most variable genes, the proportion of expression variance explained (eta squared) by anatomical site and by healing status.
    columns: Symbol, eta2_site, eta2_healing

FigureR8a_auc_by_evaluation.csv
    Area under the curve for the macrophage classifier under each evaluation scheme, distinguishing classification of cell state from prediction of patient outcome.
    columns: evaluation, auc, colour

FigureR8c_repeated_grouped_cv.csv
    Per-cell area under the curve for each of 25 independent patient-grouped cross-validation splits.
    columns: seed, auc

FigureR8d_learning_curve.csv
    Per-cell area under the curve on held-out patients as a function of the number of training patients; mean and standard deviation over 15 resamples per point.
    columns: n_train_patients, mean_auc, sd_auc, n_reps

FigureR9c_pseudotime_per_patient.csv
    Mean diffusion pseudotime of the macrophages in each library, with the library's clinical group. This is the unit at which the trajectory comparison is tested.
    columns: gid, group, pseudotime

FigureR9e_trajectory_gene_patterns.csv
    Assignment of each trajectory-associated gene to one of four expression patterns (k-means on row-scaled bin means along pseudotime).
    columns: gene, pattern
