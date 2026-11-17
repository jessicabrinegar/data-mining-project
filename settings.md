## Data and preprocessing
- Keep adult.data for training and cross-validation; reserve adult.test for final evaluation.
- Keep our Phase 3 cleaning: normalize income labels, replace ? with Unknown, drop fnlwgt and the redundant education text column, and retain plausible extreme values.
- Use log1p for capital gain/loss. For SMOTE-NC distances, standardize age, hours, and the two logged capital features using training-fold statistics only.
- Treat education_num as categorical during SMOTE-NC so synthetic education codes remain valid; use its ordered numeric representation in the decision tree.
- One-hot encode the remaining categorical predictors, allowing unseen categories. Fit encoders/scalers only on the original training portion of each fold.
## Model inputs
- Exclude sex from classifier inputs but retain it separately for subgroup sampling and fairness evaluation.
- Keep relationship, while discussing its ability to convey sex-related information.
- Use the same predictors in all four conditions. Any feature-removal experiment would be a separate, optional comparison.
Cross-validation and decision tree
- Use five shuffled folds, stratified by the four sex–income combinations, with random_state=42.
- Save the fold assignments once and reuse them across all conditions.
- Proposed tree: criterion="gini", max_depth=8, min_samples_leaf=20, class_weight=None, and random_state=42.
- Treat these as fixed experimental settings, not claimed optimal settings. This keeps the comparison focused on resampling.
## Four training conditions
- Baseline: no resampling.
- Random oversampling and SMOTE-NC: bring each sex–income subgroup up to the largest subgroup’s training-fold count.
- Random undersampling: bring each subgroup down to the smallest count.
- Use random_state=42 for samplers and k_neighbors=5 for SMOTE-NC.
- Resample training rows only. Keep validation/test distributions unchanged.
- Save before/after subgroup counts and numbers added/discarded. Check synthetic categories, education codes, and numerical ranges. Remove the temporary subgroup label before classifier fitting.
## Final test evaluation
- After freezing settings, refit each condition on all of adult.data and evaluate on the supplied test file.
- Keep matching records in the primary evaluation. Also evaluate the same predictions after excluding test rows exactly matching training rows across all original fields, including fnlwgt, after label/whitespace normalization.
- Use the same exclusion mask for every condition. Report whether conclusions change, and disclose that Phase 3 EDA examined both files.
## Shared output format
- Save predictions as CSV with:
  stage, condition, fold, row_id, sex, y_true, y_pred
- Use stage=cv/test, conditions baseline/ros/smotenc/rus, and labels 0=<=50K, 1=>50K. Leave fold blank for test predictions.
- Preserve stable original row IDs. Each condition should have predictions for exactly the same evaluation rows.
- Also save sampling counts and the final configuration/package versions.
## My evaluation side
- Report accuracy, precision, recall, F1, and overall/group confusion matrices.
- Show each sex group’s favorable prediction rate and true-positive rate, plus female-minus-male gaps.
- Compare matched-fold results and their variability, explaining whether smaller gaps came from improvements or deterioration in either group.

**Before running all experiments, generate predictions from one baseline fold and check that the evaluation code can read the agreed CSV format and calculate the required metrics.**