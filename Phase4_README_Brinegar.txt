PHASE 4 README - COMPARING RESAMPLING METHODS AND FAIRNESS
Data Mining 5415, Group Project Phase 4 (Group leader: Jessica Brinegar)

WHAT THIS PROJECT DOES
We train one decision tree on the UCI Adult Income data under four training
conditions: no resampling (baseline), random oversampling (ROS), SMOTE-NC and
random undersampling (RUS). The resampling balances the four sex/income
subgroups. We compare accuracy, precision/recall/F1 for the >50K class, and the
female-minus-male statistical parity and equal opportunity gaps across 5 matched
cross-validation folds and on the supplied test file.


FILES
Phase4_Codebase_Brinegar.ipynb   Primary entry point. Runs everything: Phase 3 EDA
                                 and data preparation, the Phase 4 experiments, and
                                 the Phase 4 evaluation. Saved with its outputs.
requirements.txt                 Pinned package versions.
Phase4_README_Brinegar.txt       This file.
Phase4_FinalReport_Brinegar.pdf  Final report (6 pages, two columns).
data/                            UCI Adult files: adult.data, adult.test,
                                 adult.names, old.adult.names, Index.
outputs/                         Everything the notebook writes (see OUTPUTS).


DATA SOURCE
UCI Machine Learning Repository, Adult dataset (Becker and Kohavi, 1996)
  Page: https://archive.ics.uci.edu/dataset/2/adult
  DOI:  https://doi.org/10.24432/C5XW20
The files are included in data/. If they are missing, download the dataset zip
from the page above and put adult.data, adult.test and adult.names in data/.
adult.data has 32,561 rows and adult.test has 16,281 rows (its first line is a
metadata line that the notebook skips).


SETUP
Tested with Python 3.14.2 on macOS 26.6.2 (Apple silicon, arm64).

  cd <project folder>
  python3.14 -m venv .venv
  source .venv/bin/activate            (Windows: .venv\Scripts\activate)
  pip install -r requirements.txt
  pip install notebook nbconvert ipykernel

Package versions used for the saved run:
  pandas 3.0.5, numpy 2.5.2, scikit-learn 1.9.0, imbalanced-learn 0.14.2,
  matplotlib 3.11.1, seaborn 0.13.2
  notebook 7.6.5, nbconvert 7.17.2, nbclient 0.11.0, ipykernel 7.4.0
outputs/config.json also records the Python and package versions of the run.


HOW TO RUN
Run the notebook from the project folder (all paths are relative, e.g. data/
and outputs/). Either open it in Jupyter and use "Restart Kernel and Run All
Cells", or run it from the command line:

  jupyter nbconvert --to notebook --execute --inplace Phase4_Codebase_Brinegar.ipynb

A full run takes about 30-60 seconds. It recreates every file in outputs/.


SETTINGS (fixed before the final test; not tuned)
- Target: y = 1 if income is >50K (favorable class), else 0. Test labels have
  their trailing "." removed.
- Cleaning: "?" replaced by "Unknown" in workclass, occupation, native_country;
  all rows kept; fnlwgt and the education text column dropped; capital_gain and
  capital_loss replaced with log1p versions.
- Classifier inputs: age, education_num, capital_gain_log, capital_loss_log,
  hours_per_week (as numbers) plus one-hot encoded workclass, marital_status,
  occupation, relationship, race, native_country
  (OneHotEncoder(handle_unknown="ignore"), fit on original training rows only).
  sex is NOT a classifier input; it is only used for sampling and evaluation.
- Folds: StratifiedKFold(n_splits=5, shuffle=True, random_state=42) on adult.data,
  stratified on sex|income, saved once in outputs/folds.csv and reused by all
  conditions.
- Tree: DecisionTreeClassifier(criterion="gini", max_depth=8,
  min_samples_leaf=20, class_weight=None, random_state=42).
- ROS: RandomOverSampler(sampling_strategy="not majority", random_state=42).
  SMOTE-NC: SMOTENC(k_neighbors=5, sampling_strategy="not majority",
  random_state=42), on standardized age/hours/log capital (scaler fit on the
  training rows), education_num treated as a category, then converted back.
  RUS: RandomUnderSampler(sampling_strategy="not minority", random_state=42).
  All samplers balance the four sex/income subgroups of the training rows only.
- Final test: each condition is refit on all of adult.data and predicts all of
  adult.test. A sensitivity check also drops the 23 test rows that match a
  training row on all 15 original columns (same predictions, no refitting).


OUTPUTS
Experiment outputs (Phase 4 experiment section):
  outputs/predictions_cv.csv            
    - CV predictions, 4 conditions x 32,561 rows
      (stage, condition, fold, row_id, sex, y_true, y_pred)
  outputs/predictions_test.csv          
    - Test predictions, 4 x 16,281 rows, plus matches_train (1 = matches a training row)
  outputs/predictions_cv_baseline_fold1.csv  
    - Early format check (baseline, fold 1)
  outputs/folds.csv                     
    - row_id -> fold (1-5)
  outputs/sampling_counts.csv, sampling_totals.csv, ros_copies.csv,
  outputs/smotenc_checks.csv, smotenc_category_shift.csv,
  outputs/resampling_summary.csv        
    - Sampling and SMOTE-NC diagnostics
  outputs/config.json                   
    - Settings and package versions of the run
  outputs/figures/subgroup_counts_before_after.png
Evaluation outputs (Phase 4 evaluation section):
  outputs/evaluation/cv_fold_metrics.csv     
    - Metrics, group rates and gaps for each condition and fold
  outputs/evaluation/cv_mean.csv, cv_std.csv 
    - Mean and SD of the 5 folds
  outputs/evaluation/test_metrics.csv        
    - Test results (all 16,281 rows)
  outputs/evaluation/test_metrics_without_matches.csv
    - Test results without the 23 rows that match a training row
  outputs/figures/eval_*.png
    - Evaluation figures
In the evaluation files, group rates are percents and gaps (Female minus Male)
are percentage points; accuracy, precision, recall and F1 are 0-1.


REPRODUCIBILITY NOTE
Running the notebook twice on the same computer gives identical outputs. When we
ran it on a second computer with the same package versions, the baseline, ROS
and RUS predictions were identical, but about 1% of the SMOTE-NC predictions
changed (the synthetic rows were slightly different). The SMOTE-NC conclusions
did not change, but its exact numbers can differ a little between computers.
It is not confirmed why. The saved notebook, outputs and report all come from 
the same run (Python 3.14.2, macOS arm64).
