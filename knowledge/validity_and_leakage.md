# Validity and leakage

## Target leakage
If a single feature carries almost all of the importance, or the test score is near perfect, suspect leakage: a column that encodes the outcome, is recorded after the outcome, or is a transformed copy of the target. Remove it and re-evaluate before trusting the model.

## Train/test hygiene
Imputation, scaling and encoding must be fit on training data only. Fitting them on the full dataset leaks test information. Use a pipeline so cross-validation refits preprocessing in every fold.

## Small samples
With fewer than a few hundred rows, cross-validation scores vary widely. Report the standard deviation across folds and avoid fine-grained rankings of features.
