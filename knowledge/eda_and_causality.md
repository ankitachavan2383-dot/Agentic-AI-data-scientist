# EDA and causal language

## Correlation is not causation
Associations found in observational data may be driven by confounders. Use wording such as "is associated with" or "is a strong predictor of" instead of "causes", unless the data come from a randomised experiment.

## Multicollinearity
Pairs of features with absolute correlation above 0.8 carry overlapping information. Importance scores for such pairs can be split arbitrarily; interpret them as a group.

## Missing data
Missingness that is itself predictive (missing-not-at-random) is informative. Flag columns with substantial missingness and keep missing-indicator features where useful. Columns above roughly 60% missing are usually dropped.

## Outliers
IQR-flagged outliers are not necessarily errors. Prefer robust models or report sensitivity before removing them.
