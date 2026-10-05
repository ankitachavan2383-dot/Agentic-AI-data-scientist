# Model metrics guide

## F1-macro and accuracy
Accuracy can look high on imbalanced data even for a model that always predicts the majority class. F1-macro averages per-class F1 so minority classes count equally. Always compare against a majority-class baseline; a lift below 0.05 means the model has learned little.

## ROC-AUC
ROC-AUC measures ranking quality independent of threshold. 0.5 is random; 0.7-0.8 is acceptable; above 0.9 is strong but warrants a leakage check.

## R2, RMSE, MAE
R2 is the share of variance explained relative to predicting the mean. RMSE penalises large errors more than MAE. Report both in target units so stakeholders can judge practical error size.

## Permutation importance
Permutation importance measures the drop in held-out score when one feature is shuffled. It is model-agnostic and computed on unseen data, but correlated features can share or mask importance. It shows predictive relevance, not causation.
