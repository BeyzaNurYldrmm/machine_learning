
import numpy as np
import pandas as pd
import seaborn as sns

from sklearn.preprocessing import RobustScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix
from sklearn.model_selection import train_test_split, cross_validate


df=pd.read_csv("datasets\\diabetes.csv")

y = df["Outcome"]
X = df.drop(["Outcome"], axis=1)

log_model = LogisticRegression().fit(X, y)

cv_results = cross_validate(log_model,
                            X, y,
                            cv=5,
                            scoring=["accuracy", "precision", "recall", "f1", "roc_auc"])


print(cv_results['test_accuracy'].mean())
# Accuracy: 0.7721
print(cv_results['test_precision'].mean())
# Precision: 0.7192
print(cv_results['test_recall'].mean())
# Recall: 0.5747
print(cv_results['test_f1'].mean())
# F1-score: 0.6371
print(cv_results['test_roc_auc'].mean())
# AUC: 0.8327













