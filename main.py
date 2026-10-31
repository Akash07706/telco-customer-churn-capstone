"""
Telco Customer Churn Prediction - Final Capstone
Run from this folder:
    python -m pip install -r requirements.txt
    python main.py

Uses a stratified 80:20 holdout split (random_state=42), training-only preprocessing,
and compares Logistic Regression, Decision Tree, Random Forest and Gradient Boosting.
"""
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, average_precision_score, confusion_matrix, ConfusionMatrixDisplay
import joblib

ROOT = Path(__file__).parent
DATA_PATH = ROOT / "Telco-Customer-Churn.csv"
OUT = ROOT / "visualizations"
OUT.mkdir(exist_ok=True)

df = pd.read_csv(DATA_PATH)
df["TotalCharges_num"] = pd.to_numeric(df["TotalCharges"].replace(" ", np.nan), errors="coerce")
df = df.drop(columns=["TotalCharges"])
y = (df.pop("Churn") == "Yes").astype(int)
X = df.drop(columns=["customerID"])
numeric = ["SeniorCitizen", "tenure", "MonthlyCharges", "TotalCharges_num"]
categorical = [c for c in X.columns if c not in numeric]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)
preprocessor = ColumnTransformer([
    ("numeric", Pipeline([("imputer", SimpleImputer(strategy="median")),
                          ("scaler", StandardScaler())]), numeric),
    ("categorical", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),
                              ("onehot", OneHotEncoder(handle_unknown="ignore"))]), categorical)
])
models = {
    "Logistic Regression": LogisticRegression(max_iter=2000, class_weight="balanced", random_state=42),
    "Decision Tree": DecisionTreeClassifier(max_depth=6, min_samples_leaf=20, class_weight="balanced", random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=300, min_samples_leaf=3, class_weight="balanced_subsample", random_state=42, n_jobs=-1),
    "Gradient Boosting": GradientBoostingClassifier(random_state=42)
}
rows, fitted = [], {}
for name, estimator in models.items():
    pipe = Pipeline([("preprocessor", preprocessor), ("classifier", estimator)])
    pipe.fit(X_train, y_train)
    pred = pipe.predict(X_test)
    prob = pipe.predict_proba(X_test)[:, 1]
    fitted[name] = pipe
    rows.append({
        "Model": name,
        "Accuracy": accuracy_score(y_test, pred),
        "Precision": precision_score(y_test, pred),
        "Recall": recall_score(y_test, pred),
        "F1-score": f1_score(y_test, pred),
        "ROC-AUC": roc_auc_score(y_test, prob),
        "PR-AUC": average_precision_score(y_test, prob)
    })
results = pd.DataFrame(rows)
print("\\nModel comparison (holdout test set):")
print(results.round(4).to_string(index=False))
results.to_csv(ROOT / "model_comparison.csv", index=False)

# Select Decision Tree for this demonstration because churn recall is the primary
# retention-oriented objective. It is not the highest-accuracy model.
selected = fitted["Decision Tree"]
pred = selected.predict(X_test)
cm = confusion_matrix(y_test, pred)
ConfusionMatrixDisplay(cm, display_labels=["Stayed", "Churned"]).plot(values_format="d")
plt.title("Decision Tree Confusion Matrix (Test Set)")
plt.tight_layout()
plt.savefig(OUT / "decision_tree_confusion_matrix.png", dpi=170)
plt.close()
joblib.dump(selected, ROOT / "decision_tree_churn_pipeline.joblib")
print("\\nSelected Decision Tree test metrics:")
print("Accuracy:", round(accuracy_score(y_test, pred), 4))
print("Precision:", round(precision_score(y_test, pred), 4))
print("Recall:", round(recall_score(y_test, pred), 4))
print("F1:", round(f1_score(y_test, pred), 4))
print("Confusion matrix [ [TN, FP], [FN, TP] ]:", cm.tolist())
