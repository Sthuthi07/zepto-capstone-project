import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from joblib import dump
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix, mean_absolute_error,
    mean_squared_error, r2_score
)
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

BASE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(BASE, "output")
MODEL_DIR = os.path.join(BASE, "models")
os.makedirs(OUT, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

df = pd.read_csv(os.path.join(BASE, "titanic.csv"))

# Apply the same cleaning concept to the saved raw CSV without another raw-data load.
# The modeling pipeline itself performs train-only imputation.
for col in ["sex", "embarked"]:
    df[col] = df[col].astype("object")

target = "survived"
features = ["pclass", "sex", "age", "sibsp", "parch", "fare", "embarked"]
X = df[features]
y = df[target]

print("CLASS BALANCE")
print(y.value_counts())
print(y.value_counts(normalize=True))

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

numeric_features = ["pclass", "age", "sibsp", "parch", "fare"]
categorical_features = ["sex", "embarked"]

numeric_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])
categorical_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])
preprocessor = ColumnTransformer([
    ("num", numeric_pipe, numeric_features),
    ("cat", categorical_pipe, categorical_features)
])

models = {
    "Logistic Regression": LogisticRegression(max_iter=2000, random_state=42),
    "Decision Tree": DecisionTreeClassifier(max_depth=5, random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42)
}

results = []
fitted = {}
for name, estimator in models.items():
    pipe = Pipeline([("preprocess", preprocessor), ("model", estimator)])
    pipe.fit(X_train, y_train)
    pred = pipe.predict(X_test)
    prob = pipe.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, prob)
    results.append({
        "Model": name,
        "Accuracy": accuracy_score(y_test, pred),
        "Precision": precision_score(y_test, pred, zero_division=0),
        "Recall": recall_score(y_test, pred, zero_division=0),
        "F1": f1_score(y_test, pred, zero_division=0),
        "AUC": auc
    })
    fitted[name] = (pipe, pred, prob)
    print(f"\n{name}\nConfusion matrix:\n{confusion_matrix(y_test, pred)}")

metrics_df = pd.DataFrame(results)
metrics_df.to_csv(os.path.join(OUT, "classification_metrics.csv"), index=False)
print("\nCLASSIFICATION COMPARISON\n", metrics_df)

# ROC curve.
plt.figure(figsize=(8, 6))
for name, (pipe, pred, prob) in fitted.items():
    fpr, tpr, _ = roc_curve(y_test, prob)
    plt.plot(fpr, tpr, label=f"{name} (AUC={roc_auc_score(y_test, prob):.3f})")
plt.plot([0, 1], [0, 1], linestyle="--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curves")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUT, "roc_curves.png"))
plt.close()

# Decision tree visualization.
tree_pipe = fitted["Decision Tree"][0]
feature_names = tree_pipe.named_steps["preprocess"].get_feature_names_out()
plt.figure(figsize=(20, 10))
plot_tree(
    tree_pipe.named_steps["model"],
    feature_names=feature_names,
    class_names=["Not Survived", "Survived"],
    filled=False,
    max_depth=4,
    fontsize=7
)
plt.tight_layout()
plt.savefig(os.path.join(OUT, "decision_tree.png"))
plt.close()

# Imbalance comparison on Random Forest.
imbalance_rows = []
variants = [
    ("baseline", RandomForestClassifier(n_estimators=200, random_state=42), False),
    ("class_weight_balanced", RandomForestClassifier(
        n_estimators=200, class_weight="balanced", random_state=42), False),
]
for label, estimator, _ in variants:
    pipe = Pipeline([("preprocess", preprocessor), ("model", estimator)])
    pipe.fit(X_train, y_train)
    p = pipe.predict(X_test)
    imbalance_rows.append({
        "Strategy": label,
        "Precision": precision_score(y_test, p, zero_division=0),
        "Recall": recall_score(y_test, p, zero_division=0),
        "F1": f1_score(y_test, p, zero_division=0)
    })

smote_pipe = ImbPipeline([
    ("preprocess", preprocessor),
    ("smote", SMOTE(random_state=42)),
    ("model", RandomForestClassifier(n_estimators=200, random_state=42))
])
smote_pipe.fit(X_train, y_train)
p = smote_pipe.predict(X_test)
imbalance_rows.append({
    "Strategy": "SMOTE_train_only",
    "Precision": precision_score(y_test, p, zero_division=0),
    "Recall": recall_score(y_test, p, zero_division=0),
    "F1": f1_score(y_test, p, zero_division=0)
})
imbalance_df = pd.DataFrame(imbalance_rows)
imbalance_df.to_csv(os.path.join(OUT, "imbalance_comparison.csv"), index=False)
print("\nIMBALANCE COMPARISON\n", imbalance_df)

# RF GridSearchCV. The selected estimator itself is OOB-enabled after tuning.
rf_base = RandomForestClassifier(oob_score=True, random_state=42, bootstrap=True)
rf_pipe = Pipeline([("preprocess", preprocessor), ("model", rf_base)])
grid = GridSearchCV(
    rf_pipe,
    {
        "model__n_estimators": [100, 200],
        "model__max_depth": [None, 5, 10],
        "model__max_features": ["sqrt", "log2"]
    },
    cv=5, scoring="f1", n_jobs=-1
)
grid.fit(X_train, y_train)
best_params = grid.best_params_
best_model = grid.best_estimator_
oob = best_model.named_steps["model"].oob_score_
print("\nGRID BEST:", best_params)
print("OOB SCORE:", oob)

# Regression: fare from the other available features.
reg_features = ["survived", "pclass", "age", "sibsp", "parch", "sex", "embarked"]
Xr = df[reg_features]
yr = df["fare"]
Xr_train, Xr_test, yr_train, yr_test = train_test_split(
    Xr, yr, test_size=0.2, random_state=42
)
reg_num = ["survived", "pclass", "age", "sibsp", "parch"]
reg_cat = ["sex", "embarked"]
reg_pre = ColumnTransformer([
    ("num", Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ]), reg_num),
    ("cat", Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ]), reg_cat)
])
reg_pipe = Pipeline([
    ("preprocess", reg_pre),
    ("model", LinearRegression())
])
reg_pipe.fit(Xr_train, yr_train)
yr_pred = reg_pipe.predict(Xr_test)
mae = mean_absolute_error(yr_test, yr_pred)
rmse = np.sqrt(mean_squared_error(yr_test, yr_pred))
r2 = r2_score(yr_test, yr_pred)
n = len(yr_test)
p = reg_pipe.named_steps["preprocess"].transform(Xr_test).shape[1]
adj_r2 = 1 - (1-r2)*(n-1)/(n-p-1) if n > p+1 else np.nan
residuals = yr_test - yr_pred

plt.figure(figsize=(8, 6))
plt.scatter(yr_pred, residuals)
plt.axhline(0, linestyle="--")
plt.xlabel("Predicted Fare")
plt.ylabel("Residual")
plt.title("Regression Residual Plot")
plt.tight_layout()
plt.savefig(os.path.join(OUT, "regression_residuals.png"))
plt.close()

# Simple text-based heteroscedasticity statement using residual spread across fitted-value halves.
tmp = pd.DataFrame({"pred": yr_pred, "resid": residuals})
tmp["half"] = pd.qcut(tmp["pred"], 2, duplicates="drop")
spread = tmp.groupby("half", observed=True)["resid"].std()
hetero = bool(spread.max() > 1.5 * max(spread.min(), 1e-9))

reg_metrics = {
    "MAE": mae, "RMSE": rmse, "R2": r2, "Adjusted_R2": adj_r2,
    "Heteroscedasticity_indication": hetero
}
print("\nREGRESSION METRICS\n", reg_metrics)

# Final recommendation is metric-based and kept in the report.
best_classifier = metrics_df.sort_values(["F1", "AUC"], ascending=False).iloc[0]
recommendation = (
    f"Based on the held-out test metrics, the classifier with the highest F1 in this run "
    f"is {best_classifier['Model']} (F1={best_classifier['F1']:.3f}, "
    f"precision={best_classifier['Precision']:.3f}, recall={best_classifier['Recall']:.3f}, "
    f"AUC={best_classifier['AUC']:.3f}). Accuracy was {best_classifier['Accuracy']:.3f}. "
    f"This choice emphasizes the observed balance between precision and recall rather than "
    f"accuracy alone. The regression task is reported separately because regression metrics "
    f"are not directly comparable with classification metrics."
)

with open(os.path.join(OUT, "model_report.txt"), "w", encoding="utf-8") as f:
    f.write("Classification metrics:\n")
    f.write(metrics_df.to_string(index=False))
    f.write("\n\nImbalance comparison:\n")
    f.write(imbalance_df.to_string(index=False))
    f.write("\n\nGridSearch best parameters:\n")
    f.write(str(best_params))
    f.write(f"\nOOB score: {oob}\n")
    f.write(f"\nRegression metrics: {reg_metrics}\n")
    f.write("\n\nFinal recommendation:\n")
    f.write(recommendation)

# Save complete preprocessing + estimator pipeline.
dump(best_model, os.path.join(MODEL_DIR, "best_random_forest_pipeline.joblib"))
print("\nSaved complete pipeline.")
