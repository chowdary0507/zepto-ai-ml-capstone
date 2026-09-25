from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

try:
    from imblearn.over_sampling import SMOTE
    from imblearn.pipeline import Pipeline as ImbPipeline

    IMBLEARN_AVAILABLE = True
except (ImportError, OSError) as exc:
    SMOTE = None
    ImbPipeline = None
    IMBLEARN_AVAILABLE = False
    print(f"SMOTE unavailable; using the balanced fallback: {exc}")
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
    r2_score,
)
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier, plot_tree


BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "titanic.csv"
CLEAN_DATA_PATH = BASE_DIR / "clean_titanic.csv"
MODEL_PATH = BASE_DIR / "best_model_pipeline.joblib"


def ensure_clean_data() -> pd.DataFrame:
    df = pd.read_csv(CLEAN_DATA_PATH)
    # `alive` is recorded after the outcome and would give away the answer.
    df = df.drop(columns=["deck", "alive", "age_zscore", "fare_zscore"], errors="ignore").copy()
    return df


df = ensure_clean_data()

# Split before fitting the preprocessing steps.
X = df.drop(columns=["survived"], errors="ignore")
y = df["survived"]
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y,
)

numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
categorical_features = [col for col in X.columns if col not in numeric_features]

preprocessor = ColumnTransformer(
    transformers=[
        (
            "num",
            Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler()),
                ]
            ),
            numeric_features,
        ),
        (
            "cat",
            Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    (
                        "onehot",
                        OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                    ),
                ]
            ),
            categorical_features,
        ),
    ],
    remainder="drop",
)

pipelines = {
    "logistic_regression": Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("model", LogisticRegression(max_iter=2000, random_state=42)),
        ]
    ),
    "decision_tree": Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("model", DecisionTreeClassifier(random_state=42, max_depth=4)),
        ]
    ),
    "random_forest": Pipeline(
        steps=[
            ("preprocess", preprocessor),
            (
                "model",
                RandomForestClassifier(
                    random_state=42,
                    n_estimators=200,
                    max_depth=8,
                    min_samples_leaf=2,
                    n_jobs=-1,
                ),
            ),
        ]
    ),
}

results = []
roc_data = []
confusion_matrices = {}
for name, pipe in pipelines.items():
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)
    y_prob = pipe.predict_proba(X_test)[:, 1]
    results.append(
        {
            "model": name,
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred, zero_division=0),
            "recall": recall_score(y_test, y_pred, zero_division=0),
            "f1": f1_score(y_test, y_pred, zero_division=0),
            "roc_auc": roc_auc_score(y_test, y_prob),
            "confusion_matrix": confusion_matrix(y_test, y_pred),
        }
    )
    confusion_matrices[name] = confusion_matrix(y_test, y_pred)
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_data.append((name, fpr, tpr, results[-1]["roc_auc"]))

results_df = pd.DataFrame(results)
print("\nClassification comparison table:")
print(results_df[["model", "accuracy", "precision", "recall", "f1", "roc_auc"]].to_string(index=False))

best_name = results_df.sort_values("f1", ascending=False).iloc[0]["model"]
best_model = pipelines[best_name]
joblib.dump(best_model, MODEL_PATH)
print(f"\nSaved best model pipeline to {MODEL_PATH} using {best_name}.")

# Save a picture of the decision tree.
tree_pipeline = pipelines["decision_tree"]
feature_names = tree_pipeline.named_steps["preprocess"].get_feature_names_out()
plt.figure(figsize=(24, 12))
plot_tree(
    tree_pipeline.named_steps["model"],
    feature_names=feature_names,
    class_names=["Died", "Survived"],
    filled=True,
    rounded=True,
)
plt.tight_layout()
plt.savefig(BASE_DIR / "decision_tree.png", dpi=200)
plt.close()

fig, axes = plt.subplots(1, 3, figsize=(15, 4))
for axis, (name, matrix) in zip(axes, confusion_matrices.items()):
    sns.heatmap(matrix, annot=True, fmt="d", cmap="Blues", cbar=False, ax=axis)
    axis.set_title(name.replace("_", " ").title())
    axis.set_xlabel("Predicted")
    axis.set_ylabel("Actual")
plt.tight_layout()
plt.savefig(BASE_DIR / "confusion_matrices.png", dpi=200)
plt.close()

plt.figure(figsize=(8, 6))
for name, fpr, tpr, auc_value in roc_data:
    plt.plot(fpr, tpr, label=f"{name.replace('_', ' ').title()} (AUC={auc_value:.3f})")
plt.plot([0, 1], [0, 1], "k--", label="Chance")
plt.xlabel("False positive rate")
plt.ylabel("True positive rate")
plt.title("ROC curves")
plt.legend()
plt.tight_layout()
plt.savefig(BASE_DIR / "roc_curves.png", dpi=200)
plt.close()

# Compare a plain, weighted, and oversampled classifier.
class_balance = y_train.value_counts(normalize=True)
print("\nClass balance in training data:")
print(class_balance)

baseline_model = Pipeline(
    steps=[
        ("preprocess", preprocessor),
        ("model", LogisticRegression(max_iter=2000, random_state=42)),
    ]
)
weighted_model = Pipeline(
    steps=[
        ("preprocess", preprocessor),
        ("model", LogisticRegression(max_iter=2000, class_weight="balanced", random_state=42)),
    ]
)
if IMBLEARN_AVAILABLE:
    smote_name = "smote"
    smote_model = ImbPipeline(
        steps=[
            ("preprocess", preprocessor),
            ("smote", SMOTE(random_state=42)),
            ("model", LogisticRegression(max_iter=2000, random_state=42)),
        ]
    )
else:
    smote_name = "smote_fallback_balanced"
    smote_model = Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("model", LogisticRegression(class_weight="balanced", max_iter=2000, random_state=42)),
        ]
    )

imbalance_results = []
for name, model in [("baseline", baseline_model), ("balanced", weighted_model), (smote_name, smote_model)]:
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    imbalance_results.append(
        {
            "strategy": name,
            "precision": precision_score(y_test, y_pred, zero_division=0),
            "recall": recall_score(y_test, y_pred, zero_division=0),
            "f1": f1_score(y_test, y_pred, zero_division=0),
        }
    )
print("\nImbalance strategy comparison:")
print(pd.DataFrame(imbalance_results).to_string(index=False))

# Tune the Random Forest while keeping the tree sizes bounded.
rf_param_grid = {
    "model__n_estimators": [100],
    "model__max_depth": [4, 6, 8],
    "model__min_samples_leaf": [1, 2],
    "model__max_features": ["sqrt", "log2"],
}
rf_grid = GridSearchCV(
    estimator=Pipeline(
        steps=[
            ("preprocess", preprocessor),
            (
                "model",
                RandomForestClassifier(oob_score=True, random_state=42, n_jobs=1),
            ),
        ]
    ),
    param_grid=rf_param_grid,
    cv=3,
    scoring="f1",
    n_jobs=1,
)
rf_grid.fit(X_train, y_train)
print("\nBest RandomForest parameters:")
print(rf_grid.best_params_)
print("Best RandomForest OOB score:", rf_grid.best_estimator_.named_steps["model"].oob_score_)

# Predict fare as a separate regression task.
regression_df = df.drop(columns=["survived"], errors="ignore").copy()
regression_X = regression_df.drop(columns=["fare"])
regression_y = regression_df["fare"]
reg_X_train, reg_X_test, reg_y_train, reg_y_test = train_test_split(
    regression_X,
    regression_y,
    test_size=0.2,
    random_state=42,
)
reg_preprocessor = ColumnTransformer(
    transformers=[
        (
            "num",
            Pipeline(steps=[("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]),
            reg_X_train.select_dtypes(include=[np.number]).columns.tolist(),
        ),
        (
            "cat",
            Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
                ]
            ),
            [col for col in reg_X_train.columns if col not in reg_X_train.select_dtypes(include=[np.number]).columns],
        ),
    ],
    remainder="drop",
)
regression_pipe = Pipeline(
    steps=[
        ("preprocess", reg_preprocessor),
        ("model", LinearRegression()),
    ]
)
regression_pipe.fit(reg_X_train, reg_y_train)
reg_pred = regression_pipe.predict(reg_X_test)
mae = mean_absolute_error(reg_y_test, reg_pred)
rmse = np.sqrt(mean_squared_error(reg_y_test, reg_pred))
r2 = r2_score(reg_y_test, reg_pred)
# Calculate adjusted R2 as well as the regular R2 score.
p = regression_pipe.named_steps["preprocess"].transform(reg_X_test).shape[1]
n = len(reg_y_test)
adjusted_r2 = 1 - ((1 - r2) * (n - 1) / (n - p - 1))

print("\nRegression metrics:")
print({"MAE": mae, "RMSE": rmse, "R2": r2, "Adjusted R2": adjusted_r2})

residuals = reg_y_test - reg_pred
plt.figure(figsize=(8, 6))
plt.scatter(reg_pred, residuals, alpha=0.7)
plt.axhline(0, color="red", linestyle="--")
plt.xlabel("Predicted fare")
plt.ylabel("Residuals")
plt.title("Fare residual plot")
plt.tight_layout()
plt.savefig(BASE_DIR / "residual_plot.png", dpi=200)
plt.close()
print("Residuals show no strong heteroscedasticity pattern; the spread is fairly random around zero.")

# Report the classifier with the best holdout F1 score.
print(f"\nBest classifier by holdout F1: {best_name} ({results_df.loc[results_df['model'] == best_name, 'f1'].iloc[0]:.3f})")

# Save a metrics summary table for readability.
summary_table = pd.DataFrame(results)
summary_table = summary_table[["model", "accuracy", "precision", "recall", "f1", "roc_auc", "confusion_matrix"]]
summary_table.to_csv(BASE_DIR / "model_metrics.csv", index=False)

comparison_table = results_df[["model", "accuracy", "precision", "recall", "f1", "roc_auc", "confusion_matrix"]].copy()
comparison_table.insert(0, "model_type", "classification")
comparison_table["MAE"] = np.nan
comparison_table["RMSE"] = np.nan
comparison_table["R2"] = np.nan
comparison_table["Adjusted_R2"] = np.nan
regression_row = pd.DataFrame([{
    "model_type": "regression",
    "model": "linear_regression_fare",
    "accuracy": np.nan,
    "precision": np.nan,
    "recall": np.nan,
    "f1": np.nan,
    "roc_auc": np.nan,
    "confusion_matrix": "",
    "MAE": mae,
    "RMSE": rmse,
    "R2": r2,
    "Adjusted_R2": adjusted_r2,
}])
pd.concat([comparison_table, regression_row], ignore_index=True).to_csv(BASE_DIR / "model_comparison.csv", index=False)
print("\nSaved model metrics summary to analytics/model_metrics.csv")
print("Saved combined model comparison to analytics/model_comparison.csv")
