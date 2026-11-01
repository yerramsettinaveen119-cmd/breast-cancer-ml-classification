"""Train and persist five classifiers for the breast-cancer assignment.

Run once:
    python train_models.py

Outputs are written to model/, test_data.csv, and model_comparison.csv.
"""
from pathlib import Path
import json
import joblib
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score, recall_score,
    f1_score, matthews_corrcoef
)

RANDOM_STATE = 42
BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "model"
MODEL_DIR.mkdir(exist_ok=True)


def metric_row(y_true, y_pred, y_score):
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "AUC": roc_auc_score(y_true, y_score),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "F1": f1_score(y_true, y_pred, zero_division=0),
        "MCC": matthews_corrcoef(y_true, y_pred),
    }


def main():
    dataset = load_breast_cancer(as_frame=True)
    X = dataset.data.copy()
    # sklearn's original target is 0=malignant, 1=benign. Invert it so
    # malignant is the positive class, which makes recall clinically meaningful.
    y = (dataset.target == 0).astype(int).rename("target")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=RANDOM_STATE, stratify=y
    )

    models = {
        "Logistic Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("classifier", LogisticRegression(max_iter=5000, random_state=RANDOM_STATE))
        ]),
        "Decision Tree": DecisionTreeClassifier(
            max_depth=5, min_samples_leaf=3, random_state=RANDOM_STATE
        ),
        "K-Nearest Neighbors": Pipeline([
            ("scaler", StandardScaler()),
            ("classifier", KNeighborsClassifier(n_neighbors=7, weights="distance"))
        ]),
        "Gaussian Naive Bayes": GaussianNB(),
        "Support Vector Machine": Pipeline([
            ("scaler", StandardScaler()),
            ("classifier", SVC(C=2.0, kernel="rbf", probability=True,
                               random_state=RANDOM_STATE))
        ]),
    }

    slugs = {
        "Logistic Regression": "logistic_regression",
        "Decision Tree": "decision_tree",
        "K-Nearest Neighbors": "knn",
        "Gaussian Naive Bayes": "naive_bayes",
        "Support Vector Machine": "support_vector_machine",
    }

    rows = []
    for name, model in models.items():
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        score = model.predict_proba(X_test)[:, 1]
        rows.append({"ML Model": name, **metric_row(y_test, pred, score)})
        joblib.dump(model, MODEL_DIR / f"{slugs[name]}.joblib")

    test_df = X_test.copy()
    test_df["target"] = y_test.values
    test_df.to_csv(BASE_DIR / "test_data.csv", index=False)

    results = pd.DataFrame(rows).sort_values(["MCC", "AUC"], ascending=False)
    results.to_csv(BASE_DIR / "model_comparison.csv", index=False)

    metadata = {
        "feature_names": list(X.columns),
        "target_column": "target",
        "positive_class": 1,
        "class_labels": {0: "Benign", 1: "Malignant"},
        "model_files": {name: f"{slug}.joblib" for name, slug in slugs.items()},
        "split": {"test_size": 0.20, "random_state": RANDOM_STATE, "stratified": True},
        "dataset": "Breast Cancer Wisconsin (Diagnostic), sklearn copy of UCI dataset"
    }
    joblib.dump(metadata, MODEL_DIR / "model_metadata.joblib")
    with open(MODEL_DIR / "model_metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print("Training complete. Saved models:")
    for name in models:
        print(f"  - {name}: model/{slugs[name]}.joblib")
    print("\nModel comparison:")
    print(results.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print(f"\nTest rows: {len(test_df)}; columns: {len(test_df.columns)}")


if __name__ == "__main__":
    main()
