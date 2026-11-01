from pathlib import Path
import io
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score, recall_score,
    f1_score, matthews_corrcoef, confusion_matrix, classification_report,
    roc_curve
)



st.set_page_config(page_title="Breast Cancer Classification Analysis", page_icon="🔬", layout="wide")


st.markdown("""
<style>
.block-container {padding-top: 1.6rem; padding-bottom: 2rem;}
[data-testid="stMetric"] {background:#f7f9fc; border:1px solid #e5eaf2; padding:14px; border-radius:14px;}
.hero {padding:1.4rem 1.6rem; border-radius:18px; color:white;
background:linear-gradient(120deg,#12355b,#1f7a8c); margin-bottom:1rem;}
.small-note {color:#52606d; font-size:.92rem;}
</style>
""", unsafe_allow_html=True)

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "model"

@st.cache_resource
def load_assets():
    metadata = joblib.load(MODEL_DIR / "model_metadata.joblib")
    models = {
        name: joblib.load(MODEL_DIR / filename)
        for name, filename in metadata["model_files"].items()
    }
    return metadata, models


def evaluate(model, X, y):
    pred = model.predict(X)
    if hasattr(model, "predict_proba"):
        score = model.predict_proba(X)[:, 1]
    else:
        score = model.decision_function(X)
    metrics = {
        "Accuracy": accuracy_score(y, pred),
        "AUC": roc_auc_score(y, score),
        "Precision": precision_score(y, pred, zero_division=0),
        "Recall": recall_score(y, pred, zero_division=0),
        "F1": f1_score(y, pred, zero_division=0),
        "MCC": matthews_corrcoef(y, pred),
    }
    return pred, score, metrics


def read_csv_safely(uploaded_file):
    try:
        return pd.read_csv(uploaded_file)
    except UnicodeDecodeError:
        uploaded_file.seek(0)
        return pd.read_csv(uploaded_file, encoding="latin-1")

metadata, models = load_assets()
features = metadata["feature_names"]
target = metadata["target_column"]

st.markdown("""
<div class="hero">
<h1 style="margin:0">🔬 Breast Cancer Classification Analysis</h1>
<p style="margin:.45rem 0 0">Compare five saved machine-learning classifiers on uploaded test data without retraining.</p>
</div>
""", unsafe_allow_html=True)

st.caption(
    "Machine Learning Assignment 2 | "
    "Prepared by: YERRAMSETTI Naveen | "
    "Student ID: 2026AJ05698"
)

with st.sidebar:
    st.header("Model Evaluation Panel")
    selected = st.selectbox("Select a trained model", list(models.keys()))
    uploaded = st.file_uploader("Upload test CSV", type=["csv"], help="CSV must contain 30 feature columns and the target column.")
    use_sample = st.button("Evaluate Project Test Dataset", use_container_width=True)
    st.divider()
    st.caption("Target convention: 0 = Benign, 1 = Malignant. Malignant is the positive class.")

st.info("Academic demonstration only. This application is not a medical device and must not be used for diagnosis.")

if "use_sample" not in st.session_state:
    st.session_state.use_sample = False
if use_sample:
    st.session_state.use_sample = True
if uploaded is not None:
    st.session_state.use_sample = False

input_df = None
source_label = None
if uploaded is not None:
    try:
        input_df = read_csv_safely(uploaded)
        source_label = uploaded.name
    except Exception as exc:
        st.error(f"Could not read the CSV: {exc}")
elif st.session_state.use_sample:
    input_df = pd.read_csv(BASE_DIR / "test_data.csv")
    source_label = "bundled test_data.csv"

if input_df is None:
    st.subheader("How to use this app")
    st.markdown("""
1. Choose one of the five saved classifiers.
2. Upload a CSV containing the required features plus `target`, or use the bundled test set.
3. Review six evaluation metrics, the confusion matrix, ROC curve, and classification report.

The app evaluates saved models only. It never calls `fit()` and therefore does not retrain on uploaded data.
""")
    st.dataframe(pd.DataFrame({"Required feature": features}), use_container_width=True, height=320)
    st.stop()

missing = [c for c in features + [target] if c not in input_df.columns]
if missing:
    st.error("The CSV is missing required columns: " + ", ".join(missing))
    st.stop()

extra = [c for c in input_df.columns if c not in features + [target]]
if extra:
    st.warning("Extra columns will be ignored: " + ", ".join(extra))

working = input_df[features + [target]].copy()
non_numeric = [c for c in features + [target] if not pd.api.types.is_numeric_dtype(working[c])]
if non_numeric:
    st.error("These columns must be numeric: " + ", ".join(non_numeric))
    st.stop()
if working.isna().any().any():
    bad = working.columns[working.isna().any()].tolist()
    st.error("Missing values detected in: " + ", ".join(bad))
    st.stop()

classes = set(working[target].astype(int).unique())
if not classes.issubset({0, 1}) or len(classes) < 2:
    st.error("The target must contain both binary classes 0 and 1 so that all metrics, including AUC, can be calculated.")
    st.stop()

X = working[features]
y = working[target].astype(int)
model = models[selected]
try:
    pred, score, values = evaluate(model, X, y)
except Exception as exc:
    st.error(f"Evaluation failed: {exc}")
    st.stop()

st.subheader(f"Results: {selected}")
st.caption(f"Data source: {source_label} • {len(working)} rows • {len(features)} features")
metric_cols = st.columns(6)
for col, (label, value) in zip(metric_cols, values.items()):
    col.metric(label, f"{value:.4f}")

left, right = st.columns(2)
with left:
    st.markdown("#### Confusion matrix")
    cm = confusion_matrix(y, pred, labels=[0, 1])
    fig, ax = plt.subplots(figsize=(5.2, 4.2))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False,
                xticklabels=["Benign", "Malignant"],
                yticklabels=["Benign", "Malignant"], ax=ax)
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("Actual class")
    fig.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

with right:
    st.markdown("#### ROC curve")
    fpr, tpr, _ = roc_curve(y, score)
    fig, ax = plt.subplots(figsize=(5.2, 4.2))
    ax.plot(fpr, tpr, linewidth=2, label=f"{selected} (AUC={values['AUC']:.3f})")
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Chance")
    ax.set(xlabel="False positive rate", ylabel="True positive rate", xlim=(0,1), ylim=(0,1.02))
    ax.legend(loc="lower right")
    ax.grid(alpha=.2)
    fig.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

st.markdown("#### Classification report")
report = classification_report(y, pred, target_names=["Benign", "Malignant"], output_dict=True, zero_division=0)
report_df = pd.DataFrame(report).T
st.dataframe(report_df.style.format("{:.4f}"), use_container_width=True)

results_df = working.copy()
results_df["predicted_target"] = pred
results_df["predicted_label"] = np.where(pred == 1, "Malignant", "Benign")
results_df["malignant_score"] = score

with st.expander("Preview evaluated records"):
    st.dataframe(results_df, use_container_width=True, height=360)

csv_bytes = results_df.to_csv(index=False).encode("utf-8")
st.download_button("Download predictions as CSV", csv_bytes,
                   file_name=f"{selected.lower().replace(' ', '_')}_predictions.csv",
                   mime="text/csv")
