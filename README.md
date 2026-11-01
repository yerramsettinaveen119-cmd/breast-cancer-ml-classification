# Breast Tumor Classification: Five-Model Streamlit App

An end-to-end binary classification project for the Machine Learning Assignment 2. It trains five required classifiers on the Breast Cancer Wisconsin Diagnostic dataset, saves complete preprocessing/model artifacts, and evaluates them through an interactive Streamlit interface without retraining.

> **Academic and safety note:** This project is an educational demonstration, not a medical device. Its predictions must not be used for diagnosis or treatment.

## 1. Problem statement

The objective is to classify a breast mass as **benign (0)** or **malignant (1)** from numerical features extracted from digitized fine-needle aspirate images. Five supervised classifiers are trained on one fixed, stratified split and compared using Accuracy, AUC, Precision, Recall, F1 Score, and Matthews Correlation Coefficient (MCC).

## 2. Dataset description

- Dataset: Breast Cancer Wisconsin (Diagnostic)
- Source: UCI dataset distributed with scikit-learn
- Instances: 569
- Predictive features: 30 continuous numerical features
- Target: binary diagnosis
- Missing values: none in the packaged dataset
- Positive class used here: `1 = Malignant`; `0 = Benign`
- Split: 80% training and 20% test, stratified, `random_state=42`

The original scikit-learn target uses 0 for malignant and 1 for benign. The training script intentionally inverts it so that malignant is the positive class for precision, recall, F1, and AUC.

## 3. Repository and live application links

- GitHub repository: `https://github.com/yerramsettinaveen119-cmd/breast-cancer-ml-classification`
- Live Streamlit app: `https://2026aj05698-breast-cancer-classification.streamlit.app/`

Replace both placeholders after publishing.

## 4. Models used

1. Logistic Regression with standardization
2. Decision Tree Classifier
3. K-Nearest Neighbors with standardization
4. Gaussian Naive Bayes
5. Support Vector Machine with RBF kernel, standardization, and probability estimates

Scaling is placed inside scikit-learn pipelines so it is fitted only on training data and is applied consistently to uploaded test data.

## 5. Model comparison

The following values were generated from the included scripts and the bundled 20% held-out test set.

| ML Model | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.9649 | 0.9960 | 0.9750 | 0.9286 | 0.9512 | 0.9245 |
| Decision Tree | 0.9035 | 0.9471 | 0.9429 | 0.7857 | 0.8571 | 0.7926 |
| K-Nearest Neighbors | 0.9561 | 0.9835 | 0.9744 | 0.9048 | 0.9383 | 0.9058 |
| Gaussian Naive Bayes | 0.9386 | 0.9934 | 1.0000 | 0.8333 | 0.9091 | 0.8715 |
| Support Vector Machine | 0.9825 | 0.9947 | 1.0000 | 0.9524 | 0.9756 | 0.9626 |

## 6. Observations

- **Logistic Regression:** Provides a strong linear baseline after standardization. Its balance of recall and precision shows that many diagnostic features are effectively separable in scaled feature space.
- **Decision Tree:** Captures nonlinear rules and is interpretable, but its performance is more sensitive to the selected depth and leaf-size constraints. It can overfit more easily than the regularized or neighborhood-based alternatives.
- **K-Nearest Neighbors:** Benefits substantially from standardization because it relies on distances. Distance weighting lets closer training samples contribute more strongly than farther neighbors.
- **Gaussian Naive Bayes:** Trains quickly and offers a useful probabilistic baseline. Its conditional-independence assumption is imperfect because radius, perimeter, area, and related measurements are correlated.
- **Support Vector Machine:** The RBF kernel can model nonlinear separation while scaling prevents large-valued features from dominating. Probability estimates are enabled for consistent AUC calculation.
- **Overall winner for this fixed split:** **Support Vector Machine**, selected by the highest MCC with AUC as the secondary criterion. MCC is useful here because it incorporates all four confusion-matrix outcomes. The conclusion is specific to this split and configuration.

## 7. Project structure

```text
breast_cancer_ml_assignment/
├── app.py
├── train_models.py
├── requirements.txt
├── README.md
├── test_data.csv
├── model_comparison.csv
├── model/
│   ├── logistic_regression.joblib
│   ├── decision_tree.joblib
│   ├── knn.joblib
│   ├── naive_bayes.joblib
│   ├── support_vector_machine.joblib
│   ├── model_metadata.joblib
│   └── model_metadata.json
└── screenshots/
    └── README.txt
```

## 8. Local execution

Use Python 3.10 or newer.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python train_models.py
streamlit run app.py
```

On Windows PowerShell, activate with:

```powershell
.venv\Scripts\Activate.ps1
```

The committed model files are already trained. Running `train_models.py` regenerates them, `test_data.csv`, and `model_comparison.csv` reproducibly.

## 9. Streamlit app features

- CSV upload restricted to test data
- Bundled test-data button
- Dropdown for all five saved models
- Input-column, numeric-type, missing-value, and binary-target validation
- Accuracy, AUC, Precision, Recall, F1, and MCC cards
- Confusion matrix
- ROC curve
- Classification report
- Record-level predictions and downloadable results
- No model retraining inside `app.py`

## 10. Uploaded CSV format

The CSV must contain the 30 feature columns in the metadata plus `target`. The bundled `test_data.csv` is the required example. Extra columns are ignored, but missing columns, nonnumeric values, missing values, or a target that does not contain both 0 and 1 produce a clear validation error.

## 11. Deployment on Streamlit Community Cloud

1. Create a GitHub repository and commit this project in several meaningful stages.
2. Push the files to the `main` branch.
3. Open Streamlit Community Cloud and choose **New app**.
4. Select the repository and branch.
5. Set the entry point to `app.py`.
6. Deploy and test with `test_data.csv`.
7. Insert the GitHub and live app links above.

## 12. BITS Virtual Lab evidence

Run training and the Streamlit app in the BITS Virtual Lab. Capture one genuine screenshot showing the lab environment and successful execution. Save it under `screenshots/` and include it in the final submission PDF.

## 13. Reproducibility and integrity

- Same split is used for all models.
- `stratify=y` preserves class proportions.
- Pipelines prevent preprocessing leakage.
- AUC uses probability scores rather than hard class labels.
- Models are persisted with `joblib` and loaded by the app.
- Modify UI wording, comments, observations, and repository history to reflect your own understanding and execution before submission.
