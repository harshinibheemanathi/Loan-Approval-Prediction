import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

st.set_page_config(page_title="Loan Approval Prediction", page_icon="🏦", layout="wide")

st.title("🏦 Loan Approval Prediction Using Machine Learning")
st.caption("Educational project: predictions are estimates, not actual bank decisions.")

# ---------- Navigation ----------
page = st.sidebar.radio(
    "Navigate",
    ["Home", "Predict Loan Status", "Application Details", "Model Evaluation", "Dataset & Methodology", "Project Information"]
)

# ---------- Dataset loading ----------
st.sidebar.header("Dataset")
uploaded_file = st.sidebar.file_uploader("Upload your loan CSV file", type=["csv"])

PROJECT_DIR = Path(__file__).resolve().parent
APPLICATIONS_FILE = PROJECT_DIR / "applications.csv"

def load_data():
    # Prefer a CSV stored beside app.py so it loads automatically after a restart.
    candidates = [
        PROJECT_DIR / "loan_approval_dataset.csv",
        PROJECT_DIR / "loan_data_set.csv",
        Path.cwd() / "loan_approval_dataset.csv",
        Path.cwd() / "loan_data_set.csv",
    ]
    for path in candidates:
        if path.exists():
            return pd.read_csv(path)
    # Upload is a fallback if the dataset is not stored in the project folder.
    if uploaded_file is not None:
        return pd.read_csv(uploaded_file)
    return None

def save_application(input_values, predicted_status, selected_model):
    """Save each submitted prediction so application count persists after restart."""
    record = dict(input_values)
    record["Application_ID"] = f"APP-{datetime.now().strftime('%Y%m%d%H%M%S%f')}"
    record["Prediction"] = str(predicted_status)
    record["Model_Used"] = str(selected_model)
    record["Submitted_At"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    record_df = pd.DataFrame([record])
    if APPLICATIONS_FILE.exists():
        old_records = pd.read_csv(APPLICATIONS_FILE)
        # Keep the same columns even if a dataset's feature columns change later.
        all_columns = list(dict.fromkeys(list(old_records.columns) + list(record_df.columns)))
        old_records = old_records.reindex(columns=all_columns)
        record_df = record_df.reindex(columns=all_columns)
        record_df = pd.concat([old_records, record_df], ignore_index=True)
    record_df.to_csv(APPLICATIONS_FILE, index=False)
    return record["Application_ID"]

try:
    df = load_data()
except Exception as exc:
    st.error(f"Could not read the CSV file: {exc}")
    st.stop()

if df is None:
    st.warning("Dataset not found. Use the upload control in the left sidebar to upload loan_data_set.csv.")
    st.info("Place your CSV beside app.py, or upload it using the sidebar. The target column should be Loan_Status.")
    st.stop()

# Normalize accidental spaces in column names and text cells
df.columns = [str(c).strip() for c in df.columns]
for col in df.select_dtypes(include=["object"]).columns:
    df[col] = df[col].map(lambda x: x.strip() if isinstance(x, str) else x)

target = "Loan_Status"
if target not in df.columns:
    st.error(f"Target column '{target}' was not found. Available columns: {', '.join(df.columns)}")
    st.stop()

# Drop rows with missing target; normalize target values
df = df.dropna(subset=[target]).copy()
df[target] = df[target].astype(str).str.strip()
valid_targets = df[target].str.upper().isin(["Y", "N", "YES", "NO", "1", "0", "APPROVED", "REJECTED", "APPROVE", "REJECT"])
df = df.loc[valid_targets].copy()
if df.empty:
    st.error("No recognized target labels were found in Loan_Status. Expected Y/N or Yes/No.")
    st.stop()

def normalize_label(value):
    v = str(value).strip().upper()
    if v in ("Y", "YES", "1", "APPROVED", "APPROVE"):
        return "Approved"
    if v in ("N", "NO", "0", "REJECTED", "REJECT"):
        return "Rejected"
    return str(value)

df[target] = df[target].map(normalize_label)
# Loan_ID is an identifier, not a meaningful prediction feature
feature_df = df.drop(columns=[target] + (["Loan_ID"] if "Loan_ID" in df.columns else []))
if feature_df.shape[1] == 0:
    st.error("No usable input features were found in the dataset.")
    st.stop()

X = feature_df.copy()
y = df[target].copy()
numeric_cols = X.select_dtypes(include=["number"]).columns.tolist()
categorical_cols = [c for c in X.columns if c not in numeric_cols]

numeric_pipeline = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])
categorical_pipeline = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])
preprocessor = ColumnTransformer(transformers=[
    ("num", numeric_pipeline, numeric_cols),
    ("cat", categorical_pipeline, categorical_cols)
])

model_choice = st.sidebar.selectbox("Choose model", ["Random Forest", "Logistic Regression"])
if model_choice == "Random Forest":
    classifier = RandomForestClassifier(n_estimators=200, random_state=42, class_weight="balanced")
else:
    classifier = LogisticRegression(max_iter=2000, class_weight="balanced")

# Use a stratified split when possible
stratify = y if y.value_counts().min() >= 2 else None
try:
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=stratify
    )
except ValueError:
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = Pipeline(steps=[("preprocessor", preprocessor), ("classifier", classifier)])
try:
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
except Exception as exc:
    st.error(f"Model training failed: {exc}")
    st.stop()

if page == "Home":
    st.subheader("Welcome")
    st.write(
        "This project demonstrates how supervised machine learning can analyze historical loan application "
        "records and estimate whether a new application resembles previously approved or rejected applications."
    )
    saved_count = len(pd.read_csv(APPLICATIONS_FILE)) if APPLICATIONS_FILE.exists() else 0
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Dataset rows", f"{len(df):,}")
    c2.metric("Input features", len(feature_df.columns))
    c3.metric("Test accuracy", f"{accuracy:.1%}")
    c4.metric("Applications submitted", f"{saved_count:,}")
    st.markdown("### Project workflow")
    st.markdown("**Dataset collection → Data cleaning → Feature preprocessing → Model training → Evaluation → Prediction**")
    st.info("Use the sidebar to open Predict Loan Status and enter applicant details.")

elif page == "Predict Loan Status":
    st.subheader("Enter applicant details")
    st.write("Fill in the available fields and click **Predict Loan Status**.")
    with st.form("loan_prediction_form"):
        inputs = {}
        columns = st.columns(2)
        for i, col in enumerate(feature_df.columns):
            with columns[i % 2]:
                series = feature_df[col]
                if col in numeric_cols:
                    vals = pd.to_numeric(series, errors="coerce").dropna()
                    default = float(vals.median()) if not vals.empty else 0.0
                    if col.lower() in ("credit_history",):
                        inputs[col] = st.selectbox(col.replace("_", " "), [1, 0], index=0)
                    else:
                        step = 1.0 if (vals.empty or (vals % 1 == 0).all()) else 0.1
                        inputs[col] = st.number_input(
                            col.replace("_", " "),
                            min_value=0.0 if (vals.empty or vals.min() >= 0) else float(vals.min()),
                            value=max(float(default), 0.0) if (vals.empty or vals.min() >= 0) else float(default),
                            step=step,
                            help=f"Dataset range: {vals.min():.2f} to {vals.max():.2f}" if not vals.empty else None
                        )
                else:
                    choices = [str(v) for v in series.dropna().unique().tolist()]
                    if not choices:
                        choices = ["Unknown"]
                    inputs[col] = st.selectbox(col.replace("_", " "), choices)
        submitted = st.form_submit_button("Predict Loan Status", type="primary", use_container_width=True)

    if submitted:
        input_frame = pd.DataFrame([inputs], columns=feature_df.columns)
        for col in numeric_cols:
            input_frame[col] = pd.to_numeric(input_frame[col], errors="coerce")
        try:
            prediction = model.predict(input_frame)[0]
            application_id = save_application(inputs, prediction, model_choice)
            st.divider()
            if prediction == "Approved":
                st.success("Prediction: Loan Approved")
            else:
                st.warning("Prediction: Loan Rejected")
            st.info(f"Application saved successfully. Application ID: {application_id}")
            current_count = len(pd.read_csv(APPLICATIONS_FILE)) if APPLICATIONS_FILE.exists() else 0
            st.metric("Total applications submitted", current_count)
            st.caption("This is a machine-learning estimate for academic demonstration, not a bank decision. Each click on the prediction button saves a new application record.")
            if hasattr(model.named_steps["classifier"], "predict_proba"):
                probabilities = model.predict_proba(input_frame)[0]
                class_names = model.named_steps["classifier"].classes_
                st.write("Model confidence estimates:")
                st.bar_chart(pd.DataFrame({"Estimated probability": probabilities}, index=class_names))
        except Exception as exc:
            st.error(f"Prediction failed: {exc}")

elif page == "Model Evaluation":
    st.subheader("Model evaluation")
    st.metric("Test accuracy", f"{accuracy:.1%}")
    st.write("Accuracy is the percentage of held-out test records classified correctly. It is not a guarantee of real-world performance.")
    labels = sorted(y.unique().tolist())
    st.markdown("### Classification report")
    report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    st.dataframe(pd.DataFrame(report).transpose().round(3), use_container_width=True)
    st.markdown("### Confusion matrix")
    cm = confusion_matrix(y_test, y_pred, labels=labels)
    st.dataframe(pd.DataFrame(cm, index=[f"Actual {x}" for x in labels], columns=[f"Predicted {x}" for x in labels]), use_container_width=True)

elif page == "Dataset & Methodology":
    st.subheader("Dataset overview")
    c1, c2 = st.columns(2)
    c1.metric("Rows", len(df))
    c2.metric("Columns including target", len(df.columns))
    st.write("Target variable: `Loan_Status` (Y/Yes = Approved; N/No = Rejected).")
    st.markdown("### Dataset preview")
    st.dataframe(df.head(20), use_container_width=True)
    st.markdown("### Missing values")
    missing = df.isna().sum().rename("Missing values").to_frame()
    st.dataframe(missing, use_container_width=True)
    st.markdown("### Methodology")
    st.markdown("""
1. **Data collection:** load historical loan application records.
2. **Data preprocessing:** handle missing numeric and categorical values.
3. **Feature preparation:** exclude the identifier `Loan_ID`, encode categories, and scale numeric features.
4. **Model building:** choose Logistic Regression or Random Forest.
5. **Model training:** train on a training subset of the data.
6. **Evaluation:** test on a held-out subset and report accuracy and classification metrics.
7. **Prediction:** enter a new applicant's details to estimate the loan status.
""")

elif page == "Project Information":
    st.subheader("Abstract")
    st.write("Loan Approval Prediction is a machine-learning project that estimates loan status from applicant information such as income, education, employment, credit history, loan amount, and property area.")
    st.subheader("Introduction")
    st.write("Manual review of loan applications can take time. A machine-learning model can identify patterns in historical records and provide decision support. Final lending decisions must remain with qualified financial institutions.")
    st.subheader("Why this project?")
    st.markdown("- Demonstrates a practical use of supervised machine learning.\n- Shows a data preprocessing and model evaluation workflow.\n- Provides an interactive website for testing predictions.\n- Helps explain how data-driven decision support works.")
    st.subheader("Algorithms")
    st.markdown("- **Logistic Regression:** estimates class probabilities using a linear decision function.\n- **Random Forest:** combines multiple decision trees to make a prediction.")
    st.subheader("Tools and technologies")
    st.markdown("- Python\n- Streamlit\n- Pandas and NumPy\n- Scikit-learn")
    st.subheader("Applications")
    st.markdown("- Educational demonstration of loan classification\n- Credit-risk analysis research\n- Financial decision-support prototypes")
    st.subheader("Advantages")
    st.markdown("- Fast, repeatable predictions\n- Automated preprocessing and evaluation\n- Interactive interface\n- Easy comparison of two algorithms")
    st.subheader("Limitations")
    st.markdown("- Results depend on dataset quality and representativeness.\n- Historical patterns may contain bias.\n- Model output should not be used alone for real financial decisions.")
    st.subheader("Future enhancements")
    st.markdown("- Test on a larger, more representative dataset.\n- Add explainability tools and fairness checks.\n- Improve validation and user input checks.\n- Deploy securely after reviewing privacy and responsible-use requirements.")
    st.subheader("Conclusion")
    st.write("This project demonstrates a complete educational workflow for loan status prediction, from data preparation and model training to evaluation and interactive predictions.")


elif page == "Application Details":
    st.subheader("Application Details")
    st.write("Every successful prediction is saved in a local file named `applications.csv` beside `app.py`.")
    if APPLICATIONS_FILE.exists():
        applications = pd.read_csv(APPLICATIONS_FILE)
        st.metric("Total applications submitted", len(applications))
        if not applications.empty:
            approved_count = int((applications["Prediction"].astype(str) == "Approved").sum()) if "Prediction" in applications.columns else 0
            rejected_count = int((applications["Prediction"].astype(str) == "Rejected").sum()) if "Prediction" in applications.columns else 0
            c1, c2 = st.columns(2)
            c1.metric("Predicted Approved", approved_count)
            c2.metric("Predicted Rejected", rejected_count)
            st.markdown("### Saved applications")
            st.dataframe(applications, use_container_width=True, hide_index=True)
            st.download_button(
                "Download application details as CSV",
                data=applications.to_csv(index=False).encode("utf-8"),
                file_name="applications.csv",
                mime="text/csv",
            )
            if st.button("Clear all saved application details", type="secondary"):
                APPLICATIONS_FILE.unlink(missing_ok=True)
                st.success("Saved application details cleared. Refresh the page to update the count.")
                st.rerun()
    else:
        st.info("No applications have been submitted yet. Open Predict Loan Status, enter applicant values, and click Predict Loan Status. The first successful prediction will appear here.")
