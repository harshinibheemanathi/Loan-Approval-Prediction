# Loan Approval Prediction Using Machine Learning

A beginner-friendly Streamlit web application for educational loan-status prediction.

## Included
- `app.py`: website, data preprocessing, model training, evaluation, and prediction form
- `requirements.txt`: Python packages
- `project_report.md`: report content for college documentation

## Before running
1. Keep your dataset CSV in the same folder as `app.py`. Accepted names:
   - `loan_approval_dataset.csv`
   - `loan_data_set.csv`
2. The CSV should contain a `Loan_Status` column with labels such as `Y` and `N`.
3. If your CSV is elsewhere, upload it using the sidebar of the website.

## Run on Windows
Open this folder in VS Code. Select Terminal > New Terminal, then run:

```powershell
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

Open the Local URL printed in the terminal, usually `http://localhost:8501`.

## Pages
- Home
- Predict Loan Status
- Model Evaluation
- Dataset & Methodology
- Project Information

## Important
This is an academic demonstration. Predictions are not bank decisions or financial advice. Evaluate models carefully before any real-world use.
