# Credit Risk ML Pipeline

This project was built to take a proof-of-concept LightGBM credit risk model into production. The model predicts the probability that a loan applicant will default, which can be used by a peer-to-peer lending platform to determine the interest rate charged to each borrower. The pipeline accepts raw loan application data, applies all necessary feature engineering — imputation, income and date feature creation, categorical encoding — and outputs default probability predictions ready for use at scale.

## Project Structure

```
credit_risk_ml_pipeline/
├── loan_data.csv          # Raw loan application data
├── loan_data_pipeline.py  # Production pipeline (feature engineering + evaluation)
├── main.ipynb             # Exploratory notebook (EDA, training, experimentation)
├── encoder.pkl            # Fitted OrdinalEncoder for categorical features
├── lightGBM.pkl           # Trained LightGBM classifier
├── requirements.txt       # Python dependencies
└── latest_customers.csv   # Sample of new customers for inference
```

## Setup

```bash
pip install -r requirements.txt
```

> The saved model artifacts (`encoder.pkl`, `lightGBM.pkl`) were built with scikit-learn 1.8.0.  
> To avoid version warnings, match that version: `pip install scikit-learn==1.8.0`

## Usage

Run the pipeline to evaluate the model on a train/test split:

```bash
python loan_data_pipeline.py
```

This will log ROC-AUC and a classification report for both train and test sets.

## Pipeline Steps

1. **Imputation** — fills missing values with sensible defaults per column
2. **Income features** — computes `total_income`, `debt_to_income_ratio`, and `cash`
3. **Date features** — extracts `age`, `day_of_week`, `date_of_month`, `month`, `hour` from `application_date`
4. **Rare category grouping** — replaces infrequent categories with `"rare"` for selected columns
5. **Ordinal encoding** — encodes all categorical features using the fitted encoder
6. **Age filter** — removes applicants under 18

## Model

| Split | ROC-AUC |
|-------|---------|
| Train | 0.781   |
| Test  | 0.729   |

## Known Issues

- The model was trained with a bug where `debt_to_income_ratio` `inf`/`nan` values were not replaced. The pipeline replicates this to keep predictions consistent. **Retraining with the corrected DTI calculation is recommended.**
- Model artifacts were serialised with scikit-learn 1.8.0 — using a different version may produce warnings or unexpected results.
