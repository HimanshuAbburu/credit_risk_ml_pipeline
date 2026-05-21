"""
Loan Default Prediction Pipeline
---------------------------------
Loads raw loan data, applies feature engineering, and evaluates
a pre-trained LightGBM model using ROC-AUC and classification report.
"""

import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import train_test_split

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

BASE_DIR = Path(__file__).parent
DATA_PATH = BASE_DIR / "loan_data.csv"
ENCODER_PATH = BASE_DIR / "encoder.pkl"
MODEL_PATH = BASE_DIR / "lightGBM.pkl"

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

RANDOM_SEED = 10
TEST_SIZE = 0.2
MIN_AGE = 18

FEATURES_ALL = [
    "new_customer", "application_date", "income_verification", "language",
    "date_of_birth", "gender", "country", "loan_amount", "county", "city",
    "use_of_loan", "education", "marital_status", "nr_dependants",
    "employment_status", "employment_duration", "employment_position",
    "work_experience", "occupation", "home_ownership",
    "income_from_employer", "income_from_pension",
    "income_from_family_allowance", "income_from_social_welfare",
    "income_from_leave_pay", "income_from_child_support", "income_other",
    "nr_debt_items", "total_debt", "credit_score_1", "credit_score_2",
    "credit_score_3", "credit_score_4", "nr_previous_loans",
    "amount_previous_loans", "previous_repayments",
    "previous_early_repayments", "previous_early_repayments_count",
]

# Must match the order the model was trained on
FEATURES_INPUT = [
    "new_customer", "income_verification", "language", "gender", "country",
    "loan_amount", "use_of_loan", "education", "marital_status",
    "nr_dependants", "employment_status", "employment_duration",
    "work_experience", "occupation", "home_ownership",
    "income_from_employer", "income_from_pension",
    "income_from_family_allowance", "income_from_social_welfare",
    "income_from_leave_pay", "income_from_child_support", "income_other",
    "nr_debt_items", "total_debt", "credit_score_1", "credit_score_2",
    "credit_score_3", "credit_score_4", "nr_previous_loans",
    "amount_previous_loans", "previous_repayments",
    "previous_early_repayments", "previous_early_repayments_count",
    "total_income", "debt_to_income_ratio", "cash",
    "age", "day_of_week", "date_of_month", "month", "hour",
]

FEATURES_INCOME = [
    "income_from_employer",
    "income_from_pension",
    "income_from_family_allowance",
    "income_from_social_welfare",
    "income_from_leave_pay",
    "income_from_child_support",
    "income_other",
]

FEATURES_ENCODE = [
    "income_verification",
    "language",
    "gender",
    "country",
    "use_of_loan",
    "education",
    "marital_status",
    "employment_status",
    "employment_duration",
    "work_experience",
    "occupation",
    "home_ownership",
    "credit_score_1",
    "credit_score_2",
    "credit_score_3",
]

IMPUTATION_DICT = {
    "nr_dependants": -1,
    "credit_score_4": -1,
    "previous_repayments": -1,
    "previous_early_repayments": -1,
    "income_verification": "unverified",
    "gender": "missing",
    "education": "missing",
    "marital_status": "missing",
    "employment_status": "missing",
    "employment_duration": "missing",
    "work_experience": "missing",
    "occupation": "missing",
    "home_ownership": "missing",
    "credit_score_1": "missing",
    "credit_score_2": "missing",
    "credit_score_3": "missing",
}

# Only columns where the encoder was trained with a 'rare' category.
# Values not in this list will be replaced with 'rare' before encoding.
FREQUENT_CATEGORIES_DICT = {
    "language":       ["estonian", "finnish", "spanish", "russian"],
    "use_of_loan":    ["unknown", "other", "home_improvement", "loan_consolidation"],
    "occupation":     ["missing", "other", "retail"],
    "home_ownership": ["owner", "tenant_furnished", "living_with_parents", "mortgage", "tenant_unfurnished"],
    "credit_score_1": ["missing", "M", "M1"],
    "credit_score_2": ["missing", "B"],
    "credit_score_3": ["missing", "RL2"],
}

# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------

def _impute(df: pd.DataFrame) -> pd.DataFrame:
    return df.fillna(IMPUTATION_DICT)


def _create_income_features(df: pd.DataFrame) -> pd.DataFrame:
    df["total_income"] = df[FEATURES_INCOME].sum(axis=1)
    # NOTE: the model was trained with inf/nan NOT replaced in debt_to_income_ratio
    # (due to a bug in the original notebook). This replicates that behaviour so
    # predictions are consistent. Retrain the model with the corrected version below
    # once a new training run is done:
    #   df["debt_to_income_ratio"] = (df["total_debt"] / df["total_income"].replace(0, np.nan)).fillna(0)
    df["debt_to_income_ratio"] = df["total_debt"] / df["total_income"]
    df["cash"] = df["total_income"] - df["total_debt"]
    return df


def _create_date_features(df: pd.DataFrame) -> pd.DataFrame:
    app_date = pd.to_datetime(df["application_date"])
    dob = pd.to_datetime(df["date_of_birth"])
    df["age"] = ((app_date - dob).dt.days / 365).astype(int)
    df["day_of_week"] = app_date.dt.dayofweek
    df["date_of_month"] = app_date.dt.day
    df["month"] = app_date.dt.month
    df["hour"] = app_date.dt.hour
    return df


def _group_rare_categories(df: pd.DataFrame) -> pd.DataFrame:
    for col, frequent in FREQUENT_CATEGORIES_DICT.items():
        df[col] = np.where(df[col].isin(frequent), df[col], "rare")
    return df


def _encode(df: pd.DataFrame, encoder) -> pd.DataFrame:
    df[FEATURES_ENCODE] = encoder.transform(df[FEATURES_ENCODE])
    return df


def feature_engineering_pipeline(df: pd.DataFrame, encoder) -> pd.DataFrame:
    """
    Apply all feature engineering steps to raw input data.

    Parameters
    ----------
    df : pd.DataFrame
        Raw input dataframe containing FEATURES_ALL columns.
    encoder : fitted OrdinalEncoder
        Encoder fitted on FEATURES_ENCODE columns.

    Returns
    -------
    pd.DataFrame
        Transformed dataframe with FEATURES_INPUT columns, filtered to age >= MIN_AGE.
    """
    df = df[FEATURES_ALL].copy()
    df = _impute(df)
    df = _create_income_features(df)
    df = _create_date_features(df)
    df = _group_rare_categories(df)
    df = _encode(df, encoder)
    df = df[df["age"] >= MIN_AGE]
    return df[FEATURES_INPUT]


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate(model, X: pd.DataFrame, y: pd.Series, split_name: str) -> None:
    """Log ROC-AUC and classification report for a given split."""
    proba = model.predict_proba(X)[:, 1]
    preds = model.predict(X)
    roc = roc_auc_score(y, proba)
    report = classification_report(y, preds)
    logger.info("%s ROC-AUC: %.4f", split_name, roc)
    logger.info("%s classification report:\n%s", split_name, report)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    logger.info("Loading encoder and model")
    encoder = joblib.load(ENCODER_PATH)
    model = joblib.load(MODEL_PATH)

    logger.info("Loading data from %s", DATA_PATH)
    df = pd.read_csv(DATA_PATH, low_memory=False)

    X = df.drop(columns=["default"])
    y = df["default"]

    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_SEED
    )

    logger.info("Running feature engineering pipeline")
    X_train = feature_engineering_pipeline(X_train_raw, encoder)
    X_test = feature_engineering_pipeline(X_test_raw, encoder)

    # Align labels to rows that survived the age filter
    y_train = y_train.loc[X_train.index]
    y_test = y_test.loc[X_test.index]

    logger.info("Evaluating model")
    evaluate(model, X_train, y_train, "Train")
    evaluate(model, X_test, y_test, "Test")


if __name__ == "__main__":
    main()
