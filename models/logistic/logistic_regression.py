import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from pathlib import Path

# Data preprocessing
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Train-test splitting
from sklearn.model_selection import GroupShuffleSplit

# Machine Learning algorithm
from sklearn.linear_model import LogisticRegression

# Model evaluation
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
    average_precision_score,
    RocCurveDisplay
)


print("\nLIBRARIES IMPORTED SUCCESSFULLY!")

#LOAD DATASET

# Get main project directory
BASE_DIR = Path(__file__).resolve().parents[2]

# Dataset location
DATA_PATH = BASE_DIR / "data" / "NPL_final.csv"

# Model saving location
MODEL_DIR = Path(__file__).resolve().parent

# Load dataset
df = pd.read_csv(DATA_PATH)

print("Dataset loaded successfully!")
print("Dataset path:", DATA_PATH)
print("Dataset shape:", df.shape)

# Load CSV file
df = pd.read_csv(DATA_PATH)

print("\n========== DATASET LOADED ==========")

print("Dataset Shape:", df.shape)

print("\nFirst 5 Records:")
print(df.head())

print("\nColumn Names:")
print(df.columns.tolist())

#DATA CLEANING

# Remove unwanted spaces from column names
df.columns = df.columns.str.strip()

# Check duplicate records
print("\n========== DUPLICATE ANALYSIS ==========")

print("Duplicates Before:", df.duplicated().sum())

# Remove exact duplicate records
df = df.drop_duplicates().copy()

print("Duplicates After:", df.duplicated().sum())


# Check missing values
print("\n========== MISSING VALUE ANALYSIS ==========")

print(df.isnull().sum())

#FEATURE ENGINEERING
print("\n========== FEATURE ENGINEERING ==========")

# Create target variable
# 0 = No wicket
# 1 = Wicket
# Retired Hurt is excluded because it is not a normal
# dismissal credited as a wicket.

df["is_wicket"] = (
    df["dismissal_kind"].notna()
    & df["dismissal_kind"].ne("Retired Hurt")
).astype(int)


# Convert ball_over into an over number.
# Example:
# 5.1 = Over 5
# 12.4 = Over 12
# 19.3 = Over 19

df["over_number"] = np.floor(
    df["ball_over"]
).astype(int)


df["_delivery_order"] = np.arange(len(df))

df = df.sort_values(
    ["match_id", "inning", "_delivery_order"],
    kind="stable"
).copy()


# Group deliveries by match and innings.
# This prevents previous-match information from
# leaking into another match.

groups = df.groupby(
    ["match_id", "inning"],
    sort=False
)



# Previous performance features
# Legal delivery indicator.
# Wides and no-balls do not count as legal balls.

df["is_legal_delivery"] = (
    (df["wide_runs"] == 0)
    & (df["noball_runs"] == 0)
).astype(int)


# Number of legal deliveries before current delivery.

df["balls_before"] = groups[
    "is_legal_delivery"
].transform(
    lambda x: x.cumsum().shift(1, fill_value=0)
)


# Total runs scored before current delivery.

df["runs_before"] = groups[
    "total_runs"
].transform(
    lambda x: x.cumsum().shift(1, fill_value=0)
)


# Number of wickets lost before current delivery.

df["wickets_before"] = groups[
    "is_wicket"
].transform(
    lambda x: x.cumsum().shift(1, fill_value=0)
)


# Runs scored on the previous recorded delivery.

df["previous_ball_runs"] = groups[
    "total_runs"
].shift(1).fillna(0)


# Whether the previous recorded delivery resulted in a wicket.

df["previous_ball_wicket"] = groups[
    "is_wicket"
].shift(1).fillna(0)


print("\nNew Features Created Successfully!")

print(
    df[
        [
            "inning",
            "over_number",
            "balls_before",
            "runs_before",
            "wickets_before",
            "previous_ball_runs",
            "is_wicket"
        ]
    ].head()
)

#SELECT FEATURES (X) AND TARGET (Y)

# Numerical features
numerical_features = [

    "inning",
    "over_number",
    "balls_before",
    "runs_before",
    "wickets_before",
    "previous_ball_runs",
    "previous_ball_wicket"

]


# Categorical features
categorical_features = [

    "batting_team",
    "bowling_team",
    "batsman",
    "bowler",
    "batting_style",
    "bowling_style",
    "bowling_arm"

]


# Input variables
X = df[
    numerical_features + categorical_features
].copy()


# Target variable
y = df["is_wicket"].copy()


print("\n========== FEATURES AND TARGET ==========")

print("X Shape:", X.shape)

print("Y Shape:", y.shape)

print("\nTarget Distribution:")

print(y.value_counts())

print("\nTarget Percentage:")

print(y.value_counts(normalize=True) * 100)

#TRAIN-TEST SPLIT

# We split by match_id instead of randomly splitting
# individual deliveries.
# This prevents deliveries from the same match
# appearing in both training and testing datasets.

splitter = GroupShuffleSplit(
    n_splits=1,
    test_size=0.20,
    random_state=42
)


train_idx, test_idx = next(
    splitter.split(
        X,
        y,
        groups=df["match_id"]
    )
)


# Training data
X_train = X.iloc[train_idx]

y_train = y.iloc[train_idx]


# Testing data
X_test = X.iloc[test_idx]

y_test = y.iloc[test_idx]


print("\n========== TRAIN TEST SPLIT ==========")

print("Training Records:", len(X_train))

print("Testing Records:", len(X_test))

print(
    "Training Matches:",
    df.iloc[train_idx]["match_id"].nunique()
)

print(
    "Testing Matches:",
    df.iloc[test_idx]["match_id"].nunique()
)


# Check whether both classes exist in training and testing.

if y_train.nunique() != 2 or y_test.nunique() != 2:
    raise ValueError(
        "Training or testing data contains only one class. "
        "Choose a different match split."
    )

#DATA PREPROCESSING

numerical_transformer = Pipeline(
    steps=[

        (
            "imputer",
            SimpleImputer(strategy="median")
        ),

        (
            "scaler",
            StandardScaler()
        )

    ]
)


# Categorical preprocessing:

categorical_transformer = Pipeline(
    steps=[

        (
            "imputer",
            SimpleImputer(strategy="most_frequent")
        ),

        (
            "encoder",
            OneHotEncoder(handle_unknown="ignore")
        )

    ]
)


# Combine numerical and categorical preprocessing.

preprocessor = ColumnTransformer(

    transformers=[

        (
            "numerical",
            numerical_transformer,
            numerical_features
        ),

        (
            "categorical",
            categorical_transformer,
            categorical_features
        )

    ]

)

print("\nPREPROCESSING PIPELINE CREATED!")

#CREATE LOGISTIC REGRESSION MODEL
# Pipeline combines:
# 1. Data preprocessing
# 2. Logistic Regression
#
# class_weight="balanced" helps account for
# the imbalance between wicket and non-wicket deliveries.

logistic_model = Pipeline(

    steps=[

        (
            "preprocessor",
            preprocessor
        ),

        (
            "classifier",
            LogisticRegression(

                max_iter=2000,

                class_weight="balanced",

                C=1.0,

                random_state=42

            )
        )

    ]

)

#TRAIN THE MODEL
print("\n========== MODEL TRAINING ==========")

logistic_model.fit(
    X_train,
    y_train
)

print("LOGISTIC REGRESSION TRAINED SUCCESSFULLY!")
#MAKE PREDICTIONS
# Predict class labels.

y_pred = logistic_model.predict(X_test)


# Predict probability of a wicket.

y_probability = logistic_model.predict_proba(
    X_test
)[:, 1]


print("\n========== MODEL PREDICTIONS ==========")

print("Actual Values:")

print(y_test.values[:20])

print("\nPredicted Values:")

print(y_pred[:20])

print("\nWicket Probabilities:")

print(y_probability[:20])

#MODEL EVALUATION
# Accuracy
accuracy = accuracy_score(
    y_test,
    y_pred
)


# Precision
precision = precision_score(
    y_test,
    y_pred,
    zero_division=0
)


# Recall
recall = recall_score(
    y_test,
    y_pred,
    zero_division=0
)


# F1 Score
f1 = f1_score(
    y_test,
    y_pred,
    zero_division=0
)


# ROC-AUC
roc_auc = roc_auc_score(
    y_test,
    y_probability
)


# Precision-Recall AUC
pr_auc = average_precision_score(
    y_test,
    y_probability
)


print("\n========== MODEL PERFORMANCE ==========")

print(f"Accuracy:  {accuracy:.4f}")

print(f"Precision: {precision:.4f}")

print(f"Recall:    {recall:.4f}")

print(f"F1 Score:  {f1:.4f}")

print(f"ROC-AUC:   {roc_auc:.4f}")

print(f"PR-AUC:    {pr_auc:.4f}")


# Detailed classification report

print("\n========== CLASSIFICATION REPORT ==========")

print(

    classification_report(

        y_test,

        y_pred,

        target_names=[
            "No Wicket",
            "Wicket"
        ],

        zero_division=0

    )

)

#CONFUSION MATRIX
cm = confusion_matrix(
    y_test,
    y_pred
)


plt.figure(figsize=(7, 5))


sns.heatmap(

    cm,

    annot=True,

    fmt="d",

    cmap="Blues",

    xticklabels=[
        "No Wicket",
        "Wicket"
    ],

    yticklabels=[
        "No Wicket",
        "Wicket"
    ]

)


plt.title(
    "NPL Logistic Regression - Confusion Matrix"
)

plt.xlabel("Predicted Label")

plt.ylabel("Actual Label")

plt.tight_layout()

plt.show()

#ROC CURVE
RocCurveDisplay.from_predictions(

    y_test,

    y_probability

)


plt.title(
    "NPL Logistic Regression - ROC Curve"
)

plt.tight_layout()

plt.show()

#MODEL PERFORMANCE TABLE
results = pd.DataFrame({

    "Metric": [

        "Accuracy",

        "Precision",

        "Recall",

        "F1 Score",

        "ROC-AUC",

        "PR-AUC"

    ],

    "Score": [

        accuracy,

        precision,

        recall,

        f1,

        roc_auc,

        pr_auc

    ]

})


print("\n========== FINAL RESULTS ==========")

print(results.to_string(index=False))



#SAVE TRAINED MODEL


MODEL_DIR = Path(__file__).resolve().parent

MODEL_DIR.mkdir(
    parents=True,
    exist_ok=True
)

MODEL_PATH = MODEL_DIR / "logistic_wicket_model.pkl"

joblib.dump(
    logistic_model,
    MODEL_PATH
)


print("\nMODEL SAVED SUCCESSFULLY!")

print("Location:", MODEL_PATH)


RESULT_PATH = MODEL_DIR / "logistic_results.csv"

results.to_csv(
    RESULT_PATH,
    index=False
)


print("\nEvaluation results saved!")

print("Location:", RESULT_PATH)



print("\n" + "=" * 55)

print("NPL LOGISTIC REGRESSION PROJECT COMPLETED!")

print("*" * 55)

