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
from sklearn.svm import LinearSVC

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

BASE_DIR = Path(__file__).resolve().parents[2]

DATA_PATH = BASE_DIR / "data" / "NPL_final.csv"

MODEL_DIR = Path(__file__).resolve().parent

df = pd.read_csv(DATA_PATH)

print("\nDATASET LOADED SUCCESSFULLY!")

print("Dataset Path:", DATA_PATH)

print("Dataset Shape:", df.shape)


df.columns = df.columns.str.strip()

print("Duplicates Before:", df.duplicated().sum())

df = df.drop_duplicates().copy()

print("Duplicates After:", df.duplicated().sum())


# Check missing values
print(df.isnull().sum())

#FEATURE ENGINEERING
print("\n========== FEATURE ENGINEERING ==========")


# Create target variable

# 0 = No Wicket
# 1 = Wicket

df["is_wicket"] = (
    df["dismissal_kind"].notna()
    & df["dismissal_kind"].ne("Retired Hurt")
).astype(int)


# Extract over number

df["over_number"] = np.floor(
    df["ball_over"]
).astype(int)


# Preserve original delivery order

df["_delivery_order"] = np.arange(len(df))


# Sort dataset

df = df.sort_values(
    ["match_id", "inning", "_delivery_order"],
    kind="stable"
).copy()


# Group by match and innings

groups = df.groupby(
    ["match_id", "inning"],
    sort=False
)

# Previous performance features

# Legal delivery

df["is_legal_delivery"] = (
    (df["wide_runs"] == 0)
    & (df["noball_runs"] == 0)
).astype(int)


# Balls before current delivery

df["balls_before"] = groups[
    "is_legal_delivery"
].transform(
    lambda x: x.cumsum().shift(1, fill_value=0)
)


# Runs before current delivery

df["runs_before"] = groups[
    "total_runs"
].transform(
    lambda x: x.cumsum().shift(1, fill_value=0)
)


# Wickets before current delivery

df["wickets_before"] = groups[
    "is_wicket"
].transform(
    lambda x: x.cumsum().shift(1, fill_value=0)
)


# Previous ball runs

df["previous_ball_runs"] = groups[
    "total_runs"
].shift(1).fillna(0)


# Previous ball wicket

df["previous_ball_wicket"] = groups[
    "is_wicket"
].shift(1).fillna(0)


print("\nFEATURE ENGINEERING COMPLETED!")

#SELECT FEATURES (X) AND TARGET (Y)

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


# Input features

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

#TRAIN-TEST SPLIT

# Split by match_id to prevent match leakage.

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


if y_train.nunique() != 2 or y_test.nunique() != 2:

    raise ValueError(
        "Training or testing data contains only one class."
    )

#DATA PREPROCESSING

# Numerical preprocessing

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


# Categorical preprocessing

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


# Combine preprocessing

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

#CREATE SVM MODEL

svm_model = Pipeline(

    steps=[

        (
            "preprocessor",
            preprocessor
        ),

        (
            "classifier",
            LinearSVC(

                C=1.0,

                class_weight="balanced",

                max_iter=10000,

                random_state=42

            )
        )

    ]

)
#TRAIN THE MODEL

print("\n========== MODEL TRAINING ==========")

svm_model.fit(
    X_train,
    y_train
)

print("\nSVM MODEL TRAINED SUCCESSFULLY!")

#MAKE PREDICTIONS

# Predict class labels

y_pred = svm_model.predict(X_test)


# LinearSVC does not provide predict_proba().
# decision_function provides classification scores instead.

y_scores = svm_model.decision_function(X_test)


print("\n========== MODEL PREDICTIONS ==========")

print("Actual Values:")

print(y_test.values[:20])


print("\nPredicted Values:")

print(y_pred[:20])


print("\nDecision Scores:")

print(y_scores[:20])

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
    y_scores
)


# Precision-Recall AUC

pr_auc = average_precision_score(
    y_test,
    y_scores
)


print("\n========== SVM MODEL PERFORMANCE ==========")

print(f"Accuracy:  {accuracy:.4f}")

print(f"Precision: {precision:.4f}")

print(f"Recall:    {recall:.4f}")

print(f"F1 Score:  {f1:.4f}")

print(f"ROC-AUC:   {roc_auc:.4f}")

print(f"PR-AUC:    {pr_auc:.4f}")


# Classification report

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

# Save files inside models/svm/

MODEL_DIR.mkdir(
    parents=True,
    exist_ok=True
)


MODEL_PATH = MODEL_DIR / "svm_wicket_model.pkl"


joblib.dump(
    svm_model,
    MODEL_PATH
)


print("\nSVM MODEL SAVED SUCCESSFULLY!")

print("Location:", MODEL_PATH)

#SAVE MODEL EVALUATION RESULTS

RESULT_PATH = MODEL_DIR / "svm_results.csv"


results.to_csv(
    RESULT_PATH,
    index=False
)


print("\nSVM RESULTS SAVED SUCCESSFULLY!")

print("Location:", RESULT_PATH)
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
    "NPL SVM - Confusion Matrix"
)

plt.xlabel("Predicted Label")

plt.ylabel("Actual Label")

plt.tight_layout()


plt.show()
#ROC CURVE

RocCurveDisplay.from_predictions(

    y_test,

    y_scores

)


plt.title(
    "NPL SVM - ROC Curve"
)


plt.tight_layout()


plt.show()

print("\n" + "=" * 55)

print("NPL SVM PROJECT COMPLETED!")

print("=" * 55)