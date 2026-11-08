import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from pathlib import Path

from sklearn.model_selection import GroupShuffleSplit

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    RocCurveDisplay
)


print("\nLIBRARIES IMPORTED SUCCESSFULLY!")


BASE_DIR = Path(__file__).resolve().parents[2]

DATA_PATH = BASE_DIR / "data" / "NPL_final.csv"

MODEL_DIR = Path(__file__).resolve().parent

MODEL_DIR.mkdir(parents=True, exist_ok=True)


df = pd.read_csv(DATA_PATH)

print("\n========== DATASET LOADED ==========")

print("Dataset path:", DATA_PATH)
print("Dataset shape:", df.shape)


print("\n========== DUPLICATE CHECK ==========")

print("Duplicates before:", df.duplicated().sum())

df = df.drop_duplicates().copy()

print("Duplicates after:", df.duplicated().sum())


# 0 = No Wicket
# 1 = Wicket

df["is_wicket"] = (
    df["dismissal_kind"].notna()
    & df["dismissal_kind"].ne("Retired Hurt")
).astype(int)


print("\n========== TARGET DISTRIBUTION ==========")

print(df["is_wicket"].value_counts())

print("\nTarget percentage:")

print(df["is_wicket"].value_counts(normalize=True) * 100)


group_columns = ["match_id", "inning"]


# Over number

df["over_number"] = np.floor(df["ball_over"])


# Legal delivery

df["is_legal_delivery"] = (
    (df["wide_runs"] == 0)
    & (df["noball_runs"] == 0)
).astype(int)


# Balls before current delivery

df["balls_before"] = (
    df.groupby(group_columns, sort=False)["is_legal_delivery"]
    .transform(lambda x: x.cumsum().shift(1, fill_value=0))
)


# Runs before current delivery

df["runs_before"] = (
    df.groupby(group_columns, sort=False)["total_runs"]
    .transform(lambda x: x.cumsum().shift(1, fill_value=0))
)


# Wickets before current delivery

df["wickets_before"] = (
    df.groupby(group_columns, sort=False)["is_wicket"]
    .transform(lambda x: x.cumsum().shift(1, fill_value=0))
)


# Previous delivery runs

df["previous_ball_runs"] = (
    df.groupby(group_columns, sort=False)["total_runs"]
    .shift(1)
    .fillna(0)
)


# Previous delivery wicket

df["previous_ball_wicket"] = (
    df.groupby(group_columns, sort=False)["is_wicket"]
    .shift(1)
    .fillna(0)
)


print("\nFEATURE ENGINEERING COMPLETED!")


#SELECT FEATURES


numerical_features = [
    "inning",
    "over_number",
    "balls_before",
    "runs_before",
    "wickets_before",
    "previous_ball_runs",
    "previous_ball_wicket"
]


categorical_features = [
    "batting_team",
    "bowling_team",
    "batsman",
    "bowler",
    "batting_style",
    "bowling_style",
    "bowling_arm"
]


features = numerical_features + categorical_features

X = df[features]

y = df["is_wicket"]


print("\n========== FEATURE INFORMATION ==========")

print("X shape:", X.shape)
print("y shape:", y.shape)


#TRAIN TEST SPLIT

# Keep deliveries from the same match together.

splitter = GroupShuffleSplit(
    n_splits=1,
    test_size=0.20,
    random_state=42
)


train_index, test_index = next(
    splitter.split(X, y, groups=df["match_id"])
)


X_train = X.iloc[train_index]

X_test = X.iloc[test_index]

y_train = y.iloc[train_index]

y_test = y.iloc[test_index]


print("\n========== TRAIN TEST SPLIT ==========")

print("Training rows:", len(X_train))

print("Testing rows:", len(X_test))

print(
    "Training matches:",
    df.iloc[train_index]["match_id"].nunique()
)

print(
    "Testing matches:",
    df.iloc[test_index]["match_id"].nunique()
)


#DATA PREPROCESSING
#Numerical preprocessing:
#Replace missing values with median.
#Scale numerical values.

numerical_pipeline = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ]
)


#Categorical preprocessing:
#Replace missing values.
#Convert categories into numerical columns.

categorical_pipeline = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ]
)


preprocessor = ColumnTransformer(
    transformers=[
        ("num", numerical_pipeline, numerical_features),
        ("cat", categorical_pipeline, categorical_features)
    ]
)


#CREATE KNN MODEL


knn = KNeighborsClassifier(

    n_neighbors=15,

    weights="distance",

    metric="minkowski",

    p=2,

    algorithm="brute"

)


#CREATE COMPLETE PIPELINE


knn_model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("classifier", knn)
    ]
)


#TRAIN MODEL


print("\n========== TRAINING KNN MODEL ==========")

knn_model.fit(X_train, y_train)

print("KNN MODEL TRAINED SUCCESSFULLY!")


#MAKE PREDICTIONS


y_pred = knn_model.predict(X_test)

y_prob = knn_model.predict_proba(X_test)[:, 1]


#MODEL EVALUATION

accuracy = accuracy_score(y_test, y_pred)

precision = precision_score(
    y_test,
    y_pred,
    zero_division=0
)

recall = recall_score(
    y_test,
    y_pred,
    zero_division=0
)

f1 = f1_score(
    y_test,
    y_pred,
    zero_division=0
)

roc_auc = roc_auc_score(y_test, y_prob)

pr_auc = average_precision_score(y_test, y_prob)


print("\n========== KNN RESULTS ==========")

print(f"Accuracy:  {accuracy:.4f}")

print(f"Precision: {precision:.4f}")

print(f"Recall:    {recall:.4f}")

print(f"F1 Score:  {f1:.4f}")

print(f"ROC-AUC:   {roc_auc:.4f}")

print(f"PR-AUC:    {pr_auc:.4f}")


#CLASSIFICATION REPORT

print("\n========== CLASSIFICATION REPORT ==========")

print(
    classification_report(
        y_test,
        y_pred,
        target_names=["No Wicket", "Wicket"],
        zero_division=0
    )
)


#CONFUSION MATRIX

cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(7, 5))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["No Wicket", "Wicket"],
    yticklabels=["No Wicket", "Wicket"]
)

plt.title("KNN - Confusion Matrix")

plt.xlabel("Predicted")

plt.ylabel("Actual")

plt.tight_layout()

plt.savefig(
    MODEL_DIR / "knn_confusion_matrix.png"
)

plt.close()


#ROC CURVE

RocCurveDisplay.from_predictions(
    y_test,
    y_prob
)

plt.title("KNN - ROC Curve")

plt.tight_layout()

plt.savefig(
    MODEL_DIR / "knn_roc_curve.png"
)

plt.close()


# 18. PRECISION-RECALL PERFORMANCE
# ============================================================

from sklearn.metrics import PrecisionRecallDisplay

PrecisionRecallDisplay.from_predictions(
    y_test,
    y_prob
)

plt.title("KNN - Precision Recall Curve")

plt.tight_layout()

plt.savefig(
    MODEL_DIR / "knn_precision_recall_curve.png"
)

plt.close()


# 19. SAVE TRAINED MODEL
# ============================================================

MODEL_PATH = MODEL_DIR / "knn_wicket_model.pkl"

joblib.dump(knn_model, MODEL_PATH)

print("\nModel saved:", MODEL_PATH)


#SAVE RESULTS


results = pd.DataFrame({

    "Model": ["KNN"],

    "Accuracy": [accuracy],

    "Precision": [precision],

    "Recall": [recall],

    "F1 Score": [f1],

    "ROC-AUC": [roc_auc],

    "PR-AUC": [pr_auc]

})


RESULT_PATH = MODEL_DIR / "knn_results.csv"

results.to_csv(
    RESULT_PATH,
    index=False
)


print("Results saved:", RESULT_PATH)


#FINAL OUTPUT

print("\n========== PROJECT COMPLETED ==========")

print("KNN training and evaluation completed!")

print("\nGenerated files:")

for file in MODEL_DIR.iterdir():

    if file.is_file():

        print(file.name)