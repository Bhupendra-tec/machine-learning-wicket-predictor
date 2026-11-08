import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]

OUTPUT_DIR = BASE_DIR / "models" / "comparison"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
#MODEL PERFORMANCE DATA

results = {

    "Model": [
        "Logistic Regression",
        "SVM",
        "Decision Tree",
        "KNN",
        "Random Forest"
    ],

    "Accuracy": [
        0.5771,
        0.571603,
        0.7778,
        0.9370,
        0.8078
    ],

    "Precision": [
        0.0612,
        0.060383,
        0.1012,
        0.0000,
        0.1019
    ],

    "Recall": [
        0.3981,
        0.398058,
        0.3204,
        0.0000,
        0.2621
    ],

    "F1 Score": [
        0.1061,
        0.104859,
        0.1538,
        0.0000,
        0.1467
    ],

    "ROC-AUC": [
        0.4986,
        0.496680,
        0.5597,
        0.5510,
        0.5321
    ],

    "PR-AUC": [
        0.0678,
        0.070622,
        0.0834,
        0.0739,
        0.0790
    ]

}

df = pd.DataFrame(results)

print("\n========== MODEL COMPARISON ==========\n")

print(df.to_string(index=False))

#RANK MODELS BY PR-AUC

df = df.sort_values(
    by="PR-AUC",
    ascending=False
).reset_index(drop=True)

df["Rank"] = range(1, len(df) + 1)

print("\n========== MODEL RANKING ==========\n")

print(
    df[["Rank", "Model", "PR-AUC", "F1 Score"]]
    .to_string(index=False)
)

#SAVE COMPARISON RESULTS

csv_path = OUTPUT_DIR / "model_comparison.csv"

df.to_csv(csv_path, index=False)

print("\nComparison CSV saved:", csv_path)

#VISUALIZATION - METRIC COMPARISON

metrics = [
    "Precision",
    "Recall",
    "F1 Score",
    "ROC-AUC",
    "PR-AUC"
]

plot_df = df.set_index("Model")[metrics]

ax = plot_df.plot(
    kind="bar",
    figsize=(13, 7),
    width=0.8
)

plt.title(
    "NPL Wicket Prediction - Model Performance Comparison",
    fontsize=16
)

plt.xlabel("Machine Learning Models")

plt.ylabel("Score")

plt.xticks(rotation=20)

plt.legend(
    title="Evaluation Metrics",
    bbox_to_anchor=(1.02, 1),
    loc="upper left"
)

plt.grid(
    axis="y",
    linestyle="--",
    alpha=0.5
)

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "metric_comparison.png",
    dpi=300
)

plt.close()

#VISUALIZATION - PR-AUC COMPARISON

plt.figure(figsize=(11, 6))

plt.bar(
    df["Model"],
    df["PR-AUC"]
)

plt.axhline(
    y=454 / 7470,
    linestyle="--",
    label="Positive Class Baseline"
)

plt.title(
    "Precision-Recall AUC Comparison",
    fontsize=16
)

plt.xlabel("Machine Learning Models")

plt.ylabel("PR-AUC")

plt.xticks(rotation=20)

plt.legend()

plt.grid(
    axis="y",
    linestyle="--",
    alpha=0.5
)

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "pr_auc_comparison.png",
    dpi=300
)

plt.close()

#BEST MODEL

best_model = df.iloc[0]

print("\n========== BEST MODEL ==========\n")

print("Model:", best_model["Model"])

print("PR-AUC:", round(best_model["PR-AUC"] * 100, 2), "%")

print("F1 Score:", round(best_model["F1 Score"] * 100, 2), "%")

print("\nReason:")

print(
    "This model achieved the highest PR-AUC "
    "among the five tested models."
)

print("\nFiles generated inside:")

print(OUTPUT_DIR)

print("\n1. model_comparison.csv")

print("2. metric_comparison.png")

print("3. pr_auc_comparison.png")