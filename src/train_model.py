"""
Train air quality prediction models.
"""

from pathlib import Path
import json
import joblib
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, ConfusionMatrixDisplay

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "processed" / "air_quality_processed.csv"
MODELS_DIR = ROOT / "models"
FIGURES_DIR = ROOT / "outputs" / "figures"
PRED_DIR = ROOT / "outputs" / "predictions"

MODELS_DIR.mkdir(exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)
PRED_DIR.mkdir(parents=True, exist_ok=True)

FEATURES = ["pm10", "no2", "o3", "temperature", "humidity", "wind_speed"]
TARGET = "air_quality_class"

def evaluate_model(name, model, x_test, y_test):
    predictions = model.predict(x_test)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test, predictions, average="weighted", zero_division=0
    )
    return {
        "model": name,
        "accuracy": round(float(accuracy_score(y_test, predictions)), 4),
        "precision_weighted": round(float(precision), 4),
        "recall_weighted": round(float(recall), 4),
        "f1_weighted": round(float(f1), 4),
        "predictions": predictions,
    }

def main():
    data = pd.read_csv(DATA_PATH)
    x = data[FEATURES]
    y = data[TARGET]

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, stratify=y, random_state=42
    )

    random_forest = RandomForestClassifier(
        n_estimators=150, random_state=42, class_weight="balanced"
    )

    logistic_regression = Pipeline([
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(max_iter=1000, class_weight="balanced")),
    ])

    random_forest.fit(x_train, y_train)
    logistic_regression.fit(x_train, y_train)

    rf_metrics = evaluate_model("Random Forest", random_forest, x_test, y_test)
    lr_metrics = evaluate_model("Logistic Regression", logistic_regression, x_test, y_test)

    metrics = [
        {k: v for k, v in rf_metrics.items() if k != "predictions"},
        {k: v for k, v in lr_metrics.items() if k != "predictions"},
    ]

    pd.DataFrame(metrics).to_csv(PRED_DIR / "evaluation_metrics.csv", index=False)
    with open(PRED_DIR / "evaluation_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    prediction_table = x_test.copy()
    prediction_table["true_air_quality_class"] = y_test.values
    prediction_table["random_forest_prediction"] = rf_metrics["predictions"]
    prediction_table["logistic_regression_prediction"] = lr_metrics["predictions"]
    prediction_table.to_csv(PRED_DIR / "air_quality_predictions.csv", index=False)

    joblib.dump(random_forest, MODELS_DIR / "random_forest_air_quality_model.pkl")
    joblib.dump(logistic_regression, MODELS_DIR / "logistic_regression_air_quality_model.pkl")

    labels = ["Good", "Moderate", "Poor"]
    cm = confusion_matrix(y_test, rf_metrics["predictions"], labels=labels)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot()
    plt.title("Random Forest Confusion Matrix")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "confusion_matrix_random_forest.png", dpi=200)
    plt.close()

    importances = pd.DataFrame({
        "feature": FEATURES,
        "importance": random_forest.feature_importances_
    }).sort_values("importance", ascending=False)
    importances.to_csv(PRED_DIR / "feature_importance.csv", index=False)

    plt.figure(figsize=(8, 5))
    plt.bar(importances["feature"], importances["importance"])
    plt.title("Random Forest Feature Importance")
    plt.ylabel("Importance")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "feature_importance_random_forest.png", dpi=200)
    plt.close()

    plt.figure(figsize=(7, 5))
    plt.hist(data["pm10"], bins=20)
    plt.title("PM10 Distribution")
    plt.xlabel("PM10")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "pm10_histogram.png", dpi=200)
    plt.close()

    print("Training complete. Outputs saved.")

if __name__ == "__main__":
    main()
