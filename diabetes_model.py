from pathlib import Path
from typing import Dict, List, Union

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

FEATURE_COLUMNS = [
    "pregnancies",
    "glucose",
    "blood_pressure",
    "skin_thickness",
    "insulin",
    "bmi",
    "age",
    "gender",
]
TARGET_COLUMN = "diabetes"
MODEL_PATH = Path(__file__).resolve().parent / "artifacts" / "diabetes_model.joblib"


def generate_synthetic_dataset(n_samples: int = 1500, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    data = {
        "pregnancies": rng.integers(0, 8, size=n_samples),
        "glucose": np.clip(rng.normal(110, 25, size=n_samples), 60, 220),
        "blood_pressure": np.clip(rng.normal(75, 15, size=n_samples), 50, 130),
        "skin_thickness": np.clip(rng.normal(25, 10, size=n_samples), 5, 60),
        "insulin": np.clip(rng.normal(110, 70, size=n_samples), 20, 500),
        "bmi": np.clip(rng.normal(30, 6, size=n_samples), 15, 50),
        "age": np.clip(rng.normal(35, 14, size=n_samples), 20, 80),
        "gender": rng.choice([0, 1], size=n_samples),
    }

    df = pd.DataFrame(data)
    risk_score = (
        -8.0
        + 0.03 * df["age"]
        + 0.03 * df["glucose"]
        + 0.07 * df["bmi"]
        + 0.01 * df["blood_pressure"]
        + 0.002 * df["insulin"]
        + 0.15 * df["pregnancies"]
        + 0.02 * df["skin_thickness"]
        + 0.2 * df["gender"]
    )
    probability = 1 / (1 + np.exp(-risk_score))
    df[TARGET_COLUMN] = np.random.default_rng(seed + 1).binomial(1, probability)
    return df


def train_model(dataset: pd.DataFrame | None = None):
    if dataset is None:
        dataset = generate_synthetic_dataset()

    X = dataset[FEATURE_COLUMNS]
    y = dataset[TARGET_COLUMN]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=1000, random_state=42),
    )
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    report = classification_report(y_test, predictions, output_dict=True)

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)

    return model, {"accuracy": float(accuracy), "classification_report": report}


def load_model(path: Union[str, Path] = MODEL_PATH):
    return joblib.load(path)


def predict_diabetes(model, patient: Union[Dict[str, float], List[Dict[str, float]]]):
    if isinstance(patient, dict):
        patient_frame = pd.DataFrame([patient], columns=FEATURE_COLUMNS)
    else:
        patient_frame = pd.DataFrame(patient, columns=FEATURE_COLUMNS)

    if "gender" not in patient_frame.columns:
        patient_frame["gender"] = 0
    patient_frame["gender"] = patient_frame["gender"].apply(lambda x: 1 if str(x).lower() in {"male", "m", "1", "true"} else 0)

    probability = model.predict_proba(patient_frame)[:, 1]
    result = []
    for score in probability:
        result.append(
            {
                "risk_probability": float(score),
                "risk_label": "High risk" if score >= 0.5 else "Lower risk",
            }
        )
    return result[0] if isinstance(patient, dict) else result


if __name__ == "__main__":
    model, metrics = train_model()
    print(f"Model trained and saved to {MODEL_PATH}")
    print(f"Accuracy: {metrics['accuracy']:.3f}")
    sample_patient = {
        "pregnancies": 2,
        "glucose": 180,
        "blood_pressure": 90,
        "skin_thickness": 35,
        "insulin": 140,
        "bmi": 33.5,
        "age": 45,
    }
    print("Example prediction:", predict_diabetes(model, sample_patient))
