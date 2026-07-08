import argparse
import json
from pathlib import Path

from diabetes_model import MODEL_PATH, load_model, predict_diabetes


def parse_args():
    parser = argparse.ArgumentParser(description="Predict diabetes risk from health metrics")
    parser.add_argument("--pregnancies", type=float, default=2)
    parser.add_argument("--glucose", type=float, required=True)
    parser.add_argument("--blood-pressure", type=float, default=80)
    parser.add_argument("--skin-thickness", type=float, default=20)
    parser.add_argument("--insulin", type=float, default=100)
    parser.add_argument("--bmi", type=float, required=True)
    parser.add_argument("--diabetes-pedigree-function", type=float, default=0.5)
    parser.add_argument("--age", type=float, required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    model_path = Path(MODEL_PATH)
    if not model_path.exists():
        raise FileNotFoundError(f"Trained model not found at {model_path}. Run python diabetes_model.py first.")

    model = load_model(model_path)
    patient = {
        "pregnancies": args.pregnancies,
        "glucose": args.glucose,
        "blood_pressure": args.blood_pressure,
        "skin_thickness": args.skin_thickness,
        "insulin": args.insulin,
        "bmi": args.bmi,
        "diabetes_pedigree_function": args.diabetes_pedigree_function,
        "age": args.age,
    }
    print(json.dumps(predict_diabetes(model, patient), indent=2))


if __name__ == "__main__":
    main()
