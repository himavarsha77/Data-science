import json
from pathlib import Path

import streamlit as st

from diabetes_model import MODEL_PATH, load_model, predict_diabetes


@st.cache_resource
def get_model():
    model_path = Path(MODEL_PATH)
    if not model_path.exists():
        raise FileNotFoundError("Model file not found. Run 'python diabetes_model.py' first.")
    return load_model(model_path)


def build_patient_payload(**kwargs):
    return {
        "pregnancies": kwargs["pregnancies"],
        "glucose": kwargs["glucose"],
        "blood_pressure": kwargs["blood_pressure"],
        "skin_thickness": kwargs["skin_thickness"],
        "insulin": kwargs["insulin"],
        "bmi": kwargs["bmi"],
        "age": kwargs["age"],
        "gender": 1 if kwargs["gender"].lower() == "male" else 0,
    }


def main():
    st.set_page_config(page_title="Diabetes Risk Checker", page_icon="🩺", layout="centered")
    st.title("Healthcare Diagnosis Support")
    st.write("Enter patient health metrics to estimate diabetes risk using a Python machine learning model.")

    with st.form("patient_form"):
        gender = st.selectbox("Gender", ["Female", "Male"], index=0)
        pregnancies = st.number_input("Pregnancies", min_value=0, max_value=20, value=2)
        glucose = st.number_input("Glucose", min_value=50, max_value=250, value=140)
        blood_pressure = st.number_input("Blood Pressure", min_value=40, max_value=180, value=85)
        skin_thickness = st.number_input("Skin Thickness", min_value=5, max_value=100, value=25)
        insulin = st.number_input("Insulin", min_value=20, max_value=800, value=120)
        bmi = st.number_input("BMI", min_value=10.0, max_value=70.0, value=28.0, step=0.1)
        age = st.number_input("Age", min_value=18, max_value=100, value=40)
        submitted = st.form_submit_button("Check Risk")

    if submitted:
        model = get_model()
        payload = build_patient_payload(
            pregnancies=pregnancies,
            glucose=glucose,
            blood_pressure=blood_pressure,
            skin_thickness=skin_thickness,
            insulin=insulin,
            bmi=bmi,
            age=age,
            gender=gender,
        )
        result = predict_diabetes(model, payload)
        st.success(f"Risk probability: {result['risk_probability']:.2%}")
        st.info(f"Assessment: {result['risk_label']}")
        with st.expander("Patient data"):
            st.json(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
