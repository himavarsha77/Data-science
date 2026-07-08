# Healthcare Diagnosis Support

This project trains a simple diabetes-risk prediction model from patient health metrics such as glucose level, BMI, age, and blood pressure.

## Files
- `diabetes_model.py`: generates a synthetic dataset, trains a logistic regression model, saves it, and supports prediction.
- `app.py`: command-line interface for scoring a patient record.
- `tests/test_model.py`: basic validation test.

## Usage
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Train the model:
   ```bash
   python diabetes_model.py
   ```
3. Run the command-line predictor:
   ```bash
   python app.py --glucose 180 --bmi 33.5 --age 45
   ```
4. Launch the frontend UI:
   ```bash
   streamlit run streamlit_app.py
   ```
