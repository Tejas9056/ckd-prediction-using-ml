# Chronic Kidney Disease Prediction App

This application uses machine learning to predict the risk of chronic kidney disease (CKD) based on medical parameters.

## Setup Instructions

1. Create a virtual environment:
```
python -m venv .venv
```

2. Activate the virtual environment:
   - Windows: `.venv\Scripts\activate`
   - macOS/Linux: `source .venv/bin/activate`

3. Install the required dependencies:
```
pip install -r requirements.txt
```

4. Run the application:
```
python app.py
```

5. Open your web browser and navigate to `http://127.0.0.1:5000/`

## Files Description

- `app.py` - The Flask web application
- `main.py` - The machine learning model training script
- `cleaned_ckd_data.csv` - The dataset used for training
- `templates/index.html` - The frontend interface
- `best_model.joblib` & `scaler.joblib` - The saved model and scaler (generated on first run)

## Usage

Fill in the medical parameters in the form and click "Predict" to get an assessment of CKD risk.

## About the Application

This application uses machine learning to predict chronic kidney disease risk based on patient data. The system trains and compares multiple models (Random Forest, SVM, Decision Tree) and selects the best performing one for predictions.

### Features:
- User-friendly web interface for entering patient data
- Instant prediction with probability visualization
- Utilizes medical parameters like blood pressure, specific gravity, albumin, etc.

## Dataset

The application uses the cleaned CKD dataset (`cleaned_ckd_data.csv`) containing various medical parameters and disease classifications. 