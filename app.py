from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import LabelEncoder

app = Flask(__name__)

# Load the model and scaler
try:
    model = joblib.load('best_model.joblib')
    scaler = joblib.load('scaler.joblib')
    
    # Identify model classes
    if hasattr(model, 'classes_'):
        print(f"Model classes: {model.classes_}")
        # Check if model has 3 classes (likely due to how 'ckd' was encoded in the dataset)
        if len(model.classes_) == 3:
            print("Model has 3 classes - adjusting CKD classification mapping")
            # We'll assume class 0 is "ckd" based on observed behavior
            CKD_CLASS = 0
            NON_CKD_CLASS = 1
        else:
            # Standard binary classification
            CKD_CLASS = 1
            NON_CKD_CLASS = 0
    else:
        CKD_CLASS = 1
        NON_CKD_CLASS = 0
        
except FileNotFoundError:
    print("Warning: Model files not found. Training will be executed first.")
    import main
    model = joblib.load('best_model.joblib')
    scaler = joblib.load('scaler.joblib')
    # Default class mappings
    CKD_CLASS = 1
    NON_CKD_CLASS = 0

# Define categorical columns (same as in main.py)
categorical_cols = ['rbc', 'pc', 'pcc', 'ba', 'htn', 'dm', 'cad', 'appet', 'pe', 'ane']

# Initialize label encoders
label_encoders = {}
for col in categorical_cols:
    label_encoders[col] = LabelEncoder()
    if col == 'rbc':
        label_encoders[col].fit(['normal', 'abnormal'])
    elif col in ['pc']:
        label_encoders[col].fit(['normal', 'abnormal'])
    elif col in ['pcc', 'ba']:
        label_encoders[col].fit(['present', 'not present'])
    elif col in ['htn', 'dm', 'cad', 'pe', 'ane']:
        label_encoders[col].fit(['yes', 'no'])
    elif col == 'appet':
        label_encoders[col].fit(['good', 'poor'])

# Define feature columns (make sure they match the training data)
feature_cols = ['age', 'bp', 'sg', 'al', 'su', 'rbc', 'pc', 'pcc', 'ba', 
                'bgr', 'bu', 'sc', 'sod', 'pot', 'hemo', 'pcv', 'wc', 'rc', 
                'htn', 'dm', 'cad', 'appet', 'pe', 'ane']

@app.route('/')
def home():
    return render_template('index.html', prediction=None)

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Get form data
        patient_data = {}
        for col in feature_cols:
            if col in categorical_cols:
                patient_data[col] = request.form.get(col)
            else:
                # Convert numerical values to float
                patient_data[col] = float(request.form.get(col))
        
        # Convert to DataFrame
        input_df = pd.DataFrame([patient_data])
        
        # Encode categorical variables
        for col in categorical_cols:
            input_df[col] = label_encoders[col].transform(input_df[col])
        
        # Scale the input data
        input_scaled = scaler.transform(input_df)
        
        # Make prediction
        prediction = model.predict(input_scaled)
        
        # Get detailed probability scores
        raw_probas = model.predict_proba(input_scaled)[0]
        
        # Get the correct probabilities based on identified classes
        ckd_index = np.where(model.classes_ == CKD_CLASS)[0][0]
        
        # Convert probabilities to percentages
        ckd_prob = raw_probas[ckd_index] * 100
        # Calculate non-CKD probability by subtracting from 100%
        non_ckd_prob = 100 - ckd_prob       
      
        # Round probabilities to prevent 100.0% display issues
        if ckd_prob > 99.9:
            ckd_prob = 99.9
            non_ckd_prob = 0.1
        if non_ckd_prob > 99.9:
            non_ckd_prob = 99.9
            ckd_prob = 0.1
        probability = max(ckd_prob, non_ckd_prob)
        
        # Determine if it's CKD based on class, not probability
        is_ckd = prediction[0] == CKD_CLASS
        
        # Return the prediction and probabilities
        return render_template('index.html', 
                               prediction=[is_ckd], 
                               ckd_prob=ckd_prob, 
                               non_ckd_prob=non_ckd_prob, 
                               probability=probability)

if __name__ == "__main__":
    app.run(debug=True) 