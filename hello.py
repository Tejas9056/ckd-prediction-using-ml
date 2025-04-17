from flask import Flask, request, jsonify, render_template_string
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
import pickle
import joblib

app = Flask(__name__)

# Load the trained model and scaler
try:
    model = joblib.load('best_model.joblib')
    scaler = joblib.load('scaler.joblib')
except:
    # If model files don't exist, train the model
    from main import models, best_model_name, scaler, X
    model = models[best_model_name]
    # Save the model and scaler
    joblib.dump(model, 'best_model.joblib')
    joblib.dump(scaler, 'scaler.joblib')

# Initialize label encoder
le = LabelEncoder()

HTML_TEMPLATE = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>CKD Prediction</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
    
    <style>
        body {
            background: linear-gradient(135deg, #6a11cb, #2575fc);
            min-height: 100vh;
            display: flex;
            justify-content: center;
            align-items: center;
            color: white;
        }
        .container {
            background: rgba(0, 0, 0, 0.7);
            padding: 20px;
            border-radius: 15px;
            box-shadow: 0 4px 10px rgba(0, 0, 0, 0.3);
            width: 50%;
            margin: 20px auto;
        }
        h2 {
            font-weight: bold;
            text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.5);
        }
        .form-control, .form-select {
            background: rgba(255, 255, 255, 0.2);
            color: white;
            border: none;
        }
        .form-control::placeholder {
            color: rgba(255, 255, 255, 0.7);
        }
        .form-control:focus, .form-select:focus {
            background: rgba(255, 255, 255, 0.3);
            color: white;
            border: none;
            box-shadow: 0 0 5px white;
        }
        .btn-primary {
            background: #ff9800;
            border: none;
            font-weight: bold;
            transition: 0.3s;
        }
        .btn-primary:hover {
            background: #e68900;
        }
        #result {
            font-size: 1.2rem;
            font-weight: bold;
            margin-top: 20px;
            padding: 15px;
            border-radius: 10px;
            text-align: center;
        }
        .result-high {
            background-color: rgba(220, 53, 69, 0.7);
        }
        .result-low {
            background-color: rgba(40, 167, 69, 0.7);
        }
        .probability-bar {
            height: 20px;
            background: linear-gradient(to right, #198754, #dc3545);
            border-radius: 10px;
            margin: 10px 0;
            position: relative;
        }
        .probability-marker {
            position: absolute;
            width: 4px;
            height: 30px;
            background: #fff;
            top: -5px;
            transform: translateX(-50%);
        }
        .probability-labels {
            display: flex;
            justify-content: space-between;
            margin-top: 5px;
            font-size: 0.9rem;
            color: #fff;
        }
    </style>
</head>
<body>
    <div class="container">
        <h2 class="text-center mb-4">Chronic Kidney Disease Prediction</h2>
        <form id="ckd-form" method="POST" action="/predict">
            <!-- Numerical Inputs -->
            <div class="row">
                <div class="col-md-6">
                    <div class="mb-3">
                        <label for="age" class="form-label">Age</label>
                        <input type="number" class="form-control" name="age" required>
                    </div>
                    <div class="mb-3">
                        <label for="bp" class="form-label">Blood Pressure</label>
                        <input type="number" class="form-control" name="bp" required>
                    </div>
                    <div class="mb-3">
                        <label for="sg" class="form-label">Specific Gravity</label>
                        <input type="number" step="0.001" class="form-control" name="sg" required>
                    </div>
                    <div class="mb-3">
                        <label for="al" class="form-label">Albumin</label>
                        <input type="number" class="form-control" name="al" required>
                    </div>
                    <div class="mb-3">
                        <label for="su" class="form-label">Sugar</label>
                        <input type="number" class="form-control" name="su" required>
                    </div>
                    <div class="mb-3">
                        <label for="bgr" class="form-label">Blood Glucose Random</label>
                        <input type="number" class="form-control" name="bgr" required>
                    </div>
                    <div class="mb-3">
                        <label for="bu" class="form-label">Blood Urea</label>
                        <input type="number" step="0.1" class="form-control" name="bu" required>
                    </div>
                </div>
                <div class="col-md-6">
                    <div class="mb-3">
                        <label for="sc" class="form-label">Serum Creatinine</label>
                        <input type="number" step="0.1" class="form-control" name="sc" required>
                    </div>
                    <div class="mb-3">
                        <label for="sod" class="form-label">Sodium</label>
                        <input type="number" class="form-control" name="sod" required>
                    </div>
                    <div class="mb-3">
                        <label for="pot" class="form-label">Potassium</label>
                        <input type="number" step="0.1" class="form-control" name="pot" required>
                    </div>
                    <div class="mb-3">
                        <label for="hemo" class="form-label">Hemoglobin</label>
                        <input type="number" step="0.1" class="form-control" name="hemo" required>
                    </div>
                    <div class="mb-3">
                        <label for="pcv" class="form-label">Packed Cell Volume</label>
                        <input type="number" class="form-control" name="pcv" required>
                    </div>
                    <div class="mb-3">
                        <label for="wc" class="form-label">White Blood Cell Count</label>
                        <input type="number" class="form-control" name="wc" required>
                    </div>
                    <div class="mb-3">
                        <label for="rc" class="form-label">Red Blood Cell Count</label>
                        <input type="number" step="0.1" class="form-control" name="rc" required>
                    </div>
                </div>
            </div>

            <!-- Categorical Inputs -->
            <div class="row">
                <div class="col-md-6">
                    <div class="mb-3">
                        <label for="rbc" class="form-label">Red Blood Cells</label>
                        <select class="form-select" name="rbc" required>
                            <option value="normal">Normal</option>
                            <option value="abnormal">Abnormal</option>
                        </select>
                    </div>
                    <div class="mb-3">
                        <label for="pc" class="form-label">Pus Cell</label>
                        <select class="form-select" name="pc" required>
                            <option value="normal">Normal</option>
                            <option value="abnormal">Abnormal</option>
                        </select>
                    </div>
                    <div class="mb-3">
                        <label for="pcc" class="form-label">Pus Cell Clumps</label>
                        <select class="form-select" name="pcc" required>
                            <option value="present">Present</option>
                            <option value="not present">Not Present</option>
                        </select>
                    </div>
                    <div class="mb-3">
                        <label for="ba" class="form-label">Bacteria</label>
                        <select class="form-select" name="ba" required>
                            <option value="present">Present</option>
                            <option value="not present">Not Present</option>
                        </select>
                    </div>
                </div>
                <div class="col-md-6">
                    <div class="mb-3">
                        <label for="htn" class="form-label">Hypertension</label>
                        <select class="form-select" name="htn" required>
                            <option value="yes">Yes</option>
                            <option value="no">No</option>
                        </select>
                    </div>
                    <div class="mb-3">
                        <label for="dm" class="form-label">Diabetes Mellitus</label>
                        <select class="form-select" name="dm" required>
                            <option value="yes">Yes</option>
                            <option value="no">No</option>
                        </select>
                    </div>
                    <div class="mb-3">
                        <label for="cad" class="form-label">Coronary Artery Disease</label>
                        <select class="form-select" name="cad" required>
                            <option value="yes">Yes</option>
                            <option value="no">No</option>
                        </select>
                    </div>
                    <div class="mb-3">
                        <label for="appet" class="form-label">Appetite</label>
                        <select class="form-select" name="appet" required>
                            <option value="good">Good</option>
                            <option value="poor">Poor</option>
                        </select>
                    </div>
                    <div class="mb-3">
                        <label for="pe" class="form-label">Pedal Edema</label>
                        <select class="form-select" name="pe" required>
                            <option value="yes">Yes</option>
                            <option value="no">No</option>
                        </select>
                    </div>
                    <div class="mb-3">
                        <label for="ane" class="form-label">Anemia</label>
                        <select class="form-select" name="ane" required>
                            <option value="yes">Yes</option>
                            <option value="no">No</option>
                        </select>
                    </div>
                </div>
            </div>

            <button type="submit" class="btn btn-primary w-100">Predict</button>
        </form>
        
        {% if prediction is not none %}
        <div id="result" class="{% if prediction[0] == 1 %}result-high{% else %}result-low{% endif %}">
            <div class="mb-3">
                {% if prediction[0] == 1 %}
                    High Risk of Chronic Kidney Disease
                {% else %}
                    Low Risk of Chronic Kidney Disease
                {% endif %}
            </div>
            
            <div class="mt-3">
                <div class="probability-bar">
                    <div class="probability-marker" style="left: {{ ckd_prob }}%;"></div>
                </div>
                <div class="probability-labels">
                    <span>Low Risk ({{ "%.1f"|format(non_ckd_prob) }}%)</span>
                    <span>High Risk ({{ "%.1f"|format(ckd_prob) }}%)</span>
                </div>
                <div class="text-center mt-2">
                    Prediction Confidence: {{ "%.1f"|format(probability) }}%
                </div>
            </div>
        </div>
        {% endif %}
    </div>

    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/js/bootstrap.bundle.min.js"></script>
</body>
</html>
'''

@app.route('/')
def home():
    return render_template_string(HTML_TEMPLATE, 
                                prediction=None,
                                probability=None,
                                ckd_prob=None,
                                non_ckd_prob=None,
                                error=None)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get form data
        data = {
            'age': float(request.form['age']),
            'bp': float(request.form['bp']),
            'sg': float(request.form['sg']),
            'al': float(request.form['al']),
            'su': float(request.form['su']),
            'rbc': request.form['rbc'],
            'pc': request.form['pc'],
            'pcc': request.form['pcc'],
            'ba': request.form['ba'],
            'bgr': float(request.form['bgr']),
            'bu': float(request.form['bu']),
            'sc': float(request.form['sc']),
            'sod': float(request.form['sod']),
            'pot': float(request.form['pot']),
            'hemo': float(request.form['hemo']),
            'pcv': float(request.form['pcv']),
            'wc': float(request.form['wc']),
            'rc': float(request.form['rc']),
            'htn': request.form['htn'],
            'dm': request.form['dm'],
            'cad': request.form['cad'],
            'appet': request.form['appet'],
            'pe': request.form['pe'],
            'ane': request.form['ane']
        }

        # Create DataFrame with ordered columns matching training data
        column_order = ['age', 'bp', 'sg', 'al', 'su', 'rbc', 'pc', 'pcc', 'ba', 'bgr', 
                       'bu', 'sc', 'sod', 'pot', 'hemo', 'pcv', 'wc', 'rc', 
                       'htn', 'dm', 'cad', 'appet', 'pe', 'ane']
        
        input_df = pd.DataFrame([data])[column_order]

        # Encode categorical variables
        categorical_cols = ['rbc', 'pc', 'pcc', 'ba', 'htn', 'dm', 'cad', 'appet', 'pe', 'ane']
        
        # Create a mapping dictionary for each categorical variable
        category_mappings = {
            'rbc': {'normal': 0, 'abnormal': 1},
            'pc': {'normal': 0, 'abnormal': 1},
            'pcc': {'not present': 0, 'present': 1},
            'ba': {'not present': 0, 'present': 1},
            'htn': {'no': 0, 'yes': 1},
            'dm': {'no': 0, 'yes': 1},
            'cad': {'no': 0, 'yes': 1},
            'appet': {'poor': 0, 'good': 1},
            'pe': {'no': 0, 'yes': 1},
            'ane': {'no': 0, 'yes': 1}
        }
        
        # Apply the mappings
        for col in categorical_cols:
            input_df[col] = input_df[col].map(category_mappings[col])

        # Scale numerical features
        input_scaled = scaler.transform(input_df)

        # Make prediction and get probability scores
        prediction = model.predict(input_scaled)
        probabilities = model.predict_proba(input_scaled)[0]
        
        # Calculate confidence percentage
        ckd_probability = probabilities[1] * 100
        non_ckd_probability = probabilities[0] * 100
        confidence = ckd_probability if prediction[0] == 1 else non_ckd_probability

        return render_template_string(HTML_TEMPLATE, 
                                   prediction=prediction,
                                   probability=confidence,
                                   ckd_prob=ckd_probability,
                                   non_ckd_prob=non_ckd_probability,
                                   error=None)

    except Exception as e:
        print("Error occurred:", str(e))
        return render_template_string(HTML_TEMPLATE, 
                                   prediction=None,
                                   probability=None,
                                   ckd_prob=None,
                                   non_ckd_prob=None,
                                   error=str(e))

if __name__ == '__main__':
    app.run(debug=True)