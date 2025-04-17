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
    <meta name="description" content="Chronic Kidney Disease Risk Assessment Tool for Healthcare Professionals">
    <title>CKD Risk Assessment | Clinical Prediction Tool</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
    <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bootstrap-icons@1.11.1/font/bootstrap-icons.css">
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">
    
    <style>
        body {
            background: linear-gradient(120deg, #e9eef6, #d1dbed);
            min-height: 100vh;
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, sans-serif;
            color: #4a5568;
            padding-top: 20px;
            padding-bottom: 20px;
        }
        .container {
            background: #fff;
            padding: 30px;
            border-radius: 12px;
            box-shadow: 0 8px 30px rgba(0, 0, 0, 0.12);
            width: 90%;
            max-width: 1200px;
            margin: 20px auto;
        }
        .app-header {
            margin-bottom: 25px;
            border-bottom: 1px solid #edf2f7;
            padding-bottom: 20px;
        }
        h2 {
            font-weight: 600;
            color: #3a5999;
            font-size: 1.8rem;
        }
        .form-control, .form-select {
            background: #f8f9fa;
            border: 1px solid #e2e8f0;
            border-radius: 6px;
            padding: 10px 15px;
            color: #4a5568;
            transition: all 0.2s ease;
        }
        .form-control:focus, .form-select:focus {
            background: #fff;
            border-color: #a0aec0;
            box-shadow: 0 0 0 0.25rem rgba(66, 103, 153, 0.15);
        }
        .form-label {
            font-weight: 500;
            color: #4a5568;
            margin-bottom: 0.5rem;
        }
        .btn-primary {
            background: #3a5999;
            border: none;
            font-weight: 600;
            padding: 12px 20px;
            border-radius: 6px;
            transition: 0.3s;
            box-shadow: 0 2px 5px rgba(58, 89, 153, 0.2);
        }
        .btn-primary:hover {
            background: #344e87;
            transform: translateY(-2px);
            box-shadow: 0 4px 8px rgba(58, 89, 153, 0.3);
        }
        #result {
            font-size: 1.1rem;
            margin-top: 25px;
            padding: 20px;
            border-radius: 8px;
            background-color: #f8fafc;
            border: 1px solid #edf2f7;
            text-align: center;
        }
        .result-high {
            background-color: #fff5f7 !important;
            border-color: #fed7e2 !important;
            color: #b83280;
        }
        .result-low {
            background-color: #f0fff4 !important;
            border-color: #c6f6d5 !important;
            color: #38a169;
        }
        .probability-bar {
            height: 16px;
            background: #edf2f7;
            border-radius: 10px;
            margin: 15px 0;
            position: relative;
            overflow: hidden;
        }
        .probability-fill {
            height: 100%;
            background: linear-gradient(to right, #38a169, #b83280);
            border-radius: 10px;
            transition: width 1s ease-in-out;
        }
        .probability-marker {
            position: absolute;
            width: 3px;
            height: 24px;
            background: #333;
            top: -4px;
            transform: translateX(-50%);
            z-index: 2;
        }
        .probability-labels {
            display: flex;
            justify-content: space-between;
            margin-top: 8px;
            font-size: 0.9rem;
            color: #777;
        }
        .card {
            background-color: #fff;
            border: none;
            box-shadow: 0 2px 8px rgba(0, 0, 0, 0.05);
        }
        .card-body {
            padding: 25px;
        }
        .form-section {
            background-color: #ffffff;
            border-radius: 8px;
            padding: 20px;
            margin-bottom: 20px;
            border: 1px solid #e2e8f0;
            box-shadow: 0 2px 4px rgba(0, 0, 0, 0.02);
        }
        .form-section-title {
            color: #3a5999;
            font-weight: 600;
            margin-bottom: 15px;
            border-bottom: 1px solid #edf2f7;
            padding-bottom: 10px;
        }
        .input-with-help {
            position: relative;
        }
        .help-icon {
            position: absolute;
            right: 10px;
            top: 10px;
            color: #6c757d;
            cursor: pointer;
        }
        .parameter-info {
            font-size: 0.75rem;
            color: #6c757d;
            margin-top: 0.25rem;
        }
        .progress-indicator {
            display: flex;
            justify-content: space-between;
            margin-bottom: 30px;
        }
        .progress-step {
            flex: 1;
            text-align: center;
            padding: 10px;
            position: relative;
        }
        .progress-step::after {
            content: '';
            position: absolute;
            width: 100%;
            height: 3px;
            background: #e9ecef;
            top: 50%;
            left: 50%;
            z-index: -1;
        }
        .progress-step:last-child::after {
            display: none;
        }
        .step-number {
            width: 30px;
            height: 30px;
            border-radius: 50%;
            background: #e9ecef;
            display: inline-flex;
            align-items: center;
            justify-content: center;
            font-weight: bold;
            margin-bottom: 5px;
        }
        .step-active .step-number {
            background: #2c73d2;
            color: white;
        }
        .preset-buttons {
            display: flex;
            justify-content: flex-end;
            gap: 10px;
            margin-bottom: 20px;
        }
        .btn-risk {
            padding: 8px 14px;
            border-radius: 50px;
            font-size: 0.85rem;
            font-weight: 500;
            transition: all 0.3s ease;
            border: none;
            box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        }
        .btn-risk:hover {
            transform: translateY(-1px);
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }
        .btn-risk-high {
            background-color: #fff5f7;
            color: #b83280;
            border: 1px solid #fed7e2;
        }
        .btn-risk-medium {
            background-color: #fffaf0;
            color: #dd6b20;
            border: 1px solid #feebc8;
        }
        .btn-risk-low {
            background-color: #f0fff4;
            color: #38a169;
            border: 1px solid #c6f6d5;
        }
        @media (max-width: 768px) {
            .container {
                width: 95%;
                padding: 15px;
            }
            .preset-buttons {
                flex-direction: column;
                align-items: flex-end;
            }
        }
        
        /* Modal styles */
        .modal-content {
            border: none;
            border-radius: 12px;
            box-shadow: 0 10px 30px rgba(0, 0, 0, 0.1);
        }
        .modal-header {
            background-color: #f8fafc;
            border-bottom: 1px solid #edf2f7;
            border-radius: 12px 12px 0 0;
        }
        .modal-title {
            color: #3a5999;
            font-weight: 600;
        }
        .modal-body {
            padding: 1.5rem;
        }
        .modal-footer {
            border-top: 1px solid #edf2f7;
            background-color: #f8fafc;
            border-radius: 0 0 12px 12px;
        }
        .mb-5 {
            margin-bottom: 2.5rem !important;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="app-header mb-5">
            <h2 class="text-center">Chronic Kidney Disease Risk Assessment</h2>
            <p class="text-center text-muted">Enter patient data to predict the risk of chronic kidney disease</p>
        </div>
        
        <div class="card shadow-sm mb-5">
            <div class="card-body">
                <div class="preset-buttons">
                    <button type="button" class="btn btn-sm btn-risk btn-risk-low" id="lowRiskBtn">Low Risk Sample</button>
                    <button type="button" class="btn btn-sm btn-risk btn-risk-medium" id="mediumRiskBtn">Medium Risk Sample</button>
                    <button type="button" class="btn btn-sm btn-risk btn-risk-high" id="highRiskBtn">High Risk Sample</button>
                </div>
                
                <form id="ckd-form" method="POST" action="/predict">
                    <!-- Numerical Inputs -->
                    <div class="form-section">
                        <h4 class="form-section-title">Patient Information & Vital Signs</h4>
                        <div class="row">
                            <div class="col-md-4">
                                <div class="mb-3 input-with-help">
                                    <label for="age" class="form-label">Age</label>
                                    <input type="number" class="form-control" name="age" id="age" required>
                                    <div class="parameter-info">Patient's age in years</div>
                                </div>
                            </div>
                            <div class="col-md-4">
                                <div class="mb-3 input-with-help">
                                    <label for="bp" class="form-label">Blood Pressure</label>
                                    <input type="number" class="form-control" name="bp" id="bp" required>
                                    <div class="parameter-info">mm/Hg</div>
                                </div>
                            </div>
                            <div class="col-md-4">
                                <div class="mb-3 input-with-help">
                                    <label for="sg" class="form-label">Specific Gravity</label>
                                    <input type="number" step="0.001" class="form-control" name="sg" id="sg" required>
                                    <div class="parameter-info">Range: 1.005-1.025</div>
                                </div>
                            </div>
                        </div>
                    </div>

                    <div class="form-section">
                        <h4 class="form-section-title">Urinalysis Results</h4>
                        <div class="row">
                            <div class="col-md-3">
                                <div class="mb-3 input-with-help">
                                    <label for="al" class="form-label">Albumin</label>
                                    <input type="number" class="form-control" name="al" id="al" required>
                                    <div class="parameter-info">0-5 scale</div>
                                </div>
                            </div>
                            <div class="col-md-3">
                                <div class="mb-3 input-with-help">
                                    <label for="su" class="form-label">Sugar</label>
                                    <input type="number" class="form-control" name="su" id="su" required>
                                    <div class="parameter-info">0-5 scale</div>
                                </div>
                            </div>
                            <div class="col-md-3">
                                <div class="mb-3">
                                    <label for="rbc" class="form-label">Red Blood Cells</label>
                                    <select class="form-select" name="rbc" id="rbc" required>
                                        <option value="normal">Normal</option>
                                        <option value="abnormal">Abnormal</option>
                                    </select>
                                </div>
                            </div>
                            <div class="col-md-3">
                                <div class="mb-3">
                                    <label for="pc" class="form-label">Pus Cell</label>
                                    <select class="form-select" name="pc" id="pc" required>
                                        <option value="normal">Normal</option>
                                        <option value="abnormal">Abnormal</option>
                                    </select>
                                </div>
                            </div>
                        </div>
                        <div class="row">
                            <div class="col-md-3">
                                <div class="mb-3">
                                    <label for="pcc" class="form-label">Pus Cell Clumps</label>
                                    <select class="form-select" name="pcc" id="pcc" required>
                                        <option value="present">Present</option>
                                        <option value="not present">Not Present</option>
                                    </select>
                                </div>
                            </div>
                            <div class="col-md-3">
                                <div class="mb-3">
                                    <label for="ba" class="form-label">Bacteria</label>
                                    <select class="form-select" name="ba" id="ba" required>
                                        <option value="present">Present</option>
                                        <option value="not present">Not Present</option>
                                    </select>
                                </div>
                            </div>
                        </div>
                    </div>

                    <div class="form-section">
                        <h4 class="form-section-title">Blood Test Results</h4>
                        <div class="row">
                            <div class="col-md-3">
                                <div class="mb-3 input-with-help">
                                    <label for="bgr" class="form-label">Blood Glucose Random</label>
                                    <input type="number" class="form-control" name="bgr" id="bgr" required>
                                    <div class="parameter-info">mg/dL</div>
                                </div>
                            </div>
                            <div class="col-md-3">
                                <div class="mb-3 input-with-help">
                                    <label for="bu" class="form-label">Blood Urea</label>
                                    <input type="number" step="0.1" class="form-control" name="bu" id="bu" required>
                                    <div class="parameter-info">mg/dL</div>
                                </div>
                            </div>
                            <div class="col-md-3">
                                <div class="mb-3 input-with-help">
                                    <label for="sc" class="form-label">Serum Creatinine</label>
                                    <input type="number" step="0.1" class="form-control" name="sc" id="sc" required>
                                    <div class="parameter-info">mg/dL</div>
                                </div>
                            </div>
                            <div class="col-md-3">
                                <div class="mb-3 input-with-help">
                                    <label for="sod" class="form-label">Sodium</label>
                                    <input type="number" class="form-control" name="sod" id="sod" required>
                                    <div class="parameter-info">mEq/L</div>
                                </div>
                            </div>
                        </div>
                        <div class="row">
                            <div class="col-md-3">
                                <div class="mb-3 input-with-help">
                                    <label for="pot" class="form-label">Potassium</label>
                                    <input type="number" step="0.1" class="form-control" name="pot" id="pot" required>
                                    <div class="parameter-info">mEq/L</div>
                                </div>
                            </div>
                            <div class="col-md-3">
                                <div class="mb-3 input-with-help">
                                    <label for="hemo" class="form-label">Hemoglobin</label>
                                    <input type="number" step="0.1" class="form-control" name="hemo" id="hemo" required>
                                    <div class="parameter-info">g/dL</div>
                                </div>
                            </div>
                            <div class="col-md-3">
                                <div class="mb-3 input-with-help">
                                    <label for="pcv" class="form-label">Packed Cell Volume</label>
                                    <input type="number" class="form-control" name="pcv" id="pcv" required>
                                    <div class="parameter-info">%</div>
                                </div>
                            </div>
                            <div class="col-md-3">
                                <div class="mb-3 input-with-help">
                                    <label for="wc" class="form-label">White Blood Cell Count</label>
                                    <input type="number" class="form-control" name="wc" id="wc" required>
                                    <div class="parameter-info">cells/cumm</div>
                                </div>
                            </div>
                        </div>
                        <div class="row">
                            <div class="col-md-3">
                                <div class="mb-3 input-with-help">
                                    <label for="rc" class="form-label">Red Blood Cell Count</label>
                                    <input type="number" step="0.1" class="form-control" name="rc" id="rc" required>
                                    <div class="parameter-info">millions/cmm</div>
                                </div>
                            </div>
                        </div>
                    </div>

                    <div class="form-section">
                        <h4 class="form-section-title">Medical History & Symptoms</h4>
                        <div class="row">
                            <div class="col-md-3">
                                <div class="mb-3">
                                    <label for="htn" class="form-label">Hypertension</label>
                                    <select class="form-select" name="htn" id="htn" required>
                                        <option value="yes">Yes</option>
                                        <option value="no">No</option>
                                    </select>
                                </div>
                            </div>
                            <div class="col-md-3">
                                <div class="mb-3">
                                    <label for="dm" class="form-label">Diabetes Mellitus</label>
                                    <select class="form-select" name="dm" id="dm" required>
                                        <option value="yes">Yes</option>
                                        <option value="no">No</option>
                                    </select>
                                </div>
                            </div>
                            <div class="col-md-3">
                                <div class="mb-3">
                                    <label for="cad" class="form-label">Coronary Artery Disease</label>
                                    <select class="form-select" name="cad" id="cad" required>
                                        <option value="yes">Yes</option>
                                        <option value="no">No</option>
                                    </select>
                                </div>
                            </div>
                            <div class="col-md-3">
                                <div class="mb-3">
                                    <label for="appet" class="form-label">Appetite</label>
                                    <select class="form-select" name="appet" id="appet" required>
                                        <option value="good">Good</option>
                                        <option value="poor">Poor</option>
                                    </select>
                                </div>
                            </div>
                        </div>
                        <div class="row">
                            <div class="col-md-3">
                                <div class="mb-3">
                                    <label for="pe" class="form-label">Pedal Edema</label>
                                    <select class="form-select" name="pe" id="pe" required>
                                        <option value="yes">Yes</option>
                                        <option value="no">No</option>
                                    </select>
                                </div>
                            </div>
                            <div class="col-md-3">
                                <div class="mb-3">
                                    <label for="ane" class="form-label">Anemia</label>
                                    <select class="form-select" name="ane" id="ane" required>
                                        <option value="yes">Yes</option>
                                        <option value="no">No</option>
                                    </select>
                                </div>
                            </div>
                        </div>
                    </div>

                    <div class="text-center">
                        <button type="submit" class="btn btn-primary px-5">Analyze Risk</button>
                    </div>
                </form>
                
                {% if prediction is not none %}
                <div id="result" class="{% if prediction[0] == 1 %}result-high{% else %}result-low{% endif %} mt-5 card shadow-sm">
                    <div class="card-body">
                        <div class="mb-3">
                            <h4>
                            {% if prediction[0] == 1 %}
                                <i class="bi bi-exclamation-triangle-fill me-2"></i> High Risk of Chronic Kidney Disease
                            {% else %}
                                <i class="bi bi-shield-check me-2"></i> Risk of Chronic Kidney Disease
                            {% endif %}
                            </h4>
                        </div>
                        
                        <div class="mt-4">
                            <div class="probability-bar">
                                <div class="probability-fill" style="width: {{ "%.1f"|format(non_ckd_prob) }}%;"></div>
                                <div class="probability-marker" style="left: {{ ckd_prob }}%;"></div>
                            </div>
                            <div class="probability-labels">
                                <span>0%</span>
                                <span>100%</span>
                            </div>
                            <div class="text-center mt-3">
                                <strong>CKD Risk: {{ "%.1f"|format(non_ckd_prob) }}%</strong>
                            </div>
                        </div>
                    </div>
                </div>
                {% endif %}

                <!-- Add this modal right before closing the container div -->
                <div class="modal fade" id="resultModal" tabindex="-1" aria-labelledby="resultModalLabel" aria-hidden="true">
                    <div class="modal-dialog modal-dialog-centered">
                        <div class="modal-content">
                            <div class="modal-header">
                                <h5 class="modal-title" id="resultModalLabel">CKD Risk Assessment Result</h5>
                                <button type="button" class="btn-close" data-bs-dismiss="modal" aria-label="Close"></button>
                            </div>
                            <div class="modal-body">
                                {% if prediction is not none %}
                                <div class="mb-3">
                                    <h4 class="text-center {% if prediction[0] == 1 %}text-danger{% else %}text-primary{% endif %}">
                                    {% if prediction[0] == 1 %}
                                        <i class="bi bi-exclamation-triangle-fill me-2"></i> High Risk of Chronic Kidney Disease
                                    {% else %}
                                        <i class="bi bi-shield-check me-2"></i> Risk of Chronic Kidney Disease
                                    {% endif %}
                                    </h4>
                                </div>
                                
                                <div class="mt-4">
                                    <div class="probability-bar">
                                        <div class="probability-fill" style="width: {{ "%.1f"|format(non_ckd_prob) }}%;"></div>
                                        <div class="probability-marker" style="left: {{ ckd_prob }}%;"></div>
                                    </div>
                                    <div class="probability-labels">
                                        <span>0%</span>
                                        <span>100%</span>
                                    </div>
                                    <div class="text-center mt-3">
                                        <strong>CKD Risk: {{ "%.1f"|format(non_ckd_prob) }}%</strong>
                                    </div>
                                </div>
                                {% endif %}
                            </div>
                            <div class="modal-footer">
                                <button type="button" class="btn btn-secondary" data-bs-dismiss="modal">Close</button>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    </div>

    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/js/bootstrap.bundle.min.js"></script>
    <script>
        document.addEventListener('DOMContentLoaded', function() {
            // Initialize tooltips
            const tooltipTriggerList = document.querySelectorAll('[data-bs-toggle="tooltip"]');
            const tooltipList = [...tooltipTriggerList].map(tooltipTriggerEl => new bootstrap.Tooltip(tooltipTriggerEl));
            
            // Show result modal if prediction exists
            {% if prediction is not none %}
            setTimeout(() => {
                const resultModal = new bootstrap.Modal(document.getElementById('resultModal'));
                resultModal.show();
                
                // Animate probability bars in both modal and main view after modal is shown
                const modalShown = document.getElementById('resultModal');
                modalShown.addEventListener('shown.bs.modal', function() {
                    const fillElements = document.querySelectorAll('.probability-fill');
                    fillElements.forEach(function(fillElement) {
                        const currentWidth = fillElement.style.width;
                        fillElement.style.width = '0%';
                        
                        setTimeout(() => {
                            fillElement.style.transition = 'width 1s ease-in-out';
                            fillElement.style.width = currentWidth;
                        }, 200);
                    });
                });
            }, 500);
            {% endif %}
            
            // Add validation classes on form submission
            const form = document.getElementById('ckd-form');
            if (form) {
                form.addEventListener('submit', function(event) {
                    if (!form.checkValidity()) {
                        event.preventDefault();
                        event.stopPropagation();
                    }
                    form.classList.add('was-validated');
                });
            }
            
            // Animate probability bar on result display
            const resultElement = document.getElementById('result');
            if (resultElement) {
                setTimeout(() => {
                    const fillElement = document.querySelector('.probability-fill');
                    if (fillElement) {
                        fillElement.style.transition = 'width 1s ease-in-out';
                    }
                }, 100);
            }
            
            // High Risk patient values
            document.getElementById('highRiskBtn').addEventListener('click', function() {
                // Extreme values known to trigger high CKD risk
                document.getElementById('age').value = '57';
                document.getElementById('bp').value = '90';
                document.getElementById('sg').value = '1.015';
                document.getElementById('al').value = '5';
                document.getElementById('su').value = '0';
                document.getElementById('bgr').value = '121';
                document.getElementById('bu').value = '322';
                document.getElementById('sc').value = '13.0';
                document.getElementById('sod').value = '126';
                document.getElementById('pot').value = '4.8';
                document.getElementById('hemo').value = '8.0';
                document.getElementById('pcv').value = '24';
                document.getElementById('wc').value = '4200';
                document.getElementById('rc').value = '3.3';
                
                // Set select options
                selectOption('rbc', 'abnormal');
                selectOption('pc', 'abnormal');
                selectOption('pcc', 'not present');
                selectOption('ba', 'present');
                selectOption('htn', 'yes');
                selectOption('dm', 'yes');
                selectOption('cad', 'yes');
                selectOption('appet', 'poor');
                selectOption('pe', 'yes');
                selectOption('ane', 'yes');
                
                highlightButton('highRiskBtn');
            });
            
            // Medium Risk patient values
            document.getElementById('mediumRiskBtn').addEventListener('click', function() {
                // Values that should give a less certain prediction
                document.getElementById('age').value = '48';
                document.getElementById('bp').value = '70';
                document.getElementById('sg').value = '1.020';
                document.getElementById('al').value = '1';
                document.getElementById('su').value = '0';
                document.getElementById('bgr').value = '150';
                document.getElementById('bu').value = '36';
                document.getElementById('sc').value = '1.8';
                document.getElementById('sod').value = '137';
                document.getElementById('pot').value = '4.3';
                document.getElementById('hemo').value = '13.8';
                document.getElementById('pcv').value = '44';
                document.getElementById('wc').value = '8000';
                document.getElementById('rc').value = '4.5';
                
                // Set select options - fewer risk factors than high risk
                selectOption('rbc', 'normal');
                selectOption('pc', 'normal');
                selectOption('pcc', 'not present');
                selectOption('ba', 'not present');
                selectOption('htn', 'yes');
                selectOption('dm', 'no');
                selectOption('cad', 'no');
                selectOption('appet', 'good');
                selectOption('pe', 'no');
                selectOption('ane', 'no');
                
                highlightButton('mediumRiskBtn');
            });
            
            // Low Risk patient values
            document.getElementById('lowRiskBtn').addEventListener('click', function() {
                // Values characteristic of a healthy person
                document.getElementById('age').value = '40';
                document.getElementById('bp').value = '80';
                document.getElementById('sg').value = '1.025';
                document.getElementById('al').value = '0';
                document.getElementById('su').value = '0';
                document.getElementById('bgr').value = '140';
                document.getElementById('bu').value = '10';
                document.getElementById('sc').value = '1.2';
                document.getElementById('sod').value = '150';
                document.getElementById('pot').value = '4.6';
                document.getElementById('hemo').value = '17.0';
                document.getElementById('pcv').value = '48';
                document.getElementById('wc').value = '10400';
                document.getElementById('rc').value = '4.5';
                
                // Set select options - all healthy
                selectOption('rbc', 'normal');
                selectOption('pc', 'normal');
                selectOption('pcc', 'not present');
                selectOption('ba', 'not present');
                selectOption('htn', 'no');
                selectOption('dm', 'no');
                selectOption('cad', 'no');
                selectOption('appet', 'good');
                selectOption('pe', 'no');
                selectOption('ane', 'no');
                
                highlightButton('lowRiskBtn');
            });
            
            // Helper function to select an option in a dropdown
            function selectOption(elementId, value) {
                const select = document.getElementById(elementId);
                if (!select) return;
                
                for (let i = 0; i < select.options.length; i++) {
                    if (select.options[i].value === value) {
                        select.options[i].selected = true;
                        break;
                    }
                }
            }
            
            // Highlight the active sample button
            function highlightButton(activeButtonId) {
                const buttons = ['lowRiskBtn', 'mediumRiskBtn', 'highRiskBtn'];
                
                buttons.forEach(btnId => {
                    const btn = document.getElementById(btnId);
                    if (btnId === activeButtonId) {
                        btn.classList.add('fw-bold');
                        btn.style.transform = 'scale(1.05)';
                        btn.style.boxShadow = '0 4px 8px rgba(0,0,0,0.15)';
                        btn.style.border = '2px solid currentColor';
                    } else {
                        btn.classList.remove('fw-bold');
                        btn.style.transform = 'scale(1)';
                        btn.style.boxShadow = '0 2px 4px rgba(0,0,0,0.05)';
                        btn.style.border = '1px solid';
                    }
                });
                
                // Scroll to form submit button
                document.querySelector('button[type="submit"]').scrollIntoView({ behavior: 'smooth', block: 'center' });
            }
            
            // Add mobile-friendly hover effects for form sections
            const formSections = document.querySelectorAll('.form-section');
            formSections.forEach(section => {
                section.addEventListener('mouseenter', function() {
                    this.style.boxShadow = '0 4px 12px rgba(0, 0, 0, 0.05)';
                    this.style.transform = 'translateY(-2px)';
                    this.style.transition = 'all 0.2s ease';
                });
                
                section.addEventListener('mouseleave', function() {
                    this.style.boxShadow = 'none';
                    this.style.transform = 'translateY(0)';
                });
            });
        });
    </script>
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
