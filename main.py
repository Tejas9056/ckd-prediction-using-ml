# Import required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import joblib

# Load the cleaned dataset
df = pd.read_csv("cleaned_ckd_data.csv")

# Convert all column names to lowercase (to avoid inconsistency)
df.columns = df.columns.str.lower()

# Encode categorical variables
le = LabelEncoder()
categorical_cols = ['rbc', 'pc', 'pcc', 'ba', 'htn', 'dm', 'cad', 'appet', 'pe', 'ane', 'classification']
for col in categorical_cols:
    df[col] = le.fit_transform(df[col])

# Check if all data types are correct
df.info()

# Split features and target
X = df.drop(columns=['classification'])
y = df['classification']

# Standardize numerical features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split into train and test sets (stratified sampling to balance classes)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)

# Define models for comparison
models = {
    "Random Forest": RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42),
    "SVM": SVC(kernel='linear', probability=True),
    "Decision Tree": DecisionTreeClassifier(max_depth=10, random_state=42)
}

# Train models and evaluate accuracy
accuracy_scores = {}
conf_matrices = {}

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy_scores[name] = accuracy_score(y_test, y_pred) * 100  # Convert to percentage
    conf_matrices[name] = confusion_matrix(y_test, y_pred)

# Print accuracy scores
print("Model Accuracies:")
for model, acc in accuracy_scores.items():
    print(f"{model}: {acc:.2f}%")  # Display accuracy in percentage format

# Find the best model based on accuracy
best_model_name = max(accuracy_scores, key=accuracy_scores.get)
best_model = models[best_model_name]
best_cm = conf_matrices[best_model_name]

# Save the best model and scaler
joblib.dump(best_model, 'best_model.joblib')
joblib.dump(scaler, 'scaler.joblib')

print(f"\n✅ Best Model: {best_model_name} with Accuracy: {accuracy_scores[best_model_name]:.2f}%")
print("✅ Model and scaler have been saved for web application use")

# Display the confusion matrix of the best model
plt.figure(figsize=(6, 5))
sns.heatmap(best_cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Not CKD', 'CKD'], yticklabels=['Not CKD', 'CKD'])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title(f"Confusion Matrix - Best Model ({best_model_name})")
plt.show()

# Plot accuracy comparison (Bar Graph)
plt.figure(figsize=(6, 5))
sns.barplot(x=list(accuracy_scores.keys()), y=list(accuracy_scores.values()), palette='viridis')
plt.xlabel("Model")
plt.ylabel("Accuracy (%)")  # Update y-axis label
plt.title("Comparison of Models (Bar Graph)")
plt.ylim(80, 100)  # Adjust y-axis range for better visualization
plt.show()

# Plot accuracy comparison (Line Graph)
plt.figure(figsize=(6, 5))
plt.plot(list(accuracy_scores.keys()), list(accuracy_scores.values()), marker='o', linestyle='-', color='b')
plt.xlabel("Model")
plt.ylabel("Accuracy (%)")  # Update y-axis label
plt.title("Comparison of Models (Line Graph)")
plt.ylim(80, 100)  # Adjust y-axis range
plt.grid()
plt.show()

print(f"\n✅ Best Model: {best_model_name} with Accuracy: {accuracy_scores[best_model_name]:.2f}%")  # Display as percentage
