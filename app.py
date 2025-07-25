from flask import Flask, render_template, request
import pandas as pd
import joblib
import os

app = Flask(__name__)

# Load your trained model pipeline
model_pipeline = joblib.load("readmission_model_pipeline.pkl")

# Feature names and dummy importances
feature_names = ['Age', 'Medical Condition', 'Insurance Provider', 'Admission Type', 'Medication', 'Billing Amount', 'Length of Stay']
feature_importances = [0.2, 0.15, 0.1, 0.1, 0.1, 0.15, 0.2]  # Replace with real values if available

def get_stats():
    if not os.path.exists('prediction_stats.csv'):
        return {'low': 0, 'high': 0}
    df = pd.read_csv('prediction_stats.csv')
    return {
        'low': (df['prediction'] == 0).sum(),
        'high': (df['prediction'] == 1).sum()
    }

@app.route('/')
def home():
    stats = get_stats()
    return render_template("index.html",
                           prediction_text=None,
                           stats=stats,
                           feature_names=feature_names,
                           feature_importances=feature_importances)

@app.route('/predict', methods=['POST'])
def predict():
    input_data = {
        'Age': int(request.form['Age']),
        'Medical Condition': request.form['Medical Condition'],
        'Insurance Provider': request.form['Insurance Provider'],
        'Admission Type': request.form['Admission Type'],
        'Medication': request.form['Medication'],
        'Billing Amount': float(request.form['Billing Amount']),
        'Length of Stay': int(request.form['Length of Stay'])
    }

    input_df = pd.DataFrame([input_data])
    prediction = model_pipeline.predict(input_df)[0]
    prediction_text = "High Risk" if prediction == 1 else "Low Risk"

    # Save prediction to CSV for tracking
    if os.path.exists('prediction_stats.csv'):
        stats_df = pd.read_csv('prediction_stats.csv')
        stats_df = pd.concat([stats_df, pd.DataFrame({'prediction': [prediction]})], ignore_index=True)
    else:
        stats_df = pd.DataFrame({'prediction': [prediction]})
    stats_df.to_csv('prediction_stats.csv', index=False)

    stats = get_stats()

    return render_template("index.html",
                           prediction_text=prediction_text,
                           stats=stats,
                           feature_names=feature_names,
                           feature_importances=feature_importances)

if __name__ == '__main__':
    app.run(debug=True)
