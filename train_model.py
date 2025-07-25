import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import joblib

# Load dataset
df = pd.read_csv("healthcare_dataset.csv")

# Clean up column names (remove leading/trailing whitespace)
df.columns = df.columns.str.strip()


# Convert 'Date of Admission' and 'Discharge Date' to datetime objects
df['Date of Admission'] = pd.to_datetime(df['Date of Admission'])
df['Discharge Date'] = pd.to_datetime(df['Discharge Date'])

# Calculate 'Length of Stay'
df['Length of Stay'] = (df['Discharge Date'] - df['Date of Admission']).dt.days

# === UPDATE THESE COLUMN NAMES IF NEEDED ===
# Using available columns as features and 'Test Results' as a placeholder target
features = ['Age', 'Medical Condition', 'Insurance Provider', 'Admission Type', 'Medication', 'Billing Amount', 'Length of Stay']
target = 'Test Results'  # Placeholder target - please change if needed

# Check that all necessary columns exist
missing = [col for col in features + [target] if col not in df.columns]
if missing:
    raise ValueError(f"Missing columns in dataset: {missing}")

# Drop missing values in selected columns
df = df.dropna(subset=features + [target])

# Separate features and target
X = df[features]
y = df[target]

# Identify categorical and numerical features
categorical_features = ['Medical Condition', 'Insurance Provider', 'Admission Type', 'Medication']
numerical_features = ['Age', 'Billing Amount', 'Length of Stay']

# Create a column transformer for preprocessing
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ],
    remainder='passthrough' # Keep other columns not specified in transformers
)

# Create a pipeline with preprocessing and model
model_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                                 ('classifier', RandomForestClassifier(random_state=42))])


# Split into training and test (optional)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model_pipeline.fit(X_train, y_train)

# Save model and scaler (saving the whole pipeline)
joblib.dump(model_pipeline, "readmission_model_pipeline.pkl")


print("✅ Model pipeline saved as 'readmission_model_pipeline.pkl'.")