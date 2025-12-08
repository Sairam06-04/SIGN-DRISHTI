import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import pickle
import os
import numpy as np

# --- Configuration ---
DATA_FILE = os.path.join('data', 'sign_language_data.csv')
MODEL_DIR = 'model'
if not os.path.exists(MODEL_DIR):
    os.makedirs(MODEL_DIR)
MODEL_FILE = os.path.join(MODEL_DIR, 'sign_language_model.p')

# --- Load Data ---
print(f"Loading data from {DATA_FILE}...")
try:
    df = pd.read_csv(DATA_FILE)
except FileNotFoundError:
    print(f"Error: Data file not found at {DATA_FILE}")
    print("Please run '1_data_collection.py' first to collect data.")
    exit()

if df.empty:
    print("Error: Data file is empty. Please collect data.")
    exit()

# Handle potential NaN values (e.g., if landmark detection failed)
df.fillna(0, inplace=True) # Replace NaNs with 0

print(f"Found {len(df)} samples.")
print(f"Classes found: {df['label'].unique()}")

if len(df['label'].unique()) < 2:
    print("Error: Need data for at least two different signs to train a classifier.")
    exit()

# --- Prepare Data ---
X = df.drop('label', axis=1) # Features (all landmark columns)
y = df['label']              # Labels (the sign names)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print(f"Training samples: {len(X_train)}, Testing samples: {len(X_test)}")

# --- Train Model ---
print("Training Logistic Regression model...")
# Increased max_iter for convergence, adjusted C for regularization
model = LogisticRegression(max_iter=1000, C=0.1, solver='liblinear', random_state=42)
model.fit(X_train, y_train)
print("Model training complete.")

# --- Evaluate Model ---
y_pred_train = model.predict(X_train)
train_accuracy = accuracy_score(y_train, y_pred_train)
y_pred_test = model.predict(X_test)
test_accuracy = accuracy_score(y_test, y_pred_test)

print(f"\nTraining Accuracy: {train_accuracy * 100:.2f}%")
print(f"Testing Accuracy: {test_accuracy * 100:.2f}%")

# --- Save Model ---
print(f"Saving model to {MODEL_FILE}...")
with open(MODEL_FILE, 'wb') as f:
    pickle.dump(model, f)

print("Model saved successfully! 🚀")