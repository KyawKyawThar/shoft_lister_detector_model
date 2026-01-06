"""
Evaluate the best model with an adjustable threshold to analyze the trade-off
between precision and recall, particularly for reducing false positives.
"""

import os
import pickle
import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import custom_object_scope
from sklearn.metrics import classification_report
from helper import make_confusion_matrix
from gcn_model import GraphConvolution, SpatialTemporalBlock # Import custom layers

# Configuration
MODEL_PATH = "saved_models/lstm_gcn_final.h5"
FEATURE_PATH = "extracted_features/pose_features.pkl"
EVAL_SAVE_PATH = "evaluation_results"
CUSTOM_THRESHOLD = 0.7  # Adjust this threshold (default is 0.5)

# Create output directory
os.makedirs(EVAL_SAVE_PATH, exist_ok=True)

def load_data():
    """Load the test data split."""
    if not os.path.exists(FEATURE_PATH):
        print(f"Error: Feature file not found at {FEATURE_PATH}")
        return None, None
        
    with open(FEATURE_PATH, 'rb') as f:
        data = pickle.load(f)
    X, y = data['X'], data['y']
    
    # Simple split for demonstration (in a real scenario, use the exact same splits)
    from sklearn.model_selection import train_test_split
    _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    return X_test, y_test

def main():
    """Main evaluation pipeline."""
    print("="*60)
    print("Model Evaluation with Custom Threshold")
    print("="*60)
    
    # 1. Load Model
    print(f"\nLoading model from {MODEL_PATH}...")
    if not os.path.exists(MODEL_PATH):
        print("Error: Model file not found. Please run the training script first.")
        return
        
    # When loading custom layers, you might need to provide them
    with custom_object_scope({'GraphConvolution': GraphConvolution, 'SpatialTemporalBlock': SpatialTemporalBlock}):
        model = tf.keras.models.load_model(MODEL_PATH)
    model.summary()
    
    # 2. Load Data
    print("\nLoading test data...")
    X_test, y_test = load_data()
    if X_test is None:
        return
        
    print(f"Loaded {len(X_test)} test samples.")
    
    # 3. Get Model Predictions (Probabilities)
    print("\nGenerating model predictions...")
    y_pred_proba = model.predict(X_test)
    
    # 4. Evaluate with default threshold (0.5)
    print("\n" + "="*60)
    print("Evaluation with Default Threshold (0.5)")
    print("="*60)
    y_pred_default = (y_pred_proba > 0.5).astype(int).flatten()
    
    print("\nClassification Report (Default):")
    print(classification_report(y_test, y_pred_default, target_names=['Normal', 'Shoplifting']))
    
    make_confusion_matrix(y_test, y_pred_default, "LSTM-GCN_Default_Threshold", EVAL_SAVE_PATH, classes=['Normal', 'Shoplifting'])

    # 5. Evaluate with custom threshold
    print("\n" + "="*60)
    print(f"Evaluation with Custom Threshold ({CUSTOM_THRESHOLD})")
    print("="*60)
    y_pred_custom = (y_pred_proba > CUSTOM_THRESHOLD).astype(int).flatten()

    print(f"\nClassification Report (Threshold={CUSTOM_THRESHOLD}):")
    print(classification_report(y_test, y_pred_custom, target_names=['Normal', 'Shoplifting']))

    make_confusion_matrix(y_test, y_pred_custom, f"LSTM-GCN_Custom_Threshold_{CUSTOM_THRESHOLD}", EVAL_SAVE_PATH, classes=['Normal', 'Shoplifting'])
    
    print("\nEvaluation complete. Check the 'evaluation_results' directory for the confusion matrices.")

if __name__ == "__main__":
    main()
