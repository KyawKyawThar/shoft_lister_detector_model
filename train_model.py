"""
Main Training Script for Shoplifting Detection
Combines feature extraction + GCN model training using TensorFlow
"""

import os
import pickle
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report
from datetime import datetime

# Import our modules
from gcn_model import build_stgcn_model, build_lstm_gcn_model
from feature_extraction import process_dataset
from helper import plot_loss_curves, make_confusion_matrix

# Configuration
FEATURE_PATH = "extracted_features/pose_features.pkl"
MODEL_SAVE_PATH = "saved_models"
SEQUENCE_LENGTH = 30
NUM_KEYPOINTS = 17
NUM_FEATURES = 3
BATCH_SIZE = 16
EPOCHS = 100
LEARNING_RATE = 0.001

# Create directories
os.makedirs(MODEL_SAVE_PATH, exist_ok=True)
os.makedirs("logs", exist_ok=True)


def load_or_extract_features():
    """
    Load pre-extracted features or extract them from videos
    """
    if os.path.exists(FEATURE_PATH):
        print(f"Loading pre-extracted features from {FEATURE_PATH}...")
        with open(FEATURE_PATH, 'rb') as f:
            data = pickle.load(f)
        X, y = data['X'], data['y']
        print(f"Loaded {len(X)} sequences")
    else:
        print("Features not found. Running feature extraction...")
        print("This may take a while depending on your hardware...\n")
        X, y = process_dataset()
    
    return X, y


def prepare_data(X, y, test_size=0.2, val_size=0.1):
    """
    Prepare data for training with train/val/test split
    """
    # First split: train+val vs test
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )
    
    # Second split: train vs val
    val_ratio = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval, test_size=val_ratio, random_state=42, stratify=y_trainval
    )
    
    print(f"\nData split:")
    print(f"  Training:   {len(X_train)} samples ({np.sum(y_train==0)} normal, {np.sum(y_train==1)} shoplifting)")
    print(f"  Validation: {len(X_val)} samples ({np.sum(y_val==0)} normal, {np.sum(y_val==1)} shoplifting)")
    print(f"  Test:       {len(X_test)} samples ({np.sum(y_test==0)} normal, {np.sum(y_test==1)} shoplifting)")
    
    return (X_train, y_train), (X_val, y_val), (X_test, y_test)


def create_data_augmentation_layer():
    """
    Create data augmentation for skeleton sequences
    """
    def augment(x, y):
        # Random horizontal flip (mirror the skeleton)
        if tf.random.uniform([]) > 0.5:
            # Swap left/right keypoints
            # COCO format: left/right pairs are adjacent indices
            left_indices = [1, 3, 5, 7, 9, 11, 13, 15]
            right_indices = [2, 4, 6, 8, 10, 12, 14, 16]
            
            x_flipped = x.numpy()
            x_flipped[:, left_indices, :], x_flipped[:, right_indices, :] = \
                x_flipped[:, right_indices, :].copy(), x_flipped[:, left_indices, :].copy()
            
            # Flip x-coordinates
            x_flipped[:, :, 0] = 1.0 - x_flipped[:, :, 0]
            x = tf.constant(x_flipped)
        
        # Random time shift (shift sequence in time)
        if tf.random.uniform([]) > 0.5:
            shift = tf.random.uniform([], -3, 3, dtype=tf.int32)
            x = tf.roll(x, shift, axis=0)
        
        # Add small random noise
        if tf.random.uniform([]) > 0.5:
            noise = tf.random.normal(tf.shape(x), mean=0.0, stddev=0.02)
            x = x + noise
        
        return x, y
    
    return augment


def create_tf_dataset(X, y, batch_size, training=False):
    """
    Create TensorFlow dataset with optional augmentation
    """
    dataset = tf.data.Dataset.from_tensor_slices((X.astype(np.float32), y.astype(np.float32)))
    
    if training:
        dataset = dataset.shuffle(buffer_size=len(X))
    
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    
    return dataset


def compile_model(model, learning_rate=LEARNING_RATE):
    """
    Compile model with optimizer, loss, and metrics
    """
    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
    
    model.compile(
        optimizer=optimizer,
        loss='binary_crossentropy',
        metrics=[
            'accuracy',
            keras.metrics.Precision(name='precision'),
            keras.metrics.Recall(name='recall'),
            keras.metrics.AUC(name='auc')
        ]
    )
    
    return model


def create_callbacks(model_name):
    """
    Create training callbacks
    """
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    
    callbacks = [
        # Model checkpoint - save best model weights only
        keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(MODEL_SAVE_PATH, f"{model_name}_best_weights.h5"),
            monitor='val_auc',
            mode='max',
            save_best_only=True,
            save_weights_only=True, # Change: Save only weights
            verbose=1
        ),
        
        # Early stopping - still restores best weights
        keras.callbacks.EarlyStopping(
            monitor='val_auc',
            mode='max',
            patience=15,
            restore_best_weights=True,
            verbose=1
        ),
        
        # Learning rate reduction
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-6,
            verbose=1
        ),
        
        # TensorBoard logging
        keras.callbacks.TensorBoard(
            log_dir=f"logs/{model_name}_{timestamp}",
            histogram_freq=1
        )
    ]
    
    return callbacks


def train_model(model, model_name, train_data, val_data, class_weights=None):
    """
    Train the model
    """
    X_train, y_train = train_data
    X_val, y_val = val_data
    
    # Create datasets
    train_dataset = create_tf_dataset(X_train, y_train, BATCH_SIZE, training=True)
    val_dataset = create_tf_dataset(X_val, y_val, BATCH_SIZE, training=False)
    
    # Compile model
    model = compile_model(model)
    
    # Create callbacks
    callbacks = create_callbacks(model_name)
    
    print(f"\n{'='*60}")
    print(f"Training {model_name}")
    print(f"{'='*60}")
    
    # Train
    history = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=EPOCHS,
        callbacks=callbacks,
        class_weight=class_weights,
        verbose=1
    )
    
    return model, history


def evaluate_model(model, test_data, model_name):
    """
    Evaluate model on test set
    """
    X_test, y_test = test_data
    
    print(f"\n{'='*60}")
    print(f"Evaluating {model_name} on Test Set")
    print(f"{'='*60}")
    
    # Evaluate
    test_dataset = create_tf_dataset(X_test, y_test, BATCH_SIZE, training=False)
    results = model.evaluate(test_dataset, verbose=1)
    
    # Print metrics
    metric_names = model.metrics_names
    print(f"\nTest Results:")
    for name, value in zip(metric_names, results):
        print(f"  {name}: {value:.4f}")
    
    # Predictions for detailed metrics
    y_pred_proba = model.predict(X_test)
    y_pred = (y_pred_proba > 0.5).astype(int).flatten()
    
    # Classification report
    print(f"\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['Normal', 'Shoplifting']))
    
    # Confusion matrix
    make_confusion_matrix(y_test, y_pred, model_name, MODEL_SAVE_PATH, classes=['Normal', 'Shoplifting'])
    
    return results, y_pred, y_pred_proba


def main():
    """
    Main training pipeline
    """
    print("="*60)
    print("Shoplifting Detection - Model Training Pipeline")
    print("="*60)
    
    # Check GPU availability
    print(f"\nTensorFlow version: {tf.__version__}")
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        print(f"GPU available: {gpus}")
        # Enable memory growth
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    else:
        print("No GPU found, using CPU")
    
    # Step 1: Load or extract features
    print("\n" + "="*60)
    print("Step 1: Loading/Extracting Features")
    print("="*60)
    X, y = load_or_extract_features()
    
    print(f"\nFeature shape: {X.shape}")
    print(f"  - Sequences: {X.shape[0]}")
    print(f"  - Frames per sequence: {X.shape[1]}")
    print(f"  - Keypoints: {X.shape[2]}")
    print(f"  - Features per keypoint: {X.shape[3]}")
    
    # Step 2: Prepare data
    print("\n" + "="*60)
    print("Step 2: Preparing Data")
    print("="*60)
    train_data, val_data, test_data = prepare_data(X, y)
    
    # Compute class weights for imbalanced data
    class_weights = compute_class_weight(
        'balanced',
        classes=np.unique(train_data[1]),
        y=train_data[1]
    )
    class_weight_dict = {i: w for i, w in enumerate(class_weights)}
    print(f"\nClass weights: {class_weight_dict}")
    
    # Step 3: Build models
    print("\n" + "="*60)
    print("Step 3: Building Models")
    print("="*60)
    
    # Option 1: ST-GCN Model (more complex, potentially better)
    print("\nBuilding ST-GCN model...")
    stgcn_model = build_stgcn_model(
        sequence_length=SEQUENCE_LENGTH,
        num_keypoints=NUM_KEYPOINTS,
        num_features=NUM_FEATURES,
        num_classes=2
    )
    stgcn_model.summary()
    
    # Option 2: LSTM-GCN Model (simpler, faster to train)
    print("\nBuilding LSTM-GCN model...")
    lstm_gcn_model = build_lstm_gcn_model(
        sequence_length=SEQUENCE_LENGTH,
        num_keypoints=NUM_KEYPOINTS,
        num_features=NUM_FEATURES,
        num_classes=2
    )
    lstm_gcn_model.summary()
    
    # Step 4: Train models
    print("\n" + "="*60)
    print("Step 4: Training Models")
    print("="*60)
    
    # Train LSTM-GCN first (faster, good baseline)
    lstm_gcn_model, lstm_history = train_model(
        lstm_gcn_model, 
        "lstm_gcn",
        train_data, 
        val_data,
        class_weight_dict
    )
    
    # Train ST-GCN
    stgcn_model, stgcn_history = train_model(
        stgcn_model,
        "stgcn",
        train_data,
        val_data,
        class_weight_dict
    )
    
    # Step 5: Evaluate models
    print("\n" + "="*60)
    print("Step 5: Evaluating Models")
    print("="*60)
    
    lstm_results, _, _ = evaluate_model(lstm_gcn_model, test_data, "LSTM-GCN")
    stgcn_results, _, _ = evaluate_model(stgcn_model, test_data, "ST-GCN")
    
    # Step 6: Plot training history
    print("\n" + "="*60)
    print("Step 6: Plotting Training History")
    print("="*60)
    
    plot_loss_curves(lstm_history, "LSTM-GCN", MODEL_SAVE_PATH)
    plot_loss_curves(stgcn_history, "ST-GCN", MODEL_SAVE_PATH)
    
    # Step 7: Load best weights and save final models
    print("\n" + "="*60)
    print("Step 7: Saving Models")
    print("="*60)
    
    # Load best weights for LSTM-GCN and save
    lstm_gcn_model.load_weights(os.path.join(MODEL_SAVE_PATH, "lstm_gcn_best_weights.h5"))
    lstm_gcn_model.save(os.path.join(MODEL_SAVE_PATH, "lstm_gcn_final.h5"))

    # Load best weights for ST-GCN and save
    stgcn_model.load_weights(os.path.join(MODEL_SAVE_PATH, "stgcn_best_weights.h5"))
    stgcn_model.save(os.path.join(MODEL_SAVE_PATH, "stgcn_final.h5"))
    
    print(f"Models saved to {MODEL_SAVE_PATH}/")
    
    print("\n" + "="*60)
    print("Training Complete!")
    print("="*60)
    
    # Summary
    print("\nModel Performance Summary:")
    print(f"  LSTM-GCN Test AUC: {lstm_results[4]:.4f}")
    print(f"  ST-GCN Test AUC: {stgcn_results[4]:.4f}")
    
    if lstm_results[4] > stgcn_results[4]:
        print("\n  Best model: LSTM-GCN")
    else:
        print("\n  Best model: ST-GCN")


if __name__ == "__main__":
    main()
