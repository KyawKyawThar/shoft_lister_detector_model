"""
Helper functions for the Shoplifting Detection project.
"""

import os
import matplotlib.pyplot as plt
import numpy as np
import itertools
from sklearn.metrics import confusion_matrix, classification_report

def plot_loss_curves(history, model_name, save_path):
    """
    Plots and saves separate loss and accuracy curves for training and validation metrics.
    """
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot Loss
    axes[0].plot(history.history['loss'], label='Train Loss')
    axes[0].plot(history.history['val_loss'], label='Val Loss')
    axes[0].set_title(f'{model_name} - Loss Curves')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].legend()
    axes[0].grid(True)
    
    # Plot Accuracy
    axes[1].plot(history.history['accuracy'], label='Train Accuracy')
    axes[1].plot(history.history['val_accuracy'], label='Val Accuracy')
    axes[1].set_title(f'{model_name} - Accuracy Curves')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].legend()
    axes[1].grid(True)
    
    plt.suptitle(f'{model_name} Training History', fontsize=14)
    plt.tight_layout()
    
    filepath = os.path.join(save_path, f'{model_name}_training_history.png')
    plt.savefig(filepath, dpi=150)
    plt.show()
    print(f"Training history saved to {filepath}")

def make_confusion_matrix(y_true, y_pred, model_name, save_path, classes=None, figsize=(8, 8), text_size=12, norm=False): 
    """
    Plots a labeled confusion matrix.
    """
    cm = confusion_matrix(y_true, y_pred)
    if norm:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    n_classes = cm.shape[0]
    fig, ax = plt.subplots(figsize=figsize)
    cax = ax.matshow(cm, cmap=plt.cm.Blues)
    fig.colorbar(cax)

    labels = classes if classes else np.arange(cm.shape[0])
    
    ax.set(title=f"{model_name} - Confusion Matrix",
           xlabel="Predicted label",
           ylabel="True label",
           xticks=np.arange(n_classes),
           yticks=np.arange(n_classes), 
           xticklabels=labels,
           yticklabels=labels)
    
    ax.xaxis.set_label_position("bottom")
    ax.xaxis.tick_bottom()

    threshold = (cm.max() + cm.min()) / 2.

    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        color = "white" if cm[i, j] > threshold else "black"
        value = f"{cm[i, j]:.2f}" if norm else f"{cm[i, j]}"
        plt.text(j, i, value, horizontalalignment="center", color=color, size=text_size)
        
    filepath = os.path.join(save_path, f'{model_name}_confusion_matrix.png')
    plt.savefig(filepath, dpi=150)
    plt.show()
    print(f"Confusion matrix saved to {filepath}")
