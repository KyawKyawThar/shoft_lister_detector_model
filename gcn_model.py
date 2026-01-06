"""
Spatial-Temporal Graph Convolutional Network (ST-GCN) for Shoplifting Detection
Built with TensorFlow/Keras
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np

# COCO Skeleton Graph Structure (17 keypoints)
# Keypoint indices: 0-nose, 1-left_eye, 2-right_eye, 3-left_ear, 4-right_ear,
# 5-left_shoulder, 6-right_shoulder, 7-left_elbow, 8-right_elbow,
# 9-left_wrist, 10-right_wrist, 11-left_hip, 12-right_hip,
# 13-left_knee, 14-right_knee, 15-left_ankle, 16-right_ankle

# Define skeleton edges (bone connections)
SKELETON_EDGES = [
    (0, 1), (0, 2),  # nose to eyes
    (1, 3), (2, 4),  # eyes to ears
    (0, 5), (0, 6),  # nose to shoulders (via neck conceptually)
    (5, 6),          # shoulder to shoulder
    (5, 7), (7, 9),  # left arm
    (6, 8), (8, 10), # right arm
    (5, 11), (6, 12), # shoulders to hips
    (11, 12),        # hip to hip
    (11, 13), (13, 15), # left leg
    (12, 14), (14, 16)  # right leg
]

NUM_KEYPOINTS = 17


def build_adjacency_matrix():
    """
    Build the adjacency matrix for the skeleton graph
    Returns normalized adjacency matrix for GCN
    """
    A = np.zeros((NUM_KEYPOINTS, NUM_KEYPOINTS), dtype=np.float32)
    
    # Add edges (bidirectional)
    for i, j in SKELETON_EDGES:
        A[i, j] = 1
        A[j, i] = 1
    
    # Add self-loops
    A = A + np.eye(NUM_KEYPOINTS, dtype=np.float32)
    
    # Normalize (symmetric normalization: D^(-1/2) * A * D^(-1/2))
    D = np.sum(A, axis=1)
    D_inv_sqrt = np.diag(np.power(D, -0.5))
    A_normalized = D_inv_sqrt @ A @ D_inv_sqrt
    
    return A_normalized


class GraphConvolution(layers.Layer):
    """
    Graph Convolutional Layer
    Performs spatial convolution over skeleton joints
    """
    def __init__(self, output_dim, adjacency_matrix, **kwargs):
        super(GraphConvolution, self).__init__(**kwargs)
        self.output_dim = output_dim
        self.adjacency_matrix = tf.constant(adjacency_matrix, dtype=tf.float32)
        
    def build(self, input_shape):
        self.kernel = self.add_weight(
            name='kernel',
            shape=(input_shape[-1], self.output_dim),
            initializer='glorot_uniform',
            trainable=True
        )
        self.bias = self.add_weight(
            name='bias',
            shape=(self.output_dim,),
            initializer='zeros',
            trainable=True
        )
        
    def call(self, inputs):
        # inputs shape: (batch, time, num_joints, features)
        # Apply graph convolution: A * X * W
        
        # Reshape for matrix multiplication
        batch_size = tf.shape(inputs)[0]
        time_steps = tf.shape(inputs)[1]
        
        # Reshape to (batch * time, num_joints, features)
        x = tf.reshape(inputs, [-1, NUM_KEYPOINTS, inputs.shape[-1]])
        
        # Graph convolution: A * X
        x = tf.matmul(self.adjacency_matrix, x)
        
        # Linear transformation: (A * X) * W + b
        x = tf.matmul(x, self.kernel) + self.bias
        
        # Reshape back to (batch, time, num_joints, output_dim)
        x = tf.reshape(x, [batch_size, time_steps, NUM_KEYPOINTS, self.output_dim])
        
        return x
    
    def get_config(self):
        config = super().get_config()
        config.update({
            'output_dim': self.output_dim,
            'adjacency_matrix': self.adjacency_matrix.numpy().tolist()
        })
        return config


class SpatialTemporalBlock(layers.Layer):
    """
    Spatial-Temporal Graph Convolutional Block
    Combines spatial GCN with temporal convolution
    """
    def __init__(self, out_channels, adjacency_matrix, temporal_kernel_size=9, stride=1, **kwargs):
        super(SpatialTemporalBlock, self).__init__(**kwargs)
        self.out_channels = out_channels
        self.adjacency_matrix = adjacency_matrix
        self.temporal_kernel_size = temporal_kernel_size
        self.stride = stride
        
    def build(self, input_shape):
        in_channels = input_shape[-1]
        
        # Spatial Graph Convolution
        self.gcn = GraphConvolution(self.out_channels, self.adjacency_matrix)
        self.bn1 = layers.BatchNormalization()
        
        # Temporal Convolution (along time axis)
        self.temporal_conv = layers.Conv2D(
            self.out_channels,
            kernel_size=(self.temporal_kernel_size, 1),
            strides=(self.stride, 1),
            padding='same'
        )
        self.bn2 = layers.BatchNormalization()
        
        # Residual connection
        if in_channels != self.out_channels or self.stride != 1:
            self.residual = layers.Conv2D(
                self.out_channels,
                kernel_size=(1, 1),
                strides=(self.stride, 1),
                padding='same'
            )
            self.bn_res = layers.BatchNormalization()
        else:
            self.residual = None
        
        self.dropout = layers.Dropout(0.25)
        
    def call(self, inputs, training=False):
        # inputs shape: (batch, time, num_joints, features)
        
        # Spatial GCN
        x = self.gcn(inputs)
        x = self.bn1(x, training=training)
        x = tf.nn.relu(x)
        
        # Temporal Convolution
        x = self.temporal_conv(x)
        x = self.bn2(x, training=training)
        
        # Residual connection
        if self.residual is not None:
            res = self.residual(inputs)
            res = self.bn_res(res, training=training)
        else:
            res = inputs
        
        x = x + res
        x = tf.nn.relu(x)
        x = self.dropout(x, training=training)
        
        return x

    def get_config(self):
        config = super().get_config()
        config.update({
            "out_channels": self.out_channels,
            "adjacency_matrix": self.adjacency_matrix.tolist(), # Convert numpy array to list for serialization
            "temporal_kernel_size": self.temporal_kernel_size,
            "stride": self.stride
        })
        return config


def build_stgcn_model(sequence_length=30, num_keypoints=17, num_features=3, num_classes=2):
    """
    Build the full ST-GCN model for action classification
    
    Args:
        sequence_length: Number of frames in each sequence
        num_keypoints: Number of body keypoints (17 for COCO)
        num_features: Features per keypoint (x, y, confidence = 3)
        num_classes: Number of output classes (2: Normal, Shoplifting)
    
    Returns:
        Compiled Keras model
    """
    # Get normalized adjacency matrix
    adjacency_matrix = build_adjacency_matrix()
    
    # Input layer
    inputs = layers.Input(shape=(sequence_length, num_keypoints, num_features))
    
    # Initial batch normalization
    x = layers.BatchNormalization()(inputs)
    
    # ST-GCN Blocks (progressively increase channels)
    x = SpatialTemporalBlock(64, adjacency_matrix, temporal_kernel_size=9)(x)
    x = SpatialTemporalBlock(64, adjacency_matrix, temporal_kernel_size=9)(x)
    x = SpatialTemporalBlock(64, adjacency_matrix, temporal_kernel_size=9)(x)
    
    x = SpatialTemporalBlock(128, adjacency_matrix, temporal_kernel_size=9, stride=2)(x)
    x = SpatialTemporalBlock(128, adjacency_matrix, temporal_kernel_size=9)(x)
    x = SpatialTemporalBlock(128, adjacency_matrix, temporal_kernel_size=9)(x)
    
    x = SpatialTemporalBlock(256, adjacency_matrix, temporal_kernel_size=9, stride=2)(x)
    x = SpatialTemporalBlock(256, adjacency_matrix, temporal_kernel_size=9)(x)
    x = SpatialTemporalBlock(256, adjacency_matrix, temporal_kernel_size=9)(x)
    
    # Global Average Pooling
    x = layers.GlobalAveragePooling2D()(x)
    
    # Classification head
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(64, activation='relu')(x)
    x = layers.Dropout(0.3)(x)
    
    # Output layer
    if num_classes == 2:
        outputs = layers.Dense(1, activation='sigmoid')(x)
    else:
        outputs = layers.Dense(num_classes, activation='softmax')(x)
    
    model = keras.Model(inputs=inputs, outputs=outputs)
    
    return model


def build_lstm_gcn_model(sequence_length=30, num_keypoints=17, num_features=3, num_classes=2):
    """
    Alternative model: GCN + LSTM for temporal modeling
    This is a simpler alternative to full ST-GCN
    """
    adjacency_matrix = build_adjacency_matrix()
    
    inputs = layers.Input(shape=(sequence_length, num_keypoints, num_features))
    
    # Batch normalization
    x = layers.BatchNormalization()(inputs)
    
    # Spatial GCN layers
    x = GraphConvolution(64, adjacency_matrix)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Dropout(0.25)(x)
    
    x = GraphConvolution(128, adjacency_matrix)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Dropout(0.25)(x)
    
    # Flatten joints dimension for LSTM
    # Shape: (batch, time, num_joints * features) 
    x = layers.Reshape((sequence_length, num_keypoints * 128))(x)
    
    # Bidirectional LSTM for temporal modeling
    x = layers.Bidirectional(layers.LSTM(128, return_sequences=True))(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Bidirectional(layers.LSTM(64))(x)
    x = layers.Dropout(0.3)(x)
    
    # Classification head
    x = layers.Dense(64, activation='relu')(x)
    x = layers.Dropout(0.3)(x)
    
    if num_classes == 2:
        outputs = layers.Dense(1, activation='sigmoid')(x)
    else:
        outputs = layers.Dense(num_classes, activation='softmax')(x)
    
    model = keras.Model(inputs=inputs, outputs=outputs)
    
    return model


if __name__ == "__main__":
    # Test model building
    print("Building ST-GCN Model...")
    model = build_stgcn_model()
    model.summary()
    
    print("\n" + "="*50)
    print("\nBuilding LSTM-GCN Model...")
    model2 = build_lstm_gcn_model()
    model2.summary()
    
    # Test with dummy data
    print("\nTesting with dummy data...")
    dummy_input = np.random.randn(4, 30, 17, 3).astype(np.float32)
    output = model(dummy_input)
    print(f"ST-GCN Output shape: {output.shape}")
    
    output2 = model2(dummy_input)
    print(f"LSTM-GCN Output shape: {output2.shape}")
