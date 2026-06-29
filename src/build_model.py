import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout

def build_cnn_model(input_length):
    # Initialize a sequential neural network
    model = Sequential([
        # Block 1: Feature Extraction
        Conv1D(filters=32, kernel_size=5, activation='relu', input_shape=(input_length, 1)),
        MaxPooling1D(pool_size=2), 
        
        # Block 2: Deep Feature Extraction
        Conv1D(filters=64, kernel_size=5, activation='relu'),
        MaxPooling1D(pool_size=2),
        
        # Flatten the data for the decision layers
        Flatten(),
        
        # Block 3: The Dense Layers
        Dense(64, activation='relu'),
        Dropout(0.3), 
        
        # Final Output Layer (Sigmoid outputs a probability between 0 and 1)
        Dense(1, activation='sigmoid')
    ])
    
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

if __name__ == "__main__":
    # Locate the preprocessed data vector
    data_path = os.path.join("data", "processed", "TOI-700_ml_ready.npy")
    
    if os.path.exists(data_path):
        sample_data = np.load(data_path)
        
        # Reshape data to (batch_size, time_steps, features) -> (1, 2000, 1)
        X = sample_data.reshape(1, 2000, 1)
        print(f"Data successfully shaped for neural network: {X.shape}\n")
        
        # Build the architecture
        model = build_cnn_model(input_length=2000)
        
        # Print the neural network layers summary
        model.summary()
        
        # Save the untrained architecture architecture to the models folder
        os.makedirs("models", exist_ok=True)
        save_path = os.path.join("models", "exoscan_1d_cnn.keras")
        model.save(save_path)
        print(f"\nModel architecture compiled and saved to: {save_path}")
        
    else:
        print("Preprocessed data not found. Please run SRC\\preprocess_data.py first.")