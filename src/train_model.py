import os
import numpy as np
import tensorflow as tf

def generate_synthetic_data(num_samples, length=2000):
    print(f"Generating {num_samples} synthetic light curves...")
    # Generate background stellar noise (random baseline brightness)
    X = np.random.normal(loc=0.8, scale=0.02, size=(num_samples, length, 1))
    
    # Generate labels: 0 for No Planet, 1 for Planet
    y = np.random.randint(0, 2, size=(num_samples, 1))
    
    # Inject an artificial planet transit (a dip in brightness) into the positive samples
    for i in range(num_samples):
        if y[i] == 1:
            transit_start = np.random.randint(800, 1000)
            transit_end = transit_start + np.random.randint(50, 150)
            X[i, transit_start:transit_end, 0] -= 0.15  # Dim the starlight by 15%
            
    return X, y

def train_network():
    model_path = os.path.join("models", "exoscan_1d_cnn.keras")
    
    if not os.path.exists(model_path):
        print("Model not found! Run SRC\\build_model.py first.")
        return
        
    # Load the untrained architecture
    model = tf.keras.models.load_model(model_path)
    
    # Generate 800 examples to train on, and 200 to test/validate with
    X_train, y_train = generate_synthetic_data(num_samples=800)
    X_val, y_val = generate_synthetic_data(num_samples=200)
    
    print("\nStarting the Deep Learning training loop...")
    # Train the network over 5 epochs (passes through the data)
    model.fit(X_train, y_train, epochs=5, batch_size=32, validation_data=(X_val, y_val))
    
    # Save the newly calculated parameters/weights back to the file
    model.save(model_path)
    print(f"\nTraining complete! Updated weights saved to {model_path}")

if __name__ == "__main__":
    train_network()