import os
import numpy as np
import tensorflow as tf

def run_inference(target_name):
    print(f"Running 1D CNN model inference on target: {target_name}...")
    
    # 1. Load the saved model architecture
    model_path = os.path.join("models", "exoscan_1d_cnn.keras")
    if not os.path.exists(model_path):
        print(f"Error: Model architecture not found at {model_path}")
        return
        
    model = tf.keras.models.load_model(model_path)
    
    # 2. Load the preprocessed ML-ready array
    data_path = os.path.join("data", "processed", f"{target_name}_ml_ready.npy")
    if not os.path.exists(data_path):
        print(f"Error: Preprocessed data file not found at {data_path}")
        return
        
    sample_data = np.load(data_path)
    
    # 3. Reshape the array to fit the 3D tensor requirements: (batch_size, time_steps, features)
    X = sample_data.reshape(1, 2000, 1)
    
    # 4. Generate the prediction
    prediction = model.predict(X, verbose=0)
    probability = float(prediction[0][0])
    
    # 5. Display the classification results
    print("\n" + "="*50)
    print(f" ANALYSIS REPORT FOR TARGET: {target_name}")
    print("="*50)
    print(f"Raw Model Activation Score:  {probability:.4f}")
    print(f"Transit Detection Confidence: {probability * 100:.2%}")
    print("-"*50)
    
    # Evaluate classification based on a standard 50% sigmoid threshold
    if probability >= 0.5:
        print("STATUS: POSITIVE CANDIDATE DETECTED")
        print("RECOMMENDATION: Flag for multi-sector transit vetting pipeline.")
    else:
        print("STATUS: FALSE POSITIVE / STELLAR NOISE")
        print("RECOMMENDATION: Classify as constant flux or background variance.")
    print("="*50 + "\n")

if __name__ == "__main__":
    run_inference("TOI-700")