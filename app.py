import os
import sys
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import streamlit as st
import requests

sys.path.append(os.path.abspath("src"))
from download_data import fetch_and_process_star

# 1. Configure the Web Page Layout
st.set_page_config(page_title="ExoScan Dashboard", page_icon="🔭", layout="wide")

st.title("🔭 ExoScan-v2: Autonomous Exoplanet Pipeline")
st.markdown("Analyze live telemetry from NASA's Space Telescopes using 1D Convolutional Neural Networks.")
st.markdown("---")

# 2. Sidebar Configuration
st.sidebar.header("System Configuration")
api_key = st.sidebar.text_input("Enter Gemini API Key:", type="password")
st.sidebar.info("A valid API key is required to generate automated AI Discovery Briefs.")

# 3. Streamlined Search Interface (Columns)
col1, col2 = st.columns([3, 1])
with col1:
    # label_visibility="collapsed" hides the text above the box for a cleaner look
    target_name = st.text_input("Target Star ID", "TOI-700", label_visibility="collapsed")
with col2:
    # use_container_width makes the button stretch to fit the column perfectly
    start_scan = st.button("Initialize Deep Scan", type="primary", use_container_width=True)

if start_scan:
    # 4. Interactive Status Checklist
    with st.status(f"Scanning target {target_name}...", expanded=True) as status:
        st.write("📡 Querying NASA MAST Archive for live telemetry...")
        success = fetch_and_process_star(target_name)
        
        if not success:
            status.update(label="Data Retrieval Failed", state="error", expanded=True)
            st.stop()
            
        st.write("🧠 Formatting mathematical tensors...")
        data_path = os.path.join("data", "processed", f"{target_name}_ml_ready.npy")
        model_path = os.path.join("models", "exoscan_1d_cnn.keras")
        light_curve = np.load(data_path)
        model = tf.keras.models.load_model(model_path)
        
        st.write("⚡ Executing Deep Learning Inference...")
        X = light_curve.reshape(1, 2000, 1)
        prediction = model.predict(X, verbose=0)
        prob = float(prediction[0][0])
        
        status.update(label="Scan Complete!", state="complete", expanded=False)

    # 5. High-Visibility Alert Banners
    if prob >= 0.5:
        st.error(f"🚨 **HIGH-CONFIDENCE PLANET CANDIDATE DETECTED** (Transit Probability: {prob * 100:.2f}%)")
    else:
        st.info(f"📉 **BACKGROUND STELLAR NOISE** (Transit Probability: {prob * 100:.2f}%)")

    # 6. Tabbed Interface for Clean Output
    tab1, tab2, tab3 = st.tabs(["📊 Visual Analysis", "🤖 AI Discovery Brief", "⚙️ Raw Telemetry"])
    
    with tab1:
        fig, ax = plt.subplots(figsize=(12, 4))
        ax.plot(light_curve, color='#1f77b4', alpha=0.8, label="Normalized Starlight")
        
        if prob >= 0.5:
            ax.axvspan(800, 1150, color='red', alpha=0.15, label=f'CNN Anomaly Zone')
            
        ax.set_xlabel("Time Bins")
        ax.set_ylabel("Relative Flux")
        ax.legend()
        ax.grid(True, linestyle='--', alpha=0.5)
        st.pyplot(fig) 
        
    with tab2:
        if api_key:
            with st.spinner("Drafting astrophysics report..."):
                url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key={api_key}"
                status_text = "PLANET CANDIDATE" if prob >= 0.5 else "STELLAR NOISE"
                prompt = f"You are ExoScan, a NASA AI. Target: {target_name}. Confidence: {prob*100:.2f}%. Status: {status_text}. Write a 3-sentence discovery brief."
                
                try:
                    resp = requests.post(url, headers={'Content-Type': 'application/json'}, json={"contents": [{"parts": [{"text": prompt}]}]})
                    resp.raise_for_status()
                    st.write(resp.json()['candidates'][0]['content']['parts'][0]['text'])
                except Exception as e:
                    st.error("Failed to generate brief. Check API Key validity.")
        else:
            st.warning("Please enter your Gemini API Key in the sidebar to view the AI report.")
            
    with tab3:
        st.write("**Model Architecture:** `1D Convolutional Neural Network`")
        st.write("**Input Tensor Shape:** `(1, 2000, 1)`")
        st.line_chart(light_curve) # A quick native Streamlit chart of the raw data