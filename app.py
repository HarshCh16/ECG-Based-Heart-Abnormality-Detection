import streamlit as st
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

from features import extract_features_from_record

st.set_page_config(page_title="ECG Abnormality Detector", page_icon="❤️", layout="wide")

model = joblib.load('ecg_model.pkl')

st.title("❤️ ECG Abnormality Detector")
st.write("Select a sample ECG record to see the model classify heartbeats in real time.")

with st.sidebar:
    st.header("Model Overview")
    st.metric("Overall Accuracy", "94.9%")
    st.write("Trained on 112,566 heartbeats from 48 MIT-BIH Arrhythmia Database records.")
    st.write("**Features used:**")
    st.write("- RR interval\n- Heart rate\n- HRV (rolling)\n- Beat amplitude\n- QRS width")

data_folder = 'mit-bih-arrhythmia-database'
record_ids = sorted(set(f.split('.')[0] for f in os.listdir(data_folder) if f.endswith('.hea')))

col_a, col_b = st.columns([2, 1])
with col_a:
    selected_record = st.selectbox("Choose a sample record:", record_ids)
with col_b:
    window_sec = st.slider("Seconds of signal to display", 5, 30, 10)

if st.button("Analyze", type="primary"):
    features_df, filtered_signal, fs = extract_features_from_record(selected_record, data_folder)

    X = features_df[['rr_interval', 'heart_rate', 'hrv', 'amplitude', 'qrs_width']]
    predictions = model.predict(X)
    features_df['predicted_label'] = predictions

    accuracy_on_record = (features_df['predicted_label'] == features_df['true_label']).mean()

    # --- Summary metrics ---
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Beats Analyzed", len(features_df))
    col2.metric("Predicted Normal", int((features_df['predicted_label'] == 'Normal').sum()))
    col3.metric("Predicted Abnormal", int((features_df['predicted_label'] == 'Abnormal').sum()))
    col4.metric("Match with Ground Truth", f"{accuracy_on_record:.1%}")

    # --- Waveform plot with correct/incorrect markers ---
    n_samples = window_sec * fs
    t = np.arange(n_samples) / fs

    window_beats = features_df[features_df['sample_pos'] < n_samples].copy()
    window_beats['correct'] = window_beats['predicted_label'] == window_beats['true_label']

    fig, ax = plt.subplots(figsize=(12, 3.5))
    ax.plot(t, filtered_signal[:n_samples], color='black', linewidth=0.8)

    for _, beat in window_beats.iterrows():
        color = 'green' if beat['predicted_label'] == 'Normal' else 'red'
        marker = 'o' if beat['correct'] else 'X'
        ax.scatter(beat['sample_pos']/fs, filtered_signal[beat['sample_pos']],
                   c=color, marker=marker, s=80, zorder=5, edgecolors='black', linewidths=0.5)

    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Amplitude (mV)')
    ax.set_title(f"Record {selected_record} — Filtered ECG with Predictions")
    st.pyplot(fig)
    st.caption("🟢 Predicted Normal 🔴 Predicted Abnormal — ⭕ Correct ❌ Model mistake")

    # --- Feature importance chart ---
    st.subheader("What the model pays attention to")
    importance_df = pd.DataFrame({
        'feature': X.columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=True)

    fig2, ax2 = plt.subplots(figsize=(8, 3))
    ax2.barh(importance_df['feature'], importance_df['importance'], color='steelblue')
    ax2.set_xlabel('Importance')
    st.pyplot(fig2)

    # --- Raw prediction table (optional detail) ---
    with st.expander("See detailed beat-by-beat predictions"):
        st.dataframe(features_df[['sample_pos', 'rr_interval', 'heart_rate', 'true_label', 'predicted_label']])