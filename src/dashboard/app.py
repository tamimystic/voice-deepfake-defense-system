import os, streamlit as st, pandas as pd, numpy as np, soundfile as sf, io, matplotlib.pyplot as plt, librosa
from src.inference.predictor import ForensicPredictor
from src.dashboard.components import render_3d_spectrogram_hud
from src.utils.config import path_config

st.set_page_config(page_title="AI Voice Deepfake Defense Operations Center", layout="wide", initial_sidebar_state="expanded")

@st.cache_resource
def get_predictor():
    return ForensicPredictor()

predictor = get_predictor()

st.title("Voice Deepfake Defense & Audio Forensics Operations Platform")
st.markdown("Enterprise AI Voice Anti-Spoofing, Spectral Anomaly Interception, and Biometric Defense Center.")

tab1, tab2, tab3 = st.tabs(["Audio Inspection & Live Detection", "Dataset Forensics & Protocol Explorer", "System Architecture & Health"])

with tab1:
    st.subheader("Real-Time Audio Ingestion & Deep Forensics Analysis")
    uploaded_file = st.file_uploader("Upload audio file (.flac, .wav, .mp3)", type=["flac", "wav", "mp3", "ogg"])
    if uploaded_file is not None:
        bytes_data = uploaded_file.read()
        col1, col2 = st.columns([1, 2])
        with col1:
            st.audio(bytes_data, format="audio/wav")
            if st.button("Run Forensic Deepfake Analysis", type="primary", use_container_width=True):
                with st.spinner("Analyzing spectral phase, LFCC vectors, and vocoder artifacts..."):
                    res = predictor.predict_from_bytes(bytes_data, filename=uploaded_file.name)
                    st.session_state["last_result"] = res
                    st.session_state["last_audio"] = bytes_data

        if "last_result" in st.session_state:
            res = st.session_state["last_result"]
            with col2:
                render_3d_spectrogram_hud(res["prediction"], res["spoof_probability"], res["bonafide_probability"], res["inference_latency_ms"])
            
            st.divider()
            mcol1, mcol2, mcol3, mcol4 = st.columns(4)
            mcol1.metric("Classification Decision", res["prediction"].upper(), delta="Flagged" if res["prediction"] == "spoof" else "Passed")
            mcol2.metric("Confidence Score", f"{res['confidence_score']*100:.2f}%")
            mcol3.metric("Risk Level", res["risk_level"])
            mcol4.metric("Inference Latency", f"{res['inference_latency_ms']} ms")
            
            st.subheader("Ensemble Multi-Stream Breakdown")
            bcol1, bcol2, bcol3 = st.columns(3)
            bcol1.progress(res["model_breakdown"]["light_cnn_lfcc"], text=f"Light-CNN (LFCC): {res['model_breakdown']['light_cnn_lfcc']*100:.1f}%")
            bcol2.progress(res["model_breakdown"]["se_resnet_mel"], text=f"SE-ResNet-18 (Mel): {res['model_breakdown']['se_resnet_mel']*100:.1f}%")
            bcol3.progress(res["model_breakdown"]["rawnet_waveform"], text=f"RawNet (SincNet): {res['model_breakdown']['rawnet_waveform']*100:.1f}%")

with tab2:
    st.subheader("ASVspoof 2019 Protocol Metadata Explorer")
    manifest_path = os.path.join(path_config.data_processed, "protocol_manifest.parquet")
    if os.path.exists(manifest_path):
        df = pd.read_parquet(manifest_path)
        col_a, col_b, col_c = st.columns(3)
        col_a.metric("Total Audio Utterances", f"{len(df):,}")
        col_b.metric("Total Bonafide (Authentic)", f"{(df.key == 'bonafide').sum():,}")
        col_c.metric("Total Spoof (Synthetic)", f"{(df.key == 'spoof').sum():,}")
        st.dataframe(df.head(100), use_container_width=True)
    else:
        st.warning("Protocol manifest not found. Run dataset parsing first.")

with tab3:
    st.subheader("CPU Execution & Forensic Health Metrics")
    hcol1, hcol2, hcol3 = st.columns(3)
    hcol1.info("Target CPU: Intel Core i5-12th Gen (Multi-Threaded)")
    hcol2.info("Memory Model: Out-of-Core Disk-Buffered Lazy Loading (8GB RAM Safe)")
    hcol3.info("Inference Pipeline: 1D SincNet + 2D LFCC + 2D Mel-Spectrogram Fusion")
