import streamlit as st
import joblib
import pandas as pd
import numpy as np

st.set_page_config(layout="wide", page_title="Cricket Injury 🏏", page_icon="🏏")
st.title("🏏 Cricket Injury Detection")
st.markdown("*Linnaeus University Master's Thesis 2026*")

try:
    @st.cache_data
    def load_model():
        model = joblib.load("models/model.pkl")
        return model

    model = load_model()
    st.success("✅ Model loaded OK")

    # 8 sliders - safe defaults
    st.sidebar.header("Pose Features")
    f1 = st.sidebar.slider("knee_asym", 0.0, 5.0, 1.0)
    f2 = st.sidebar.slider("hip_tilt", 0.0, 2.0, 0.5)
    f3 = st.sidebar.slider("f3", 0.0, 1.0, 0.8)
    f4 = st.sidebar.slider("f4", 0.0, 180.0, 90.0)
    f5 = f6 = f7 = f8 = 0.5  # Fixed safe values

    X = np.array([[f1,f2,f3,f4,f5,f6,f7,f8]])
    risk = model.predict_proba(X)[0,1]
    
    col1, col2 = st.columns(2)
    col1.metric("Risk %", f"{risk:.1%}")
    col2.success("SAFE" if risk<0.6 else "HIGH RISK")

    # Frames gallery
    st.header("High Risk Frames")
    try:
        df = pd.read_csv("data/frame_index.csv")
        for i, row in df.head(6).iterrows():
            frame = f"data/frames/{int(row.track_id):06d}.jpg"
            st.image(frame, width=300, caption=f"Track {row.track_id}: {row.knee_asym:.2f}")
    except:
        st.info("Frames loading...")

except Exception as e:
    st.error(f"Error: {e}")
    st.info("Check logs or requirements.txt")

st.caption("🎓 Production RF Model Deploy")
