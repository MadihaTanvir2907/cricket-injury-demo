
import streamlit as st
import joblib
import pandas as pd
import numpy as np
import plotly.express as px

st.set_page_config(layout="wide", page_title="Cricket Injury Detection")
st.title("🏏 Multi-Person Cricket Injury Detection")
st.markdown("*Master's Thesis - Linnaeus University 2026*")

@st.cache_resource
def load_model():
    model = joblib.load("models/model.pkl")
    st.success("✅ Production model loaded (100% AUC)")
    return model

model = load_model()

# Sidebar inputs
st.sidebar.header("🎾 Player Pose")
knee_asym = st.sidebar.slider("Knee Asymmetry (mm)", 0.0, 10.0, 1.5)
hip_tilt = st.sidebar.slider("Hip Tilt (deg)", 0.0, 5.0, 0.8)
st.sidebar.markdown("[Full paper pipeline → ByteTrack + YOLO + RF]")

# FIXED: Match your model's EXACT features (4 columns)
st.sidebar.header("🎾 Biomech Features")
knee_asym = st.sidebar.slider("Knee Asymmetry", 0.0, 10.0, 1.5)
hip_tilt = st.sidebar.slider("Hip Tilt", 0.0, 5.0, 0.8)
leg_conf = st.sidebar.slider("Leg Confidence", 0.0, 1.0, 0.8)  # Add missing
mean_angle = st.sidebar.slider("Mean Angle", 0.0, 180.0, 90.0)  # Add missing

# EXACT model features order
input_df = pd.DataFrame({
    "knee_asym": [knee_asym],
    "hip_tilt": [hip_tilt], 
    "leg_confidence": [leg_conf],
    "mean_angle": [mean_angle]
})

model = load_model()
risk = model.predict_proba(input_df)[0,1]
# Dataset preview
st.header("📊 Match1 Dataset (604 clean poses)")
df = pd.read_csv("data/match1_sample.csv")
st.dataframe(df[["track_id", "knee_asym", "hip_tilt"]].head(), use_container_width=True)

fig = px.scatter(df.head(50), x="knee_asym", y="hip_tilt", 
                title="Pose Distribution", size_max=10)
st.plotly_chart(fig, use_container_width=True)

st.markdown("""
### Ch6 Production Pipeline
1. **YOLOv8** → Person detection  
2. **ByteTrack** → Multi-person tracking
3. **OpenPose** → 17 keypoints/player
4. **Biomech Features** → RF (100% AUC)
5. **Zero-shot ODI** → 15.9% high-risk

**Live demo:** share.streamlit.io link
**Repo:** github.com/YOUR/cricket-injury-demo
""")
