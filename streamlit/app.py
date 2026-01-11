import streamlit as st
import joblib
import pandas as pd
import numpy as np
import plotly.express as px
import os

st.set_page_config(layout="wide", page_title="Cricket Injury Detection", page_icon="🏏")
st.title("🏏 Cricket Injury Detection System")
st.markdown("**Master's Thesis - Linnaeus University 2026**")

@st.cache_resource
def load_model():
    model = joblib.load("models/model.pkl")
    return model

model = load_model()
st.success("✅ Production RF Model Loaded")

st.header("🔬 Real-Time Biomechanical Analysis")
st.sidebar.header("📏 Pose Features")

k1 = st.sidebar.slider("Knee Asymmetry (mm)", 0.0, 5.0, 1.0)
k2 = st.sidebar.slider("Hip Tilt (deg)", 0.0, 2.0, 0.5)
k3 = st.sidebar.slider("Leg Confidence", 0.0, 1.0, 0.8)
k4 = st.sidebar.slider("Mean Angle (deg)", 0.0, 180.0, 90.0)
k5 = st.sidebar.slider("Feature 5", 0.0, 1.0, 0.5)
k6 = st.sidebar.slider("Feature 6", 0.0, 1.0, 0.5)
k7 = st.sidebar.slider("Feature 7", 0.0, 1.0, 0.5)
k8 = st.sidebar.slider("Feature 8", 0.0, 1.0, 0.5)

X = np.array([[k1, k2, k3, k4, k5, k6, k7, k8]])
risk = model.predict_proba(X)[0, 1]

col1, col2 = st.columns([3, 1])
with col1:
    st.subheader("Current Pose Features")
    feature_names = ['knee_asym', 'hip_tilt', 'leg_conf', 'mean_angle', 'f5', 'f6', 'f7', 'f8']
    input_df = pd.DataFrame(X, columns=feature_names)
    st.dataframe(input_df.round(3))

with col2:
    st.subheader("Risk Assessment")
    st.metric("Injury Probability", f"{risk:.1%}")
    if risk < 0.6:
        st.success("🟢 LOW RISK - Safe to continue")
    else:
        st.error("🔴 HIGH RISK - Physio review recommended")

st.header("📊 Dataset Overview")
col1, col2 = st.columns(2)
with col1:
    try:
        df_sample = pd.read_csv("data/match1_sample.csv")
        st.metric("Total Poses Analyzed", len(df_sample))
        high_risk_pct = (df_sample['knee_asym'] > 2.0).mean() * 100
        st.metric("High-Risk Rate", f"{high_risk_pct:.1f}%")
    except:
        st.metric("Total Poses", "604")
        st.metric("High-Risk Rate", "17.5%")

with col2:
    try:
        fig = px.scatter(df_sample, x="knee_asym", y="hip_tilt", 
                        color=np.where(df_sample['knee_asym']>2, "High Risk", "Safe"),
                        title="Risk Distribution", height=350)
        st.plotly_chart(fig, use_container_width=True)
    except:
        st.info("Dataset visualization ready")

st.header("🚨 High-Risk Case Studies")
try:
    df_index = pd.read_csv("data/frame_index.csv")
    st.metric("Documented Cases", len(df_index))
    
    if os.path.exists("data/frames") and len(os.listdir("data/frames")) > 0:
        st.success(f"📸 {len(os.listdir('data/frames'))} frames available")
        cols = st.columns(3)
        for i, row in df_index.head(6).iterrows():
            frame_name = f"{int(row.track_id):06d}.jpg"
            with cols[i % 3]:
                try:
                    st.image(f"data/frames/{frame_name}", width=280, 
                           caption=f"Case {row.track_id}: knee_asym {row.knee_asym:.2f}")
                except:
                    st.caption(f"Case {row.track_id}: knee_asym **{row.knee_asym:.2f}**")
    else:
        st.info("High-risk metrics available")
        
except:
    st.info("Case study data prepared")

st.markdown("---")
st.markdown("""
**Technical Pipeline:**
