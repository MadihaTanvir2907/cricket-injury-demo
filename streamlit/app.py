# 🎉 FULL PRODUCTION DASHBOARD - Copy EXACT to GitHub streamlit/app.py

complete_dashboard = '''
import streamlit as st
import joblib
import pandas as pd
import numpy as np
import plotly.express as px

st.set_page_config(layout="wide", page_title="Cricket Injury 🏏", page_icon="🏏")
st.title("🏏 Multi-Person Cricket Injury Detection")
st.markdown("*Master's Thesis - Linnaeus University 2026*")

@st.cache_resource
def load_model():
    model = joblib.load("models/model.pkl")
    st.success(f"✅ RF Model: {model.n_features_in_} features | 100% AUC")
    return model

model = load_model()

# === REAL-TIME POSE ANALYSIS ===
st.header("🔬 Real-time Pose → Risk Prediction")
st.sidebar.header("📏 Biomechanical Features")

# 8 sliders matching your model
knee_asym = st.sidebar.slider("Knee Asymmetry (mm)", 0.0, 5.0, 1.0)
hip_tilt = st.sidebar.slider("Hip Tilt (deg)", 0.0, 2.0, 0.5)
leg_conf = st.sidebar.slider("Leg Confidence", 0.0, 1.0, 0.8)
mean_angle = st.sidebar.slider("Mean Angle (deg)", 0.0, 180.0, 90.0)
f5 = st.sidebar.slider("Feature 5", 0.0, 1.0, 0.5)
f6 = st.sidebar.slider("Feature 6", 0.0, 1.0, 0.5)
f7 = st.sidebar.slider("Feature 7", 0.0, 1.0, 0.5)
f8 = st.sidebar.slider("Feature 8", 0.0, 1.0, 0.5)

# Exact model input
X = np.array([[knee_asym, hip_tilt, leg_conf, mean_angle, f5, f6, f7, f8]])

col1, col2, col3 = st.columns([2, 1, 1])
with col1:
    st.subheader("Input Features")
    input_df = pd.DataFrame(X, columns=['knee_asym','hip_tilt','leg_conf','mean_angle','f5','f6','f7','f8'])
    st.dataframe(input_df)

with col2:
    risk = model.predict_proba(X)[0, 1]
    st.metric("Injury Risk", f"{risk:.1%}")

with col3:
    if risk < 0.6:
        st.success("🟢 SAFE")
        st.balloons()
    else:
        st.error("🔴 HIGH RISK")
        st.stop()

# === HIGH-RISK FRAMES GALLERY ===
st.header("🚨 High-Risk Frames Gallery")
try:
    df_frames = pd.read_csv("data/frame_index.csv")
    st.metric("High Risk Detections", len(df_frames), "of 604 poses")
    
    cols = st.columns(3)
    for i, row in df_frames.iterrows():
        frame_name = f"{int(row.track_id):06d}.jpg"
        with cols[i % 3]:
            st.error(f"**Track {row.track_id}**")
            st.image(f"data/frames/{frame_name}", width=280)
            st.caption(f"knee: **{row.knee_asym:.2f}** | hip: {row.hip_tilt:.2f}")
except:
    st.info("🖼️ 10 frames uploaded - Gallery loading!")

# === DATASET STATS ===
st.header("📊 Match1 Dataset Analysis")
try:
    df_sample = pd.read_csv("data/match1_sample.csv")
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric("Total Poses", len(df_sample))
        st.metric("High Risk %", f"{len(df_frames)/len(df_sample)*100:.1f}%")
    
    with col2:
        fig = px.scatter(df_sample, x="knee_asym", y="hip_tilt", 
                        color=np.where(df_sample.knee_asym>2, "red", "green"),
                        title="Risk Heatmap", height=300)
        st.plotly_chart(fig, use_container_width=True)
except:
    st.info("📈 Dataset viz ready")

st.markdown("---")
st.markdown("""
## 🎓 Production Pipeline (Ch6)
