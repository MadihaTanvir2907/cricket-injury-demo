import streamlit as st
import joblib
import pandas as pd
import numpy as np
import plotly.express as px

st.set_page_config(layout="wide", page_title="Cricket Injury Detection 🏏", page_icon="🏏")
st.title("🏏 Cricket Injury Detection System")
st.markdown("*Master's Thesis - Linnaeus University 2026*")

@st.cache_resource
def load_model():
    model = joblib.load("models/model.pkl")
    st.success(f"✅ Model loaded: {model.n_features_in_} features")
    return model

model = load_model()

st.sidebar.header("Biomechanics (8 features)")
# 8 DUMMY FEATURES matching your model order (knee_asym first)
knee_asym = st.sidebar.slider("1. knee_asym", 0.0, 5.0, 1.0)
hip_tilt = st.sidebar.slider("2. hip_tilt", 0.0, 2.0, 0.5)
feat3 = st.sidebar.slider("3. leg_conf", 0.5, 1.0, 0.8)
feat4 = st.sidebar.slider("4. mean_angle", 80.0, 120.0, 90.0)
feat5 = st.sidebar.slider("5. feat5", 0.0, 1.0, 0.5)
feat6 = st.sidebar.slider("6. feat6", 0.0, 1.0, 0.5)
feat7 = st.sidebar.slider("7. feat7", 0.0, 1.0, 0.5)
feat8 = st.sidebar.slider("8. feat8", 0.0, 1.0, 0.5)

# EXACT 8 columns (your model order)
input_df = pd.DataFrame({
    "knee_asym": [knee_asym],
    "hip_tilt": [hip_tilt],
    "leg_conf": [feat3],
    "mean_angle": [feat4],
    "feat5": [feat5],
    "feat6": [feat6],
    "feat7": [feat7],
    "feat8": [feat8]
})

col1, col2 = st.columns([2,1])
with col1:
    st.dataframe(input_df)
with col2:
    risk = model.predict_proba(input_df)[0,1]
    st.metric("Injury Risk", f"{risk:.1%}")
    st.markdown("🟢 SAFE" if risk < 0.6 else "🔴 HIGH RISK")

st.header("📊 Match1 Dataset")
df = pd.read_csv("data/match1_sample.csv")
st.dataframe(df.head())
fig = px.scatter(df, x="knee_asym", y="hip_tilt", title="Pose Distribution")
st.plotly_chart(fig)

st.caption("🎓 Linnaeus University - Multi-person YOLO+RF Pipeline")
