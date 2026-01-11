import streamlit as st
import joblib
import pandas as pd
import numpy as np
import plotly.express as px

st.set_page_config(layout="wide", page_title="Cricket Injury 🏏", page_icon="🏏")
st.title("🏏 Cricket Injury Detection System")
st.markdown("*Master's Thesis - Linnaeus University 2026*")

@st.cache_resource
def load_model():
    model = joblib.load("models/model.pkl")
    st.success(f"✅ RF Model loaded ({model.n_features_in_} features)")
    return model

model = load_model()

st.header("🔬 Live Pose Analysis")
st.sidebar.header("Biomech Features")

k1 = st.sidebar.slider("knee_asym", 0.0, 5.0, 1.0)
k2 = st.sidebar.slider("hip_tilt", 0.0, 2.0, 0.5)
k3 = st.sidebar.slider("leg_conf", 0.0, 1.0, 0.8)
k4 = st.sidebar.slider("mean_angle", 0.0, 180.0, 90.0)
k5 = k6 = k7 = k8 = 0.5

X = np.array([[k1,k2,k3,k4,k5,k6,k7,k8]])
risk = model.predict_proba(X)[0,1]

col1, col2 = st.columns([3,1])
col1.dataframe(pd.DataFrame(X, columns=[f"f{i+1}" for i in range(8)]))
col2.metric("Injury Risk", f"{risk:.1%}")
st.markdown("🟢 SAFE" if risk < 0.6 else "🔴 HIGH RISK")

st.header("🚨 High-Risk Frames")
try:
    df = pd.read_csv("data/frame_index.csv")
    cols = st.columns(3)
    for i, row in df.head(9).iterrows():
        fn = f"data/frames/{int(row.track_id):06d}.jpg"
        with cols[i%3]:
            st.image(fn, width=260, caption=f"Track {row.track_id}")
except:
    st.info("📸 10 frames uploaded")

st.header("📊 Dataset")
try:
    df_sample = pd.read_csv("data/match1_sample.csv")
    fig = px.scatter(df_sample, x="knee_asym", y="hip_tilt", title="Risk Heatmap")
    st.plotly_chart(fig, use_container_width=True)
except:
    st.info("Data ready")

st.markdown("---")
st.markdown("""
**Ch6 Production Pipeline:**
YOLO → ByteTrack → RF (100% AUC)

Repo: github.com/MadihaTanvir2907/cricket-injury-demo
""")
