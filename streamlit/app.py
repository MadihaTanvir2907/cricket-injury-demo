# 🚀 ADD HIGH-RISK FRAME VISUALIZATION (Copy to GitHub app.py)

viz_app_code = '''
import streamlit as st
import joblib
import pandas as pd
import numpy as np
import plotly.express as px
from PIL import Image
import plotly.graph_objects as go

st.set_page_config(layout="wide", page_title="Cricket Injury Detection 🏏", page_icon="🏏")
st.title("🏏 Cricket Injury Detection Dashboard")
st.markdown("*Master's Thesis - Linnaeus University 2026*")

@st.cache_resource
def load_model():
    model = joblib.load("models/model.pkl")
    st.success(f"✅ Model loaded: {model.n_features_in_} features")
    return model

model = load_model()

# === SIDEBAR: 8-FEATURE INPUT ===
st.sidebar.header("🎾 Biomech Features")
knee_asym = st.sidebar.slider("1. knee_asym", 0.0, 5.0, 1.0)
hip_tilt = st.sidebar.slider("2. hip_tilt", 0.0, 2.0, 0.5)
feat3 = st.sidebar.slider("3. feat3", 0.0, 1.0, 0.8)
feat4 = st.sidebar.slider("4. feat4", 0.0, 180.0, 90.0)
feat5 = st.sidebar.slider("5. feat5", 0.0, 1.0, 0.5)
feat6 = st.sidebar.slider("6. feat6", 0.0, 1.0, 0.5)
feat7 = st.sidebar.slider("7. feat7", 0.0, 1.0, 0.5)
feat8 = st.sidebar.slider("8. feat8", 0.0, 1.0, 0.5)

input_df = pd.DataFrame([[
    knee_asym, hip_tilt, feat3, feat4, feat5, feat6, feat7, feat8
]], columns=['knee_asym','hip_tilt','feat3','feat4','feat5','feat6','feat7','feat8'])

# === PREDICTION ===
col1, col2 = st.columns([2,1])
with col1:
    st.subheader("Input Features")
    st.dataframe(input_df)
with col2:
    risk = model.predict_proba(input_df)[0,1]
    st.metric("Injury Risk", f"{risk:.1%}")
    st.markdown("🟢**SAFE**" if risk < 0.6 else "🔴**HIGH RISK**")

# === HIGH-RISK FRAMES VISUALIZATION ===
st.header("🚨 High Risk Frames (Match1)")
try:
    df = pd.read_csv("data/match1_sample.csv")
    df['risk_high'] = df.eval('knee_asym > 2.0 or hip_tilt > 1.0')  # Demo threshold
    
    high_risk = df[df.risk_high].head(6)
    st.metric("High Risk Detections", len(high_risk))
    
    # FRAME GALLERY (Demo images - upload real frames to data/frames/)
    cols = st.columns(3)
    for idx, row in enumerate(high_risk.itertuples()):
        with cols[idx % 3]:
            st.error(f"**Frame {row.track_id}**")
            st.metric("Risk", f"{row.knee_asym:.2f} knee_asym")
            # st.image(f"data/frames/frame_{row.track_id}.jpg", caption="High risk pose")
            st.image("https://via.placeholder.com/300x200/ff4444/ffffff?text=High+Risk+Frame", 
                    caption=f"Player {row.track_id}: {row.knee_asym:.2f} asym")
    
    # Risk distribution
    fig = px.histogram(df, x='knee_asym', color='risk_high', 
                      title="Risk Distribution (Red=High Risk)")
    st.plotly_chart(fig, use_container_width=True)
    
except:
    st.info("📤 Upload frames to data/frames/ for live viz")

st.markdown("---")
st.caption("🎓 Linnaeus University | YOLO+ByteTrack+OpenPose+RF Pipeline")
'''

print("✅ Viz code ready! Copy to GitHub:")
print(viz_app_code)
