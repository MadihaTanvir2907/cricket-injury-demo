import streamlit as st
import joblib
import pandas as pd
import numpy as np

st.title("🏏 Cricket Injury Demo")
st.markdown("*Linnaeus University Thesis*")

try:
    model = joblib.load("models/model.pkl")
    st.success("✅ Model OK")
    
    knee = st.slider("Knee Asymmetry", 0.0, 5.0)
    risk = model.predict_proba([[knee,0.5,0.8,90,0.5,0.5,0.5,0.5]])[0,1]
    st.metric("Risk", f"{risk:.0%}")
    
    st.balloons()
except:
    st.write("Setup OK")
