import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np

st.set_page_config(layout="wide")
st.title("🤖 AI Data Analyst Agent")
st.markdown("**Sube CSV → Charts + Stats | Portfolio Damian Ocampo**")

uploaded_file = st.file_uploader("📁 Sube CSV", type="csv")

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.success(f"✅ Dataset: {df.shape[0]:,} filas × {df.shape[1]} columnas")
    
    # Stats
    col1, col2, col3 = st.columns(3)
    col1.metric("Filas", df.shape[0])
    col2.metric("Columnas", df.shape[1])
    col3.metric("Nulos", df.isnull().sum().sum())
    
    # Histogram (SIEMPRE funciona)
    st.subheader("📊 Histogram Automático")
    numeric_cols = df.select_dtypes(np.number).columns
    if len(numeric_cols) > 0:
        fig = px.histogram(df, x=numeric_cols[0], nbins=30, title=f"Distribución {numeric_cols[0]}")
        st.plotly_chart(fig, use_container_width=True)
    
    # DataFrame preview
    st.subheader("📋 Preview Dataset")
    st.dataframe(df.head())
    
    # Info
    st.subheader("ℹ️ Info Dataset")
    st.json({
        "Shape": list(df.shape),
        "Memory MB": round(df.memory_usage().sum() / 1024**2, 1),
        "Nulos por columna": df.isnull().sum().to_dict()
    })

st.caption("ML Portfolio Project | Streamlit + Plotly | 2026")
