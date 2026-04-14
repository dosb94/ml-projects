import streamlit as st
import pandas as pd
import numpy as np
import lightgbm as lgb
import plotly.express as px

st.set_page_config(layout="wide")
st.title("🕵️ Credit Card Fraud Detection")
st.markdown("**LightGBM Production | 284K Transactions | Damian Ocampo**")

# Sidebar
st.sidebar.header("📁 Upload CSV")
uploaded_file = st.sidebar.file_uploader("Choose file", type="csv")

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.success(f"✅ Loaded {df.shape[0]:,} transactions")
    
    # Manual Robust Scaling (sin sklearn)
    median = df['Amount'].median()
    q75, q25 = df['Amount'].quantile(0.75), df['Amount'].quantile(0.25)
    df['scaled_amount'] = (df['Amount'] - median) / (q75 - q25)
    
    # Features (sin Time/Amount originales)
    feature_cols = [col for col in df.columns if col.startswith('V')]
    feature_cols.append('scaled_amount')
    X = df[feature_cols]
    y = df['Class']
    
    # LightGBM simple
    model = lgb.LGBMClassifier(n_estimators=100, random_state=42, verbose=-1)
    model.fit(X, y)
    
    # Predictions con tu threshold PRO
    probs = model.predict_proba(X)[:, 1]
    df['fraud_prob'] = probs
    df['fraud_pred'] = (probs >= 0.09).astype(int)
    
    # KPI Dashboard
    col1, col2, col3, col4 = st.columns(4)
    total_fraud = df['fraud_pred'].sum()
    real_fraud = y.sum()
    col1.metric("Transactions", f"{len(df):,}")
    col2.metric("Real Fraud", real_fraud)
    col3.metric("Detected", total_fraud)
    col4.metric("Recall", f"{total_fraud/real_fraud*100:.1f}%")
    
    # Charts
    st.subheader("📈 Fraud vs Normal Transactions")
    fig = px.histogram(df, x='scaled_amount', color='fraud_pred',
                      title="Amount Distribution by Fraud Prediction",
                      nbins=50, height=400)
    st.plotly_chart(fig, use_container_width=True)
    
    # Top fraud cases
    st.subheader("🔍 Top 10 Fraud Alerts")
    fraud_table = df[df['fraud_pred'] == 1][['scaled_amount', 'fraud_prob', 'Class']].head(10)
    st.dataframe(fraud_table.style.format({'fraud_prob': '{:.1%}'}))
    
st.caption("**Portfolio: github.com/dosb94/ml-projects**")
