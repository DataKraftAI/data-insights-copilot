import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

st.set_page_config(page_title="Data → Insights Copilot", layout="wide")

st.title("📊 Data → Insights Copilot (Demo)")
st.write("Upload a CSV file and see quick insights + charts.")

# File uploader
uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    st.subheader("🔎 Preview of Data")
    st.write(df.head())

    st.subheader("📈 Basic Stats")
    st.write(df.describe())

    st.subheader("📊 Quick Chart")
    numeric_cols = df.select_dtypes(include=['float64','int64']).columns
    if len(numeric_cols) > 0:
        col = st.selectbox("Pick a numeric column for line chart:", numeric_cols)
        st.line_chart(df[col])
    else:
        st.write("No numeric columns available for charting.")

else:
    st.info("⬆️ Upload a CSV to begin")

