import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

st.title("📊 Data → Insights Copilot")

uploaded_file = st.file_uploader("Upload a CSV file", type="csv")

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.write("### Preview of data", df.head())

    st.write("### Basic stats")
    st.write(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}")

    # Example chart (first numeric column)
    numeric_cols = df.select_dtypes(include="number").columns
    if len(numeric_cols) > 0:
        col = numeric_cols[0]
        st.write(f"### Distribution of `{col}`")
        fig, ax = plt.subplots()
        df[col].hist(ax=ax)
        st.pyplot(fig)
    else:
        st.info("No numeric columns found for plotting.")

# --- Fake AI Insights Button ---
if st.button("✨ Generate AI Insights"):
    st.warning("⚠️ This feature requires an OpenAI API key. (Disabled in demo)")
