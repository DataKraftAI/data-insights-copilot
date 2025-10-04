import os
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from openai import OpenAI

st.set_page_config(page_title="Data → Insights Copilot", layout="wide")

st.title("📊 Data → Insights Copilot")
st.caption("Upload a CSV, preview it, and get AI-driven insights.")

# --- File upload ---
uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

# --- Flexible profile builder ---
def build_flexible_profile(df, max_rows=20):
    profile = []
    profile.append(f"Shape: {df.shape[0]} rows × {df.shape[1]} columns")
    profile.append(f"Columns: {list(df.columns)}")

    # Sample rows
    sample = df.head(min(max_rows, len(df)))
    profile.append("Sample rows:")
    profile.append(sample.to_csv(index=False))

    # Numeric stats
    num_summary = df.describe(include="number").transpose().round(2)
    if not num_summary.empty:
        profile.append("Numeric summary (mean, std, min, max):")
        profile.append(num_summary.to_csv())

    # Categorical stats
    cat_summary = df.describe(include="object").transpose()
    if not cat_summary.empty:
        profile.append("Categorical summary (unique counts, top, freq):")
        profile.append(cat_summary.to_csv())

    return "\n".join(profile)

# --- Main logic if file uploaded ---
if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)

        st.subheader("👀 Data Preview")
        st.dataframe(df.head(20))

        # Build profile
        profile_text = build_flexible_profile(df)

        st.subheader("📈 Quick Chart (if numeric)")
        num_cols = df.select_dtypes(include="number").columns
        if len(num_cols) > 0:
            col_x = st.selectbox("Choose column for X-axis", options=df.columns)
            col_y = st.selectbox("Choose numeric column for Y-axis", options=num_cols)
            if col_x and col_y:
                fig, ax = plt.subplots()
                df.plot.scatter(x=col_x, y=col_y, ax=ax)
                st.pyplot(fig)
        else:
            st.info("No numeric columns available for charting.")

        # --- AI Insights ---
        st.subheader("🤖 AI Insights")

        api_key = st.secrets.get("OPENAI_API_KEY", os.getenv("OPENAI_API_KEY", ""))
        if not api_key:
            st.warning("No API key found. Please set OPENAI_API_KEY in Streamlit secrets.")
            st.stop()
        else:
            client = OpenAI(api_key=api_key)

            if st.button("✨ Generate AI Insights", key="ai_button"):
                prompt = f"""
                You are a senior analyst.
                Here is a dataset summary:

                {profile_text}

                Task:
                1) Identify 3–5 notable patterns/trends/anomalies.
                2) Explain likely causes in plain language.
                3) Output 3 prioritized actions with expected impact (Low/Med/High) and why.

                Format:
                - **Findings**
                - **Causes**
                - **Actions**
                """

                try:
                    resp = client.chat.completions.create(
                        model="gpt-4o-mini",
                        messages=[{"role": "user", "content": prompt}],
                        temperature=0.4,
                        max_tokens=700,
                    )
                    st.markdown(resp.choices[0].message.content)
                    st.caption("⚡ Powered by gpt-4o-mini")

                except Exception as e:
                    st.error(f"OpenAI error: {e}")

    except Exception as e:
        st.error(f"Error reading CSV: {e}")
