import os
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from openai import OpenAI

# --- httpx/openai compatibility guard ---
import httpx
from packaging import version
if version.parse(httpx.__version__) >= version.parse("0.28.0"):
    st.error(f"Detected httpx {httpx.__version__} (incompatible with openai 1.40.3). "
             "Pin httpx==0.27.2 and httpcore<1.0.0 in requirements.txt, then restart.")
    st.stop()

st.set_page_config(page_title="Data → Insights Copilot", layout="wide")

st.title("📊 Data → Insights Copilot")
st.caption("Upload a CSV, preview it, and get AI-driven insights.")

uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

def build_flexible_profile(df, max_rows=20):
    profile = []
    profile.append(f"Shape: {df.shape[0]} rows × {df.shape[1]} columns")
    profile.append(f"Columns: {list(df.columns)}")
    sample = df.head(min(max_rows, len(df)))
    profile.append("Sample rows:")
    profile.append(sample.to_csv(index=False))
    num_summary = df.describe(include="number").transpose().round(2)
    if not num_summary.empty:
        profile.append("Numeric summary (mean, std, min, max):")
        profile.append(num_summary.to_csv())
    cat_summary = df.describe(include="object").transpose()
    if not cat_summary.empty:
        profile.append("Categorical summary (unique counts, top, freq):")
        profile.append(cat_summary.to_csv())
    return "\n".join(profile)

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
        st.subheader("👀 Data Preview")
        st.dataframe(df.head(20))

        profile_text = build_flexible_profile(df)

        st.subheader("📈 Quick Chart (if numeric)")
        num_cols = df.select_dtypes(include="number").columns
        if len(num_cols) > 0:
            col_x = st.selectbox("Choose column for X-axis", options=df.columns, key="x")
            col_y = st.selectbox("Choose numeric column for Y-axis", options=num_cols, key="y")
            if col_x and col_y:
                fig, ax = plt.subplots()
                df.plot.scatter(x=col_x, y=col_y, ax=ax)
                st.pyplot(fig)
        else:
            st.info("No numeric columns available for charting.")

        st.subheader("🤖 AI Insights")
        api_key = st.secrets.get("OPENAI_API_KEY", os.getenv("OPENAI_API_KEY", ""))
        if not api_key:
            st.warning("No API key found. Please set OPENAI_API_KEY in Streamlit secrets.")
            st.stop()

        client = OpenAI(api_key=api_key)

        if st.button("✨ Generate AI Insights", key="ai_button"):
            prompt = f"""
You are a senior analyst. A CSV dataset was uploaded.

Data profile:
{profile_text}

Task:
1) Identify 3–5 notable patterns/trends/anomalies.
2) Explain likely causes in plain language.
3) Output 3 prioritized actions with expected impact (Low/Med/High) and why.

Format:
- **Findings**
- **Causes**
- **Actions**
""".strip()

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
else:
    st.info("⬆️ Upload a CSV to begin.")
