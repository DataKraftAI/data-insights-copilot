import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

import os
import textwrap

try:
    from openai import OpenAI
    OPENAI_READY = True
except Exception:
    OPENAI_READY = False

def build_profile(df, date_col, value_col, cat_col, audience, company_type):
    parts = [f"Rows: {len(df)}, Cols: {len(df.columns)}"]
    if value_col and value_col in df.columns and str(df[value_col].dtype) != "object":
        parts.append(f"Value column: {value_col}, mean={df[value_col].mean():.2f}, median={df[value_col].median():.2f}, sum={df[value_col].sum():.2f}")
    if cat_col and cat_col in df.columns:
        top = df[cat_col].value_counts().head(5).to_dict()
        parts.append(f"Top {cat_col} (count): {top}")
    if date_col and date_col in df.columns:
        dts = pd.to_datetime(df[date_col], errors="coerce")
        if dts.notna().any():
            parts.append(f"Date range: {str(dts.min())} → {str(dts.max())}")
    if company_type:
        parts.append(f"Company type: {company_type}")
    parts.append(f"Audience: {audience}")
    return "\n".join(parts)

st.subheader("Insights & Actions (AI)")
st.caption("Uses an LLM to propose concrete, non-generic actions based on the quick profile above.")

colA, colB = st.columns([1,1])
with colA:
    use_ai = st.checkbox("Enable AI insights", value=True)
with colB:
    temp = st.slider("Creativity", 0.0, 1.0, 0.2, 0.1)

if use_ai:
    profile_text = build_profile(df, date_col, value_col, cat_col, audience, company_type)

    if st.button("✨ Generate AI Insights"):
        if not OPENAI_READY:
            st.error("OpenAI SDK not installed. Add `openai` to requirements.txt")
            st.stop()

        # Prefer reading from Streamlit Secrets (set this in Streamlit Cloud)
        api_key = st.secrets.get("OPENAI_API_KEY", os.getenv("OPENAI_API_KEY", ""))

        if not api_key:
            st.info("No API key found. Add OPENAI_API_KEY to Streamlit Secrets to enable server-side calls.")
            st.stop()

        client = OpenAI(api_key=api_key)

        prompt = f"""
You are a senior business analyst.

Context:
- Company type: {company_type or 'N/A'}
- Audience: {audience}

Data summary (compact):
{profile_text}

Task:
1) Identify 3–5 notable patterns/trends/anomalies.
2) Explain likely business causes in plain language.
3) Output 3 prioritized actions with expected impact (Low/Med/High) and why.
4) Keep it concise and non-generic. Avoid clichés.

Format:
- **Findings**: bullet list
- **Causes**: bullet list
- **Actions**: numbered list with (Impact: …) and a one-line rationale
        """.strip()

        try:
            resp = client.chat.completions.create(
                model="gpt-4o-mini",        # fast, inexpensive, good quality
                messages=[{"role": "user", "content": prompt}],
                temperature=float(temp),
                max_tokens=600,
            )
            st.markdown(resp.choices[0].message.content)
        except Exception as e:
            msg = str(e).lower()
            if "quota" in msg or "insufficient" in msg or "exceeded" in msg:
                st.warning("⚠️ Demo limit exceeded for this month. Please check back next month.")
            else:
                st.error(f"OpenAI error: {e}")
else:
    st.info("Toggle ‘Enable AI insights’ to generate findings and actions.")


hide_footer_style = """
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    .stDeployButton {visibility: hidden;}
    </style>
"""
st.markdown(hide_footer_style, unsafe_allow_html=True)


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
    st.info("This feature uses OpenAI for insights. Disabled in demo mode.")

