import os
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from openai import OpenAI

# --------------------
# Helpers
# --------------------
def robust_read_csv(uploaded_file):
    """Try multiple encodings / separators for CSV upload."""
    for enc in ["utf-8", "latin-1"]:
        try:
            return pd.read_csv(uploaded_file, encoding=enc)
        except Exception:
            continue
    return pd.read_csv(uploaded_file, sep=";", encoding="latin-1")

def build_flexible_profile(df, max_rows=20):
    """Profile summarizing dataset for large files."""
    profile_parts = []
    try:
        profile_parts.append(f"Shape: {df.shape[0]} rows × {df.shape[1]} cols\n")
        profile_parts.append("Columns: " + ", ".join(df.columns) + "\n")
        profile_parts.append("Sample rows:\n" + df.head(max_rows).to_csv(index=False) + "\n")

        numeric_summary = df.describe(include="number").to_csv()
        profile_parts.append("Numeric summary:\n" + numeric_summary + "\n")

        cat_summary = df.describe(include="object").to_csv()
        profile_parts.append("Categorical summary:\n" + cat_summary + "\n")
    except Exception as e:
        profile_parts.append(f"[Profile build failed: {e}]")

    return "\n".join(profile_parts)

# --------------------
# Streamlit UI
# --------------------
st.set_page_config(page_title="Data → Insights Copilot", layout="wide")

st.title("📊 Data → Insights Copilot")
st.write("Upload a CSV, preview it, and get AI-driven insights.")

# File upload
uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])
df = None
if uploaded_file is not None:
    try:
        df = robust_read_csv(uploaded_file)
        st.subheader("👀 Data Preview")
        st.dataframe(df.head(20))
    except Exception as e:
        st.error(f"Error reading CSV: {e}")

# --------------------
# Quick Chart
# --------------------
if df is not None:
    st.subheader("📈 Quick Chart")
    try:
        # Pick X and Y
        x_col = st.selectbox("Choose column for X-axis", options=df.columns)
        num_cols = df.select_dtypes(include="number").columns
        if len(num_cols) == 0:
            st.info("This file has no numeric columns suitable for charting.")
        else:
            y_col = st.selectbox("Choose numeric column for Y-axis", options=num_cols)
            fig, ax = plt.subplots()
            ax.plot(df[x_col], df[y_col])
            ax.set_xlabel(x_col)
            ax.set_ylabel(y_col)
            st.pyplot(fig)
    except Exception as e:
        st.error(f"Charting failed: {e}")

# --------------------
# AI Insights
# --------------------
if df is not None:
    st.subheader("🤖 AI Insights")

    creativity = st.slider("Creativity", 0.0, 1.0, 0.2)

    # Build CSV text and token estimate
    csv_text = ""
    try:
        csv_text = df.to_csv(index=False)
    except Exception:
        pass
    approx_tokens = len(csv_text) / 4 if csv_text else 999999

    # Decide mode
    profile_text = build_flexible_profile(df, max_rows=20)
    data_block = f"FULL CSV:\n{csv_text}" if approx_tokens < 5000 else f"PROFILE:\n{profile_text}"

    # ---- Friendly notice block (moved under AI Insights) ----
    ROWS, COLS = df.shape
    PROFILE_HEAD = 20

    if approx_tokens < 5000:
        st.markdown(
            f"""
### ✅ Full file will be analyzed
The AI will read **all rows and columns** of your file  
(~{ROWS:,} rows × {COLS} columns, ~{int(approx_tokens):,} tokens).
""",
        )
    else:
        st.markdown(
            f"""
### 📦 Large file detected
To keep things fast and within the demo budget, the AI will analyze a **compact summary** instead of every row.

**What the AI will see:**
- Total size: **{ROWS:,} rows × {COLS} columns**
- All **column names**
- First **{PROFILE_HEAD} rows** as a preview
- **Numeric stats** (mean, std, min, max, quartiles)
- **Text stats** (unique values, most frequent, frequency)

*Tip: Insights may be broader than if we sent the entire file.*
""",
        )

    # Button
    if st.button("✨ Generate AI Insights", key="ai_button"):
        with st.spinner("Processing, please wait..."):
            api_key = st.secrets.get("OPENAI_API_KEY", os.getenv("OPENAI_API_KEY", ""))
            if not api_key:
                st.warning("No API key found. Please set it in Streamlit secrets.")
            else:
                try:
                    client = OpenAI(api_key=api_key)
                    prompt = f"""
You are a concise data analyst.

The dataset was uploaded as a CSV.
{data_block}

Task:
1) Identify 3–5 notable patterns/trends/anomalies.
2) Explain likely causes in plain language (based on the data you see).
3) Output 3 prioritized actions with expected impact (Low/Med/High) and why.
"""
                    response = client.chat.completions.create(
                        model="gpt-4o-mini",
                        messages=[{"role": "user", "content": prompt}],
                        temperature=creativity,
                    )
                    ai_output = response.choices[0].message.content
                    st.markdown("### 📑 AI Analysis")
                    st.write(ai_output)
                except Exception as e:
                    st.error(f"AI call failed: {e}")
