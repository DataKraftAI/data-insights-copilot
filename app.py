import os, io, csv
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from openai import OpenAI

# ---- Page config ----
st.set_page_config(page_title="Data → Insights Copilot", layout="wide")
st.title("📊 Data → Insights Copilot")
st.caption("Upload a CSV, preview it, and get AI-driven insights (token-efficient).")

# ---- Robust CSV loader ----
def read_csv_robust(upload):
    """
    Read weird CSVs reliably:
    - Handles UTF-8 BOM
    - Guesses delimiter (comma/semicolon/tab/pipe)
    - Falls back if the first row is numeric (so it's not used as header)
    - Tries two-row header (common in finance exports)
    """
    # read bytes once; reuse
    raw_bytes = upload.read()
    if not raw_bytes or raw_bytes.strip() == b"":
        raise ValueError("Empty file")

    text = raw_bytes.decode("utf-8-sig", errors="replace")
    buf = io.StringIO(text)

    # 1) try vanilla
    try:
        df = pd.read_csv(io.StringIO(text))
        if df.shape[1] >= 1:
            return df
    except Exception:
        pass

    # 2) sniff delimiter
    try:
        dialect = csv.Sniffer().sniff(text.splitlines()[0])
        sep = dialect.delimiter
    except Exception:
        # fallback heuristics
        if ";" in text.splitlines()[0]:
            sep = ";"
        elif "\t" in text.splitlines()[0]:
            sep = "\t"
        elif "|" in text.splitlines()[0]:
            sep = "|"
        else:
            sep = ","

    # 2a) try with sep, header=0
    try:
        df = pd.read_csv(io.StringIO(text), sep=sep)
        # If header looks numeric (e.g., "24"), treat as no-header
        numeric_like = all(str(c).strip().lstrip("+-").replace(".","",1).isdigit() for c in df.columns)
        if numeric_like:
            raise ValueError("First row is numeric; not a real header")
        if df.shape[1] >= 1:
            return df
    except Exception:
        pass

    # 2b) try no header
    try:
        df = pd.read_csv(io.StringIO(text), sep=sep, header=None)
        # If first row is stringy headers, promote them
        first_row = df.iloc[0].astype(str).tolist()
        looks_headerish = any(any(ch.isalpha() for ch in cell) for cell in first_row)
        if looks_headerish:
            df.columns = first_row
            df = df.iloc[1:].reset_index(drop=True)
        else:
            df.columns = [f"col_{i}" for i in range(df.shape[1])]
        return df
    except Exception:
        pass

    # 2c) try two-row header (common in Yahoo/finance dumps)
    try:
        tmp = pd.read_csv(io.StringIO(text), sep=sep, header=[0,1])
        tmp.columns = [" ".join([str(a), str(b)]).strip() for a,b in tmp.columns.to_list()]
        return tmp
    except Exception:
        pass

    # final fallback: single column of lines
    lines = [ln for ln in text.splitlines() if ln.strip() != ""]
    df = pd.DataFrame({"text": lines})
    return df

# ---- Profile builder (safe) ----
def build_flexible_profile(df, max_rows=20):
    if df is None or df.shape[1] == 0:
        return "Dataframe appears empty after parsing."

    parts = []
    parts.append(f"Shape: {df.shape[0]} rows × {df.shape[1]} columns")
    parts.append(f"Columns: {list(map(str, df.columns))}")

    sample = df.head(min(max_rows, len(df)))
    try:
        parts.append("Sample rows (CSV):")
        parts.append(sample.to_csv(index=False))
    except Exception:
        parts.append("Sample rows (repr):")
        parts.append(repr(sample))

    # Numeric stats (guard empties)
    try:
        num_summary = df.select_dtypes(include="number")
        if num_summary.shape[1] > 0:
            desc = num_summary.describe().transpose().round(2)
            parts.append("Numeric summary (mean, std, min, max):")
            parts.append(desc.to_csv())
    except Exception as e:
        parts.append(f"[Note] Numeric summary unavailable: {e}")

    # Categorical stats (guard empties)
    try:
        obj_cols = df.select_dtypes(include="object")
        if obj_cols.shape[1] > 0:
            desc = obj_cols.describe().transpose()
            parts.append("Categorical summary (unique, top, freq):")
            parts.append(desc.to_csv())
    except Exception as e:
        parts.append(f"[Note] Categorical summary unavailable: {e}")

    return "\n".join(parts)

# ---- App body ----
uploaded = st.file_uploader("Upload your CSV file", type=["csv"])

if uploaded is not None:
    try:
        df = read_csv_robust(uploaded)

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

        # ---- AI Insights ----
        st.subheader("🤖 AI Insights")

        # Cheap model + single click
        api_key = st.secrets.get("OPENAI_API_KEY", os.getenv("OPENAI_API_KEY", ""))
        if not api_key:
            st.warning("No API key found. Set OPENAI_API_KEY in Streamlit → Settings → Secrets.")
        else:
            client = OpenAI(api_key=api_key)

            # Full-data mode for small files (≈<5k tokens)
            csv_text = ""
            try:
                csv_text = df.to_csv(index=False)
            except Exception:
                pass
            approx_tokens = len(csv_text) / 4 if csv_text else 999999
            data_block = f"FULL CSV:\n{csv_text}" if approx_tokens < 5000 else f"PROFILE:\n{profile_text}"

            if st.button("✨ Generate AI Insights", key="ai_button"):
                prompt = f"""
You are a concise data analyst.

The dataset was uploaded as a CSV.
{data_block}

Task:
1) Identify 3–5 notable patterns/trends/anomalies.
2) Explain likely causes in plain language (based on the data you see).
3) Output 3 prioritized actions with expected impact (Low/Med/High) and why.

Format:
- **Findings**: bullet list
- **Causes**: bullet list
- **Actions**: numbered list with (Impact: …) and a one-line rationale.
""".strip()

                try:
                    resp = client.chat.completions.create(
                        model="gpt-4o-mini",
                        messages=[{"role": "user", "content": prompt}],
                        temperature=0.4,
                        max_tokens=700,
                    )
                    st.markdown(resp.choices[0].message.content)
                    st.caption("⚡ Powered by gpt-4o-mini (full-data mode engages automatically for small files)")
                except Exception as e:
                    st.error(f"OpenAI error: {e}")

    except Exception as e:
        st.error(f"Error reading CSV: {e}")
else:
    st.info("⬆️ Upload a CSV to begin.")
