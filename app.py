import os, io, csv
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from openai import OpenAI

# ---- Page config ----
st.set_page_config(page_title="Data → Insights Copilot", layout="wide")
st.title("📊 Data → Insights Copilot")
st.caption("Upload a CSV, preview it, and get AI-driven insights (token-efficient).")

# ---- Optional: enlarge spinner & caption text a bit ----
st.markdown("""
<style>
div[data-testid="stSpinner"] p { font-size: 1.1rem; font-weight: 600; }
div[data-testid="stCaptionContainer"] { font-size: 0.95rem; }
</style>
""", unsafe_allow_html=True)

# ---- Domain selector (alphabetical) ----
with st.sidebar:
    domain = st.selectbox(
        "Focus area for recommendations",
        [
            "Auto (data-driven only)",
            "Customer Support / Success",
            "Engineering / Data",
            "Finance",
            "HR / People",
            "Marketing / Growth",
            "Operations",
            "Product",
            "Risk / Compliance",
            "Sales",
        ],
        index=0,
    )

# ---- Robust CSV loader ----
def read_csv_robust(upload):
    """
    Read weird CSVs reliably:
    - Handles UTF-8 BOM
    - Guesses delimiter (comma/semicolon/tab/pipe)
    - Falls back if the first row is numeric (so it's not used as header)
    - Tries two-row header (common in finance exports)
    """
    raw_bytes = upload.read()
    if not raw_bytes or raw_bytes.strip() == b"":
        raise ValueError("Empty file")

    text = raw_bytes.decode("utf-8-sig", errors="replace")

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
        first_line = text.splitlines()[0] if text.splitlines() else ""
        if ";" in first_line:
            sep = ";"
        elif "\t" in first_line:
            sep = "\t"
        elif "|" in first_line:
            sep = "|"
        else:
            sep = ","

    # 2a) try with sep, header=0
    try:
        df = pd.read_csv(io.StringIO(text), sep=sep)
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

    # 2c) two-row header (common in Yahoo/finance dumps)
    try:
        tmp = pd.read_csv(io.StringIO(text), sep=sep, header=[0,1])
        tmp.columns = [" ".join([str(a), str(b)]).strip() for a,b in tmp.columns.to_list()]
        return tmp
    except Exception:
        pass

    # final fallback: single column
    lines = [ln for ln in text.splitlines() if ln.strip() != ""]
    df = pd.DataFrame({"text": lines})
    return df

# ---- Numeric coercion (turn "1,234" / "12.5%" / "-" into floats) ----
def coerce_numeric_columns(df, min_fraction=0.6):
    import numpy as np
    new_df = df.copy()
    for col in df.columns:
        if new_df[col].dtype == "object":
            s = new_df[col].astype(str).str.replace(",", "", regex=False).str.replace("%", "", regex=False).str.strip()
            s = s.replace({"-": None, "": None})
            coerced = pd.to_numeric(s, errors="coerce")
            non_null = coerced.notna().sum()
            if len(coerced) and (non_null / len(coerced) >= min_fraction):
                new_df[col] = coerced
    return new_df

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
        df = coerce_numeric_columns(df)  # NEW: make numeric-looking columns truly numeric

        st.subheader("👀 Data Preview")
        st.dataframe(df.head(20))

        profile_text = build_flexible_profile(df)

        # ---- Decide how much data to send to the model ----
        csv_text = ""
        try:
            csv_text = df.to_csv(index=False)
        except Exception:
            pass

        approx_tokens = len(csv_text) / 4 if csv_text else 999_999

        if approx_tokens < 5000:
            mode = "FULL CSV"
            user_notice = "Using the entire dataset."
            data_block = f"FULL CSV:\n{csv_text}"
        else:
            mode = "PROFILE"
            user_notice = (
                "File is large. To keep it fast and within the demo budget, "
                "the AI sees a compact profile containing:\n"
                "• Shape (rows × columns)\n"
                "• Column names\n"
                "• Sample of the first 20 rows\n"
                "• Numeric summary (mean/std/min/max/quartiles) for numeric columns\n"
                "• Categorical summary (unique/top/freq) for text columns\n"
                "Tip: Results may be broader than with full data."
            )
            data_block = f"PROFILE:\n{profile_text}"

        # ---- Friendly notice about what the AI will read ----
        ROWS, COLS = df.shape
        PROFILE_HEAD = 20  # this is what build_flexible_profile uses for head()

        if approx_tokens < 5000:
            # FULL CSV path
            st.success(
                f"✅ **Using the entire file** (~{int(approx_tokens):,} tokens). "
                "The AI will analyze **all rows and columns**."
            )
        else:
            # PROFILE path
            st.warning(
                f"""### 📦 Large file detected
        To keep things fast and within the demo budget, the AI will read a **compact summary** of your file (not every row).

        **What the AI will read**
        - Total size: **{ROWS:,} rows × {COLS} columns**
        - **All column names**
        - A preview of the **first {PROFILE_HEAD} rows**
        - **Numeric column stats** (mean, std, min, max, quartiles)
        - **Text column stats** (unique values, most frequent, frequency)

        *Tip:* Insights may be broader than if we sent the entire file.
        """,
            )


        # Show friendly badge so users know exactly what we did
        st.caption(f"Mode: **{mode}** • {user_notice}")

        # ---- Quick Chart(s) ----
        st.subheader("📈 Quick Chart")
        num_cols = df.select_dtypes(include="number").columns
        if len(num_cols) > 0:
            col_x = st.selectbox("Choose column for X-axis", options=df.columns, key="x")
            col_y = st.selectbox("Choose numeric column for Y-axis", options=num_cols, key="y")
            if col_x and col_y:
                fig, ax = plt.subplots()
                df.plot.scatter(x=col_x, y=col_y, ax=ax)
                st.pyplot(fig)
        else:
            st.info("No numeric values detected after parsing — chart skipped. (You can still generate AI insights below.)")

        # ---- Optional: simple finance-aware plots if typical columns exist ----
        lc = [c.lower().strip() for c in df.columns]
        def col_match(name, aliases):
            for c in df.columns:
                if c.lower().strip() == name or c.lower().strip() in aliases:
                    return c
            return None

        strike_col = col_match("strike", {"strike"})
        last_col   = col_match("last", {"last price","last_trade_price","last price "})
        iv_col     = col_match("implied volatility", {"iv","implied volatility"})
        type_col   = col_match("type", {"option type"})

        if strike_col and last_col:
            st.write("### Strike vs Last Price")
            fig, ax = plt.subplots()
            if type_col and df[type_col].dropna().nunique() <= 3:
                for t, sub in df.groupby(df[type_col].fillna("Unknown")):
                    sub.plot.scatter(x=strike_col, y=last_col, ax=ax, label=str(t))
            else:
                df.plot.scatter(x=strike_col, y=last_col, ax=ax)
            st.pyplot(fig)

        if strike_col and iv_col:
            st.write("### Strike vs Implied Volatility")
            fig, ax = plt.subplots()
            df.plot.scatter(x=strike_col, y=iv_col, ax=ax)
            st.pyplot(fig)

        # ---- AI Insights ----
        st.subheader("🤖 AI Insights")

        api_key = st.secrets.get("OPENAI_API_KEY", os.getenv("OPENAI_API_KEY", ""))
        if not api_key:
            st.warning("No API key found. Set OPENAI_API_KEY in Streamlit → Settings → Secrets.")
        else:
            client = OpenAI(api_key=api_key)

            # Domain guidance (generic; not hard-coded to options)
            domain_hints = {
                "Auto (data-driven only)": """
Constraints:
- Actions must be tactical and directly tied to the dataset’s columns (mention fields/thresholds).
- Avoid organizational/process/marketing advice unless explicitly grounded in the data.
""",
                "Finance": """
Constraints (Finance):
- Focus on financial signals in the data: price/return behavior, volatility/variance, spreads/margins, volume/turnover,
  cash-flow or P&L drivers, risk/outlier detection, time-series patterns, and benchmark comparisons if present.
- Prefer actions like: restrict to liquid/in-range instruments; set/adjust thresholds (e.g., spread/variance/limit alerts);
  use limit/conditional orders or rebalance windows; monitor spikes/divergences relative to recent baseline.
- Do NOT propose org/team/process/marketing changes unless they are explicitly evidenced by the data.
""",
                "Marketing / Growth": """
Constraints (Marketing/Growth):
- Focus on acquisition, activation, conversion, retention, channel and creative performance, CAC/LTV proxies if present.
- Prefer actions like: reallocate to higher-ROI channels, test X vs Y, tighten targeting based on segment lift.
""",
                "Sales": """
Constraints (Sales):
- Focus on pipeline stages, win/loss, ACV, velocity, territory/segment and rep performance.
- Prefer actions like: fix stage bottlenecks, target high-propensity segments, tighten qualification rules.
""",
                "Product": """
Constraints (Product):
- Focus on usage cohorts, feature adoption, activation, retention/churn drivers, and key drop-off steps.
- Prefer actions like: ship quick wins for sticky features, address drop-off points, A/B critical flows.
""",
                "Operations": """
Constraints (Operations):
- Focus on throughput, cycle time, SLA, defects, costs, and bottlenecks.
- Prefer actions like: set SLA/threshold alerts, remove queues, standardize high-variance steps.
""",
                "Customer Support / Success": """
Constraints (CS/CX):
- Focus on CSAT/NPS, first response/resolution time, contact drivers, churn risk signals.
- Prefer actions like: deflect top drivers, improve time-to-first-response, proactive outreach to at-risk cohorts.
""",
                "HR / People": """
Constraints (HR/People):
- Focus on hiring funnel, time-to-fill, retention, performance distribution, engagement.
- Prefer actions like: strengthen top-of-funnel sources, address attrition hotspots, refine leveling/comp bands.
""",
                "Risk / Compliance": """
Constraints (Risk/Compliance):
- Focus on anomalies, threshold breaches, exposure concentration, control failures.
- Prefer actions like: escalate breaches, set tighter limits/alerts, add monitoring on high-risk segments.
""",
                "Engineering / Data": """
Constraints (Eng/Data):
- Focus on latency, error rates, infra cost, data quality/freshness, pipeline reliability.
- Prefer actions like: fix P95 outliers, add alerts, optimize hot paths, backfill data gaps.
""",
            }
            selected_hint = domain_hints.get(domain, "")

            if st.button("✨ Generate AI Insights", key="ai_button"):
                with st.spinner("⏳ Processing, please wait..."):
                    prompt = f"""
You are a concise data analyst.

Please analyse the data.

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

{selected_hint}
""".strip()

                    try:
                        resp = client.chat.completions.create(
                            model="gpt-4o-mini",
                            messages=[{"role": "user", "content": prompt}],
                            temperature=0.4,
                            max_tokens=700,
                        )
                        st.success("✅ Done!")
                        st.markdown(resp.choices[0].message.content)
                        st.caption("⚡ Powered by gpt-4o-mini (full-data mode engages automatically for small files)")
                    except Exception as e:
                        st.error(f"OpenAI error: {e}")

    except Exception as e:
        st.error(f"Error reading CSV: {e}")
else:
    st.info("⬆️ Upload a CSV to begin.")
