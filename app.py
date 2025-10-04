import os, io, csv
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from openai import OpenAI

# ---- Page config ----
st.set_page_config(page_title="Data → Insights Copilot", layout="wide")
st.title("📊 Data → Insights Copilot")
st.caption("Upload a CSV, preview it, and get AI-driven insights (token-efficient).")

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

            # --- Domain guidance (kept short + generic) ---
            domain_hints = {
                "Auto (data-driven only)": """
            Constraints:
            - Actions must be tactical and directly tied to the dataset’s columns (mention fields/thresholds).
            - Avoid organizational/process/marketing advice unless explicitly grounded in the data.
            """,
                "Finance": """
            Constraints (Finance):
            - Focus on financial signals in the data: e.g., price/return behavior, volatility/variance, spreads/margins, volume/turnover,
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

Constraints:
- Actions must be tactical and directly tied to the dataset’s columns (e.g., mention specific fields/thresholds).


{selected_hint}


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
