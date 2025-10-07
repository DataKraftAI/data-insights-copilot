import os, io, csv
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from openai import OpenAI

# ================================ Page setup & CSS ================================
st.set_page_config(page_title="Data → Insights Copilot", layout="wide")
st.markdown("""
<style>
/* Bigger primary action button, matching Minutes app vibe */
.stButton > button[kind="primary"]{
  padding: 12px 18px;
  font-size: 16px;
  font-weight: 700;
  border-radius: 10px;
}
/* Nicer spinner text */
div[data-testid="stSpinner"] p { font-size: 1.05rem; font-weight: 600; }
/* Right-align the top header language select */
.header-row { display: flex; align-items: center; justify-content: space-between; }
.header-lang { min-width: 220px; }
</style>
""", unsafe_allow_html=True)

# ============================== Language bootstrap ===============================
def get_query_lang_default() -> str:
    try:
        lang = (st.query_params.get("lang") or "en").lower()
    except Exception:
        try:
            params = st.experimental_get_query_params()
            lang = (params.get("lang", ["en"])[0] or "en").lower()
        except Exception:
            lang = "en"
    return "de" if lang.startswith("de") else "en"

if "lang" not in st.session_state:
    st.session_state["lang"] = get_query_lang_default()

def set_lang(new_lang: str):
    st.session_state["lang"] = new_lang
    try:
        st.query_params["lang"] = new_lang
    except Exception:
        st.experimental_set_query_params(lang=new_lang)
    st.rerun()

LANG = st.session_state["lang"]

# ==================================== i18n =======================================
TXT = {
    "en": {
        "title": "📊 Data → Insights Copilot",
        "caption": "Upload a CSV, preview it, and get AI-driven insights (token-efficient).",
        "lang_label": "Language / Sprache",
        "sidebar_domain": "Focus area for recommendations",
        "uploader": "Upload your CSV file",
        "preview_h": "👀 Data Preview",
        "chart_h": "📈 Quick Chart",
        "chart_x": "Choose column for X-axis",
        "chart_y": "Choose numeric column for Y-axis",
        "chart_no_num": "No numeric values detected after parsing — chart skipped. (You can still generate AI insights below.)",
        "ai_h": "🤖 AI Insights",
        "adv": "Advanced",
        "creativity": "Creativity",
        "creativity_help": "Lower = more factual/grounded. Higher = more speculative language and bolder suggestions.",
        "notice_full_h": "✅ Full file will be analyzed",
        "notice_full_p": "The AI will read **all rows and columns** of your file  \n(~{rows} rows × {cols} columns, ~{tok} tokens).",
        "notice_large_h": "📦 Large file detected",
        "notice_large_p": (
            "To keep things fast and within the demo budget, the AI will analyze a **compact summary** instead of every row.\n\n"
            "**What the AI will see:**\n"
            "- Total size: **{rows} rows × {cols} columns**\n"
            "- All **column names**\n"
            "- First **{head} rows** as a preview\n"
            "- **Numeric stats** (mean, std, min, max, quartiles)\n"
            "- **Text stats** (unique values, most frequent, frequency)\n\n"
            "*Tip: Insights may be broader than if we sent the entire file.*"
        ),
        "btn_generate": "✨ Generate AI Insights",
        "spinner": "⏳ Processing, please wait...",
        "done": "✅ Done!",
        "no_key": "No API key found. Set OPENAI_API_KEY in Streamlit → Settings → Secrets.",
        "openai_err": "OpenAI error: {err}",
        "upload_prompt": "⬆️ Upload a CSV to begin.",
        "strike_last": "### Strike vs Last Price",
        "strike_iv": "### Strike vs Implied Volatility",
        "caption_model": "⚡ Powered by gpt-4o-mini • Full data for small files, compact profile for large files.",
        # Domains
        "d_auto":"Auto (data-driven only)","d_cs":"Customer Support / Success","d_eng":"Engineering / Data",
        "d_fin":"Finance","d_hr":"HR / People","d_mkt":"Marketing / Growth","d_ops":"Operations",
        "d_prod":"Product","d_risk":"Risk / Compliance","d_sales":"Sales",
    },
    "de": {
        "title": "📊 Data → Insights Copilot",
        "caption": "CSV hochladen, Vorschau ansehen und KI-gestützte Insights erhalten (token-effizient).",
        "lang_label": "Sprache",
        "sidebar_domain": "Fokusbereich für Empfehlungen",
        "uploader": "CSV-Datei hochladen",
        "preview_h": "👀 Datenvorschau",
        "chart_h": "📈 Schnell-Chart",
        "chart_x": "Spalte für X-Achse wählen",
        "chart_y": "Numerische Spalte für Y-Achse wählen",
        "chart_no_num": "Keine numerischen Werte erkannt — Chart übersprungen. (KI-Insights unten sind weiterhin möglich.)",
        "ai_h": "🤖 KI-Insights",
        "adv": "Erweitert",
        "creativity": "Kreativität",
        "creativity_help": "Niedrig = faktenbasiert. Hoch = spekulativer und mutigere Vorschläge.",
        "notice_full_h": "✅ Ganze Datei wird analysiert",
        "notice_full_p": "Die KI liest **alle Zeilen und Spalten** Ihrer Datei  \n(~{rows} Zeilen × {cols} Spalten, ~{tok} Tokens).",
        "notice_large_h": "📦 Große Datei erkannt",
        "notice_large_p": (
            "Zur Geschwindigkeit und Budget-Schonung analysiert die KI eine **kompakte Zusammenfassung** statt jeder einzelnen Zeile.\n\n"
            "**Was die KI sieht:**\n"
            "- Gesamtgröße: **{rows} Zeilen × {cols} Spalten**\n"
            "- Alle **Spaltennamen**\n"
            "- Erste **{head} Zeilen** als Vorschau\n"
            "- **Numerik-Statistik** (Mittelwert, Std, Min, Max, Quartile)\n"
            "- **Text-Statistik** (Unique-Werte, häufigste Werte, Häufigkeit)\n\n"
            "*Hinweis: Insights sind ggf. breiter als bei vollständiger Datei.*"
        ),
        "btn_generate": "✨ KI-Insights erzeugen",
        "spinner": "⏳ Verarbeitung… bitte warten.",
        "done": "✅ Fertig!",
        "no_key": "Kein API-Key gefunden. OPENAI_API_KEY in Streamlit → Settings → Secrets setzen.",
        "openai_err": "OpenAI-Fehler: {err}",
        "upload_prompt": "⬆️ CSV hochladen, um zu starten.",
        "strike_last": "### Strike vs. Letzter Preis",
        "strike_iv": "### Strike vs. Implizite Volatilität",
        "caption_model": "⚡ Basis: gpt-4o-mini • Vollständige Daten für kleine Dateien, kompaktes Profil für große Dateien.",
        # Domains
        "d_auto":"Auto (nur datengetrieben)","d_cs":"Customer Support / Success","d_eng":"Engineering / Data",
        "d_fin":"Finanzen","d_hr":"HR / People","d_mkt":"Marketing / Growth","d_ops":"Operations",
        "d_prod":"Produkt","d_risk":"Risk / Compliance","d_sales":"Sales",
    },
}

DOMAIN_KEYS = [
    ("auto","d_auto"),("cs","d_cs"),("eng","d_eng"),("fin","d_fin"),
    ("hr","d_hr"),("mkt","d_mkt"),("ops","d_ops"),("prod","d_prod"),
    ("risk","d_risk"),("sales","d_sales"),
]

# ================================ Header row =====================================
st.markdown('<div class="header-row">', unsafe_allow_html=True)
col_left, col_right = st.columns([1,0.32])
with col_left:
    st.title(TXT[LANG]["title"])
    st.caption(TXT[LANG]["caption"])
with col_right:
    new_lang = st.selectbox(TXT[LANG]["lang_label"], ["English","Deutsch"],
                            index=(0 if LANG=="en" else 1), key="hdr_lang")
    picked = "en" if new_lang.startswith("English") else "de"
    if picked != LANG:
        set_lang(picked)
st.markdown('</div>', unsafe_allow_html=True)

# ============================== Sidebar (domain etc.) ============================
with st.sidebar:
    domain_labels = [TXT[LANG][label_key] for _, label_key in DOMAIN_KEYS]
    domain_choice = st.selectbox(TXT[LANG]["sidebar_domain"], domain_labels, index=0)
    inv_map = {TXT[LANG][label_key]: key for key, label_key in DOMAIN_KEYS}
    domain_key = inv_map.get(domain_choice, "auto")

# =============================== CSV helpers ====================================
def read_csv_robust(upload):
    raw_bytes = upload.read()
    if not raw_bytes or raw_bytes.strip() == b"": raise ValueError("Empty file")
    text = raw_bytes.decode("utf-8-sig", errors="replace")

    # 1) default
    try:
        df = pd.read_csv(io.StringIO(text))
        if df.shape[1] >= 1: return df
    except Exception: pass

    # 2) sniff delimiter
    try:
        dialect = csv.Sniffer().sniff(text.splitlines()[0])
        sep = dialect.delimiter
    except Exception:
        first_line = text.splitlines()[0] if text.splitlines() else ""
        if ";" in first_line: sep = ";"
        elif "\t" in first_line: sep = "\t"
        elif "|" in first_line: sep = "|"
        else: sep = ","

    try:
        df = pd.read_csv(io.StringIO(text), sep=sep)
        numeric_like = all(str(c).strip().lstrip("+-").replace(".","",1).isdigit() for c in df.columns)
        if numeric_like: raise ValueError("First row numeric, not header")
        if df.shape[1] >= 1: return df
    except Exception: pass

    try:
        df = pd.read_csv(io.StringIO(text), sep=sep, header=None)
        first_row = df.iloc[0].astype(str).tolist()
        looks_headerish = any(any(ch.isalpha() for ch in cell) for cell in first_row)
        if looks_headerish:
            df.columns = first_row; df = df.iloc[1:].reset_index(drop=True)
        else:
            df.columns = [f"col_{i}" for i in range(df.shape[1])]
        return df
    except Exception: pass

    try:
        tmp = pd.read_csv(io.StringIO(text), sep=sep, header=[0,1])
        tmp.columns = [" ".join([str(a), str(b)]).strip() for a,b in tmp.columns.to_list()]
        return tmp
    except Exception: pass

    lines = [ln for ln in text.splitlines() if ln.strip() != ""]
    return pd.DataFrame({"text": lines})

def coerce_numeric_columns(df, min_fraction=0.6):
    new_df = df.copy()
    for col in df.columns:
        if new_df[col].dtype == "object":
            s = new_df[col].astype(str).str.replace(",", "", regex=False).str.replace("%", "", regex=False).str.strip()
            s = s.replace({"-": None, "": None})
            coerced = pd.to_numeric(s, errors="coerce")
            if len(coerced) and (coerced.notna().sum()/len(coerced) >= min_fraction):
                new_df[col] = coerced
    return new_df

def build_flexible_profile(df, max_rows=20):
    if df is None or df.shape[1] == 0:
        return "Dataframe appears empty after parsing."
    parts = []
    parts.append(f"Shape: {df.shape[0]} rows × {df.shape[1]} columns")
    parts.append(f"Columns: {list(map(str, df.columns))}")
    sample = df.head(min(max_rows, len(df)))
    try:
        parts.append("Sample rows (CSV):"); parts.append(sample.to_csv(index=False))
    except Exception:
        parts.append("Sample rows could not be serialized to CSV.")
    try:
        num_df = df.select_dtypes(include="number")
        if num_df.shape[1] > 0:
            parts.append("Numeric summary (mean, std, min, max, quartiles):")
            parts.append(num_df.describe().transpose().round(2).to_csv())
    except Exception as e:
        parts.append(f"[Note] Numeric summary unavailable: {e}")
    try:
        obj_df = df.select_dtypes(include="object")
        if obj_df.shape[1] > 0:
            parts.append("Categorical summary (unique, top, freq):")
            parts.append(obj_df.describe().transpose().to_csv())
    except Exception as e:
        parts.append(f"[Note] Categorical summary unavailable: {e}")
    return "\n".join(parts)

# ================================ Upload & preview ===============================
uploaded = st.file_uploader(TXT[LANG]["uploader"], type=["csv"])
df = None
if uploaded is not None:
    try:
        df = read_csv_robust(uploaded)
        df = coerce_numeric_columns(df)
        st.subheader(TXT[LANG]["preview_h"])
        st.dataframe(df.head(20))
    except Exception as e:
        st.error(f"Error reading CSV: {e}")

# ================================== Charts ======================================
if df is not None:
    st.subheader(TXT[LANG]["chart_h"])
    num_cols = df.select_dtypes(include="number").columns
    if len(num_cols) > 0:
        col_x = st.selectbox(TXT[LANG]["chart_x"], options=df.columns, key="x")
        col_y = st.selectbox(TXT[LANG]["chart_y"], options=num_cols, key="y")
        if col_x and col_y:
            fig, ax = plt.subplots()
            df.plot.scatter(x=col_x, y=col_y, ax=ax)
            st.pyplot(fig)
    else:
        st.info(TXT[LANG]["chart_no_num"])

    # Optional finance-ish quick plots if columns detected
    def col_match(name, aliases):
        for c in df.columns:
            lc = c.lower().strip()
            if lc == name or lc in aliases:
                return c
        return None
    strike_col = col_match("strike", {"strike"})
    last_col   = col_match("last", {"last price","last_trade_price","last price "})
    iv_col     = col_match("implied volatility", {"iv","implied volatility"})
    type_col   = col_match("type", {"option type"})

    if strike_col and last_col:
        st.write(TXT[LANG]["strike_last"])
        fig, ax = plt.subplots()
        if type_col and df[type_col].dropna().nunique() <= 3:
            for t, sub in df.groupby(df[type_col].fillna("Unknown")):
                sub.plot.scatter(x=strike_col, y=last_col, ax=ax, label=str(t))
        else:
            df.plot.scatter(x=strike_col, y=last_col, ax=ax)
        st.pyplot(fig)

    if strike_col and iv_col:
        st.write(TXT[LANG]["strike_iv"])
        fig, ax = plt.subplots()
        df.plot.scatter(x=strike_col, y=iv_col, ax=ax)
        st.pyplot(fig)

# ================================== AI Insights ==================================
if df is not None:
    st.subheader(TXT[LANG]["ai_h"])

    with st.expander(TXT[LANG]["adv"]):
        creativity = st.slider(TXT[LANG]["creativity"], 0.0, 1.0, 0.20, help=TXT[LANG]["creativity_help"])

    # Prepare model input (token-aware)
    try:
        csv_text = df.to_csv(index=False)
    except Exception:
        csv_text = ""
    approx_tokens = len(csv_text)/4 if csv_text else 999_999

    PROFILE_HEAD = 20
    profile_text = build_flexible_profile(df, max_rows=PROFILE_HEAD)
    if approx_tokens < 5000:
        data_block = f"FULL CSV:\n{csv_text}"
    else:
        data_block = f"PROFILE:\n{profile_text}"

    ROWS, COLS = df.shape
    if approx_tokens < 5000:
        st.markdown(f"### {TXT[LANG]['notice_full_h']}\n" + TXT[LANG]['notice_full_p'].format(rows=f"{ROWS:,}", cols=COLS, tok=int(approx_tokens)))
    else:
        st.markdown(f"### {TXT[LANG]['notice_large_h']}\n" + TXT[LANG]['notice_large_p'].format(rows=f"{ROWS:,}", cols=COLS, head=PROFILE_HEAD))

    # Domain hints
    domain_hints_en = {
        "auto": "Constraints: tie actions to columns/thresholds in this dataset; avoid generic org/process/marketing advice.",
        "fin":  "Finance focus: returns, volatility, spreads, volume, margins; propose thresholds/alerts, rebalance, outlier checks.",
        "mkt":  "Marketing focus: acquisition/activation/retention; reallocate to ROI channels, test X vs Y, segment targeting.",
        "sales":"Sales focus: pipeline stages, win/loss, velocity; fix bottlenecks, prioritize high-propensity segments.",
        "prod": "Product focus: usage cohorts, adoption, retention drivers; quick wins, fix drop-offs, A/B critical flows.",
        "ops":  "Ops focus: throughput, SLA, defects, costs; add alerts, remove queues, standardize high-variance steps.",
        "cs":   "CS focus: CSAT/NPS, response/resolution time, contact drivers; deflect top drivers, proactive outreach.",
        "hr":   "HR focus: funnel, time-to-fill, retention; strengthen sources, address attrition hotspots.",
        "risk": "Risk focus: anomalies, threshold breaches, concentration; escalate breaches, tighten limits/monitoring.",
        "eng":  "Eng/Data focus: latency, error rates, infra cost, data quality; fix P95 outliers, add alerts, optimize hot paths.",
    }
    domain_hints_de = {
        "auto":"Hinweis: Maßnahmen strikt an Datenspalten/-werte binden; keine allgemeinen Orga-/Marketing-Tipps.",
        "fin":"Finance: Rendite/Volatilität/Spreads/Volumen; Schwellwerte/Alerts, Rebalancing, Ausreißer prüfen.",
        "mkt":"Marketing: Akquise/Conversion/Retention; Budget zu ROI-Kanälen, Tests X vs. Y, Segmentierung.",
        "sales":"Sales: Pipeline-Stufen, Win/Loss, Velocity; Engpässe beseitigen, Segmente priorisieren.",
        "prod":"Product: Nutzung/Adoption/Cohorts, Drop-offs; schnelle Maßnahmen, A/B kritischer Flows.",
        "ops":"Operations: Durchsatz, SLA, Fehler, Kosten; Alerts, Queues abbauen, Standardisierung.",
        "cs":"Support: CSAT/NPS, Reaktions-/Lösungszeit, Treiber; Deflection/Proaktivität.",
        "hr":"HR: Funnel, Time-to-Fill, Retention; Quellen stärken, Fluktuationspunkte adressieren.",
        "risk":"Risk: Anomalien, Grenzwertverletzungen, Konzentrationen; eskalieren, Limits/Monitoring.",
        "eng":"Eng/Data: Latenz, Fehlerquote, Kosten, Datenqualität; P95 fixen, Alerts, Hot Paths optimieren.",
    }
    selected_hint = (domain_hints_de if LANG=="de" else domain_hints_en).get(domain_key, "")

    # Big primary button (now prominent)
    if st.button(TXT[LANG]["btn_generate"], type="primary"):
        with st.spinner(TXT[LANG]["spinner"]):
            api_key = st.secrets.get("OPENAI_API_KEY", os.getenv("OPENAI_API_KEY", ""))
            if not api_key:
                st.warning(TXT[LANG]["no_key"])
            else:
                try:
                    client = OpenAI(api_key=api_key)
                    if LANG == "de":
                        prompt = f"""
Du bist ein präziser, knapper Datenanalyst.

Der Datensatz wurde als CSV hochgeladen.
{data_block}

Aufgabe:
1) Nenne 3–5 auffällige Muster/Trends/Anomalien.
2) Erkläre wahrscheinliche Ursachen in einfacher Sprache (nur aus den vorliegenden Daten ableiten).
3) Gib 3 priorisierte Maßnahmen mit erwartetem Impact (Low/Med/High) und kurzer Begründung.

Format:
- **Erkenntnisse**: Aufzählung
- **Ursachen**: Aufzählung
- **Maßnahmen**: nummerierte Liste mit (Impact: …) und Einzeiler-Rationale.

{selected_hint}
""".strip()
                    else:
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

{selected_hint}
""".strip()

                    resp = client.chat.completions.create(
                        model="gpt-4o-mini",
                        messages=[{"role": "user", "content": prompt}],
                        temperature=0.2,
                        max_tokens=700,
                    )
                    st.success(TXT[LANG]["done"])
                    st.markdown(resp.choices[0].message.content)
                    st.caption(TXT[LANG]["caption_model"])
                except Exception as e:
                    st.error(TXT[LANG]["openai_err"].format(err=e))
else:
    st.info(TXT[LANG]["upload_prompt"])
