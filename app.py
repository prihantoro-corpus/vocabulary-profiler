import streamlit as st
import pandas as pd
import io
import re
import requests
import numpy as np
from collections import Counter
from fugashi import Tagger
from lexicalrichness import LexicalRichness
from scipy.stats import zscore

# ===============================================
# --- 1. INTERPRETATION LOGIC ---
# ===============================================

def get_jread_level(score):
    if score is None: return "N/A"
    if 0.5 <= score < 1.5: return "Upper-advanced"
    elif 1.5 <= score < 2.5: return "Lower-advanced"
    elif 2.5 <= score < 3.5: return "Upper-intermediate"
    elif 3.5 <= score < 4.5: return "Lower-intermediate"
    elif 4.5 <= score < 5.5: return "Upper-elementary"
    elif 5.5 <= score < 6.5: return "Lower-elementary"
    elif score >= 6.5: return "Beginner"
    else: return "Expert/Technical"

def get_jgri_interpretation(val):
    if val < -1.0: return "Very easy / Conversational"
    elif -1.0 <= val < 0: return "Relatively easy"
    elif 0 <= val < 1.0: return "Medium complexity"
    else: return "High complexity"

def get_ttr_interpretation(val):
    """General interpretation of Type-Token Ratio."""
    if val < 0.45: return "Low diversity (Repetitive)"
    elif 0.45 <= val < 0.65: return "Moderate diversity"
    else: return "High diversity (Varied)"

def get_mtld_interpretation(val):
    """MTLD Interpretation (higher values = more diverse)."""
    if val < 40: return "Basic/Limited"
    elif 40 <= val < 80: return "Intermediate/Standard"
    else: return "Advanced/Highly Diverse"

# ===============================================
# --- 2. CORE LINGUISTIC FUNCTIONS ---
# ===============================================

def analyze_text(text, tagger):
    nodes = tagger(text)
    valid_nodes = [n for n in nodes if n.surface]
    sentences = [s for s in re.split(r'[。！？\n]', text.strip()) if s.strip()]
    num_sentences = len(sentences) if sentences else 1
    
    # 2.1 POS & Scripts
    verbs = [n for n in valid_nodes if n.feature.pos1 == "動詞"]
    particles = [n for n in valid_nodes if n.feature.pos1 == "助詞"]
    nouns = [n for n in valid_nodes if n.feature.pos1 == "名詞"]
    adverbs = [n for n in valid_nodes if n.feature.pos1 == "副詞"]
    content_words = [n for n in valid_nodes if n.feature.pos1 in ["名詞", "動詞", "形容詞", "副詞"]]
    
    scripts = {"K": 0, "H": 0, "T": 0, "O": 0}
    for n in valid_nodes:
        if re.search(r'[\u4e00-\u9faf]', n.surface): scripts["K"] += 1
        elif re.search(r'[\u3040-\u309f]', n.surface): scripts["H"] += 1
        elif re.search(r'[\u30a0-\u30ff]', n.surface): scripts["T"] += 1
        else: scripts["O"] += 1
    
    # 2.2 Rechecked JReadability Formula
    total_tokens = len(valid_nodes)
    wps = total_tokens / num_sentences
    pk = (scripts["K"] / total_tokens * 100) if total_tokens > 0 else 0
    pw = (scripts["H"] / total_tokens * 100) if total_tokens > 0 else 0
    pv = (len(verbs) / total_tokens * 100) if total_tokens > 0 else 0
    pp = (len(particles) / total_tokens * 100) if total_tokens > 0 else 0
    
    jread_score = (11.724 + (wps * -0.056) + (pk * -0.126) + 
                   (pw * -0.042) + (pv * -0.145) + (pp * -0.044))

    # 2.3 JGRI Components
    mms = total_tokens / num_sentences
    ld = len(content_words) / total_tokens if total_tokens > 0 else 0
    vps = len(verbs) / num_sentences
    mpn = len(adverbs) / len(nouns) if len(nouns) > 0 else 0

    return {
        "tokens": [n.surface for n in valid_nodes if n.feature.pos1 != "補助記号"],
        "jread": {
            "Score": round(jread_score, 3), "WPS": round(wps, 2), 
            "K%": round(pk, 2), "W%": round(pw, 2), "V%": round(pv, 2), "P%": round(pp, 2)
        },
        "jgri_raw": {"MMS": mms, "LD": ld, "VPS": vps, "MPN": mpn},
        "scripts": {k: round((v/sum(scripts.values()))*100, 1) for k, v in scripts.items()}
    }

# ===============================================
# --- 3. DATA FETCHING ---
# ===============================================

PRELOADED_CORPORA = {
    "DICO-JALF ALL": "https://raw.githubusercontent.com/prihantoro-corpus/vocabulary-profiler/main/DICO-JALF%20all.xlsx",
    "DICO-JALF 30 Files Only": "https://raw.githubusercontent.com/prihantoro-corpus/vocabulary-profiler/main/DICO-JALF%2030%20files%20only.xlsx",
}

def fetch_preloaded(url):
    try:
        resp = requests.get(url)
        df = pd.read_excel(io.BytesIO(resp.content), header=None)
        return [{"name": str(r[0]), "text": str(r[1])} for _, r in df.iterrows()]
    except: return []

# ===============================================
# --- 4. STREAMLIT UI ---
# ===============================================

st.set_page_config(layout="wide", page_title="Japanese Lexical Profiler")

# Sidebar Authentication
pwd = st.sidebar.text_input("If you are a developer, tester, or reviewer, enter password", type="password")
if pwd != "290683":
    st.info("Please enter the password in the sidebar to proceed.")
    st.stop()

st.title("📖 Japanese Text Vocabulary Profiler")

# Sidebar Configuration
st.sidebar.header("N-Gram Config")
n_exact = st.sidebar.number_input("Exact N value", 1, 5, 1)
n_range = st.sidebar.text_input("Range N values (e.g., 2, 4)", "")

source = st.sidebar.selectbox("Data Source", ["Upload Files", "DICO-JALF 30", "DICO-JALF ALL"])
corpus = []
if source == "Upload Files":
    up = st.sidebar.file_uploader("Upload .txt or .xlsx", accept_multiple_files=True)
    if up:
        for f in up: corpus.append({"name": f.name, "text": f.read().decode('utf-8')})
else:
    url_key = "DICO-JALF ALL" if "ALL" in source else "DICO-JALF 30 Files Only"
    corpus = fetch_preloaded(PRELOADED_CORPORA[url_key])

if corpus:
    tagger = Tagger()
    results = []
    global_tokens = []

    for item in corpus:
        data = analyze_text(item['text'], tagger)
        global_tokens.extend(data["tokens"])
        
        lr = LexicalRichness(" ".join(data["tokens"])) if data["tokens"] else None
        ttr_val = round(len(set(data['tokens']))/len(data['tokens']), 3) if data['tokens'] else 0
        mtld_val = round(lr.mtld(), 2) if lr and len(data["tokens"]) > 10 else 0
        
        res = {
            "File": item['name'],
            "Tokens": len(data["tokens"]),
            "TTR": ttr_val,
            "TTR_Desc": get_ttr_interpretation(ttr_val),
            "MTLD": mtld_val,
            "MTLD_Desc": get_mtld_interpretation(mtld_val),
            "Readability": data["jread"]["Score"],
            "J-Level": get_jread_level(data["jread"]["Score"]),
            **data["jread"],
            **data["jgri_raw"],
            **data["scripts"]
        }
        results.append(res)

    # 5. JGRI Relative Normalization
    df = pd.DataFrame(results)
    for col in ["MMS", "LD", "VPS", "MPN"]:
        df[f"z_{col}"] = zscore(df[col]) if df[col].std() != 0 else 0
    df["JGRI"] = df[[f"z_{col}" for col in ["MMS", "LD", "VPS", "MPN"]]].mean(axis=1).round(3)
    df["Complexity"] = df["JGRI"].apply(get_jgri_interpretation)

    # --- MAIN TABLE ---
    st.header("Analysis Matrix")
    display_cols = [
        "File", "Tokens", "TTR", "TTR_Desc", "MTLD", "MTLD_Desc", 
        "Readability", "J-Level", "JGRI", "Complexity",
        "WPS", "K%", "W%", "V%", "P%", "K", "H", "T", "O"
    ]
    st.dataframe(df[display_cols].rename(columns={
        "TTR_Desc": "TTR Interpretation", 
        "MTLD_Desc": "MTLD Interpretation",
        "K": "Kanji%", "H": "Hira%", "T": "Kata%", "O": "Other%"
    }), use_container_width=True)

    # --- N-GRAM TABLES (BELOW MATRIX) ---
    st.divider()
    st.header("N-Gram Frequencies")
    
    requested_ns = [n_exact]
    if n_range:
        try:
            parts = [int(x.strip()) for x in n_range.split(",") if x.strip()]
            if len(parts) >= 2:
                requested_ns.extend(list(range(min(parts), max(parts) + 1)))
        except: pass
    
    unique_ns = sorted(list(set([n for n in requested_ns if 1 <= n <= 5])))
    cols = st.columns(len(unique_ns))
    for i, n in enumerate(unique_ns):
        with cols[i]:
            st.subheader(f"{n}-Gram")
            grams = [" ".join(global_tokens[j:j+n]) for j in range(len(global_tokens)-n+1)]
            st.dataframe(pd.DataFrame(Counter(grams).most_common(10), columns=['Sequence', 'Freq']), hide_index=True)

else:
    st.info("Awaiting data input...")
