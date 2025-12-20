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
# --- 1. CORE LINGUISTIC FUNCTIONS ---
# ===============================================

def analyze_text(text, tagger):
    """Performs morphological analysis and extracts all required metrics."""
    nodes = tagger(text)
    valid_nodes = [n for n in nodes if n.surface]
    
    # Sentence splitting for readability/density
    sentences = [s for s in re.split(r'[。！？\n]', text.strip()) if s.strip()]
    num_sentences = len(sentences) if sentences else 1
    
    # 1.1 Token collection (excluding punctuation for specific metrics)
    surfaces = [n.surface for n in valid_nodes if n.feature.pos1 != "補助記号"]
    
    # 1.2 POS Counting for JReadability and JGRI
    verbs = [n for n in valid_nodes if n.feature.pos1 == "動詞"]
    particles = [n for n in valid_nodes if n.feature.pos1 == "助詞"]
    nouns = [n for n in valid_nodes if n.feature.pos1 == "名詞"]
    adverbs = [n for n in valid_nodes if n.feature.pos1 == "副詞"]
    content_words = [n for n in valid_nodes if n.feature.pos1 in ["名詞", "動詞", "形容詞", "副詞"]]
    
    # 1.3 Script Distribution
    scripts = {"Kanji": 0, "Hiragana": 0, "Katakana": 0, "Other": 0}
    for n in valid_nodes:
        if re.search(r'[\u4e00-\u9faf]', n.surface): scripts["Kanji"] += 1
        elif re.search(r'[\u3040-\u309f]', n.surface): scripts["Hiragana"] += 1
        elif re.search(r'[\u30a0-\u30ff]', n.surface): scripts["Katakana"] += 1
        else: scripts["Other"] += 1
    
    total_chars = sum(scripts.values())
    script_dist = {k: round((v/total_chars)*100, 2) if total_chars > 0 else 0 for k, v in scripts.items()}

    # 1.4 JReadability Components (Hasebe & Lee 2015)
    total_tokens = len(valid_nodes)
    avg_char_sent = sum(len(s) for s in sentences) / num_sentences
    pct_kango = (scripts["Kanji"] / total_tokens * 100) if total_tokens > 0 else 0
    pct_wago = (scripts["Hiragana"] / total_tokens * 100) if total_tokens > 0 else 0
    pct_verbs = (len(verbs) / total_tokens * 100) if total_tokens > 0 else 0
    pct_particles = (len(particles) / total_tokens * 100) if total_tokens > 0 else 0
    
    jread_score = 11.724 - (0.056 * avg_char_sent) - (0.126 * pct_kango) - (0.042 * pct_wago) - (0.145 * pct_verbs) - (0.044 * pct_particles)

    # 1.5 JGRI Raw Components
    mms = len(valid_nodes) / num_sentences  # Morphemes per Sentence
    ld = len(content_words) / total_tokens if total_tokens > 0 else 0 # Lexical Density
    vps = len(verbs) / num_sentences # Verbs per Sentence
    mpn = len(adverbs) / len(nouns) if len(nouns) > 0 else 0 # Adverbs per Noun

    return {
        "surfaces": surfaces,
        "script_dist": script_dist,
        "jread": {
            "Score": round(jread_score, 3),
            "AvgCharSent": round(avg_char_sent, 2),
            "PctKango": round(pct_kango, 2),
            "PctWago": round(pct_wago, 2),
            "PctVerbs": round(pct_verbs, 2),
            "PctParticles": round(pct_particles, 2)
        },
        "jgri_raw": {"MMS": mms, "LD": ld, "VPS": vps, "MPN": mpn}
    }

# ===============================================
# --- 2. PRELOADED DATA ---
# ===============================================

PRELOADED_CORPORA = {
    "DICO-JALF ALL": "https://raw.githubusercontent.com/prihantoro-corpus/vocabulary-profiler/main/DICO-JALF%20all.xlsx",
    "DICO-JALF 30 Files Only": "https://raw.githubusercontent.com/prihantoro-corpus/vocabulary-profiler/main/DICO-JALF%2030%20files%20only.xlsx",
}

class RemoteFile:
    def __init__(self, name, content):
        self.name = name
        self.content = content
    def read(self): return self.content.encode('utf-8')

def fetch_preloaded(url):
    try:
        resp = requests.get(url)
        df = pd.read_excel(io.BytesIO(resp.content), header=None)
        return [RemoteFile(str(r[0]), str(r[1])) for _, r in df.iterrows()]
    except: return []

# ===============================================
# --- 3. UI AND SIDEBAR ---
# ===============================================

st.set_page_config(layout="wide", page_title="Japanese Lexical Profiler")

# Auth
pwd = st.sidebar.text_input("If you are a developer, tester, or reviewer, enter password", type="password")
if pwd != "290683":
    st.info("Please enter the password in the sidebar to proceed.")
    st.stop()

st.title("📖 Japanese Text Vocabulary Profiler")

# Data Source
source = st.sidebar.selectbox("1. Data Source", ["Upload Files", "DICO-JALF 30", "DICO-JALF ALL"])
files = []
if source == "Upload Files":
    up = st.sidebar.file_uploader("Upload .txt or .xlsx", accept_multiple_files=True)
    if up: files = up
else:
    files = fetch_preloaded(PRELOADED_CORPORA["DICO-JALF ALL" if "ALL" in source else "DICO-JALF 30 Files Only"])

if files:
    tagger = Tagger()
    results = []
    global_tokens = []

    for f in files:
        text = f.read().decode('utf-8')
        data = analyze_text(text, tagger)
        global_tokens.extend(data["surfaces"])
        
        # Lexical Richness
        lr = LexicalRichness(" ".join(data["surfaces"])) if data["surfaces"] else None
        
        res = {
            "File": f.name,
            "Tokens (N)": len(data["surfaces"]),
            "Types (V)": len(set(data["surfaces"])),
            "TTR": round(len(set(data['surfaces']))/len(data['surfaces']), 3) if data['surfaces'] else 0,
            "MTLD": round(lr.mtld(), 2) if lr and len(data["surfaces"]) > 10 else 0,
            **data["jread"],
            **data["jgri_raw"],
            **data["script_dist"]
        }
        results.append(res)

    # 4. JGRI CALCULATION (Z-Scores across corpus)
    df = pd.DataFrame(results)
    for col in ["MMS", "LD", "VPS", "MPN"]:
        df[f"z_{col}"] = zscore(df[col]) if df[col].std() != 0 else 0
    df["JGRI"] = df[[f"z_{col}" for col in ["MMS", "LD", "VPS", "MPN"]]].mean(axis=1).round(3)

    # --- SIDEBAR N-GRAM EXPANDER ---
    with st.sidebar.expander("🔍 N-Gram Expander", expanded=True):
        n_val = st.slider("Select N", 1, 5, 1)
        grams = [" ".join(global_tokens[i:i+n_val]) for i in range(len(global_tokens)-n_val+1)]
        st.dataframe(pd.DataFrame(Counter(grams).most_common(20), columns=['Sequence', 'Freq']), hide_index=True)

    # --- MAIN TABLE ---
    st.header("Analysis Matrix")
    
    # Final column ordering
    cols = [
        "File", "Tokens (N)", "Types (V)", "TTR", "MTLD", "Score", "JGRI",
        "AvgCharSent", "PctKango", "PctWago", "PctVerbs", "PctParticles",
        "Kanji", "Hiragana", "Katakana", "Other"
    ]
    
    final_df = df[cols].rename(columns={
        "Score": "JReadability",
        "AvgCharSent": "Avg Char/Sent", "PctKango": "Kango %", "PctWago": "Wago %",
        "PctVerbs": "Verb %", "PctParticles": "Particle %",
        "Kanji": "K %", "Hiragana": "H %", "Katakana": "T %", "Other": "O %"
    })
    
    st.dataframe(final_df, use_container_width=True)
    
    st.info("**JGRI Calculation:** Calculated as the average of normalized z-scores (MMS, LD, VPS, MPN) across the current corpus.")

else:
    st.info("Awaiting data input...")
