import streamlit as st
import pandas as pd
import io
import re
import requests
import numpy as np
from collections import Counter
from fugashi import Tagger
from lexicalrichness import LexicalRichness

# ===============================================
# --- 1. JREADABILITY & JGRI LOGIC ---
# ===============================================

def analyze_morphology(text, tagger):
    """Performs tokenization and extracts linguistic features."""
    nodes = tagger(text)
    sentences = [s for s in re.split(r'[。！？\n]', text.strip()) if s.strip()]
    num_sentences = len(sentences) if sentences else 1
    
    valid_nodes = [n for n in nodes if n.surface]
    surfaces = [n.surface for n in valid_nodes if n.feature.pos1 != "補助記号"]
    
    # Counts for JReadability and JGRI
    verbs = [n for n in valid_nodes if n.feature.pos1 == "動詞"]
    particles = [n for n in valid_nodes if n.feature.pos1 == "助詞"]
    nouns = [n for n in valid_nodes if n.feature.pos1 == "名詞"]
    adverbs = [n for n in valid_nodes if n.feature.pos1 == "副詞"]
    content_words = [n for n in valid_nodes if n.feature.pos1 in ["名詞", "動詞", "形容詞", "副詞"]]
    
    # Script Detection
    scripts = {"K": 0, "H": 0, "T": 0, "O": 0}
    for n in valid_nodes:
        if re.search(r'[\u4e00-\u9faf]', n.surface): scripts["K"] += 1
        elif re.search(r'[\u3040-\u309f]', n.surface): scripts["H"] += 1
        elif re.search(r'[\u30a0-\u30ff]', n.surface): scripts["T"] += 1
        else: scripts["O"] += 1

    total_s = sum(scripts.values())
    script_dist = {k: round((v/total_s)*100, 1) if total_s > 0 else 0 for k, v in scripts.items()}

    # JReadability Components
    total_tokens = len(valid_nodes)
    a = sum(len(s) for s in sentences) / num_sentences
    b = (scripts["K"] / total_tokens * 100) if total_tokens > 0 else 0
    c = (scripts["H"] / total_tokens * 100) if total_tokens > 0 else 0 # Approximation for Wago
    d = (len(verbs) / total_tokens * 100) if total_tokens > 0 else 0
    e = (len(particles) / total_tokens * 100) if total_tokens > 0 else 0
    
    jread_score = 11.724 - (0.056 * a) - (0.126 * b) - (0.042 * c) - (0.145 * d) - (0.044 * e)

    # JGRI Raw Components
    mms = len(valid_nodes) / num_sentences
    ld = len(content_words) / total_tokens if total_tokens > 0 else 0
    vps = len(verbs) / num_sentences
    mpn = len(adverbs) / len(nouns) if len(nouns) > 0 else 0

    return {
        "surfaces": surfaces,
        "script_dist": script_dist,
        "jread_components": {"Avg Char/Sent": a, "% Kango": b, "% Wago": c, "% Verbs": d, "% Particles": e, "Score": round(jread_score, 3)},
        "jgri_raw": {"MMS": mms, "LD": ld, "VPS": vps, "MPN": mpn}
    }

def calculate_jgri(df):
    """Normalizes raw components to Z-scores and averages them for JGRI."""
    for col in ["MMS", "LD", "VPS", "MPN"]:
        if df[col].std() == 0:
            df[f"z_{col}"] = 0
        else:
            df[f"z_{col}"] = (df[col] - df[col].mean()) / df[col].std()
    
    df["JGRI"] = df[[f"z_{col}" for col in ["MMS", "LD", "VPS", "MPN"]]].mean(axis=1)
    return df

# ===============================================
# --- 2. PRELOADED CORPORA ---
# ===============================================

PRELOADED_CORPORA = {
    "DICO-JALF ALL": "https://raw.githubusercontent.com/prihantoro-corpus/vocabulary-profiler/main/DICO-JALF%20all.xlsx",
    "DICO-JALF 30 Files Only": "https://raw.githubusercontent.com/prihantoro-corpus/vocabulary-profiler/main/DICO-JALF%2030%20files%20only.xlsx",
}

class MockFile:
    def __init__(self, name, content):
        self.name, self.content = name, content
    def read(self): return self.content.encode('utf-8')

def fetch_corpus(url):
    try:
        df = pd.read_excel(io.BytesIO(requests.get(url).content), header=None)
        return [MockFile(str(r[0]), str(r[1])) for _, r in df.iterrows()]
    except: return []

# ===============================================
# --- 3. STREAMLIT UI ---
# ===============================================

st.set_page_config(layout="wide", page_title="Japanese Lexical Profiler")

# Sidebar Auth
if st.sidebar.text_input("Password", type="password") != "290683":
    st.info("Enter password 290683 to start.")
    st.stop()

st.title("📖 Japanese Text Vocabulary Profiler")

# Source Selection
source = st.sidebar.selectbox("1. Data Source", ["Upload Files", "DICO-JALF 30", "DICO-JALF ALL"])
selected_files = []

if source == "Upload Files":
    up = st.sidebar.file_uploader("Upload .txt or .xlsx", accept_multiple_files=True)
    if up: selected_files = up
else:
    url_key = "DICO-JALF ALL" if "ALL" in source else "DICO-JALF 30 Files Only"
    selected_files = fetch_corpus(PRELOADED_CORPORA[url_key])

if selected_files:
    tagger = Tagger()
    raw_data_list = []
    global_tokens = []

    for f in selected_files:
        text = f.read().decode('utf-8')
        analysis = analyze_morphology(text, tagger)
        global_tokens.extend(analysis["surfaces"])
        
        lr = LexicalRichness(" ".join(analysis["surfaces"])) if analysis["surfaces"] else None
        
        # Merge all data points
        entry = {
            "File": f.name,
            "Tokens (N)": len(analysis["surfaces"]),
            "Types (V)": len(set(analysis["surfaces"])),
            "TTR": round(len(set(analysis["surfaces"]))/len(analysis["surfaces"]), 3) if analysis["surfaces"] else 0,
            "MTLD": round(lr.mtld(), 2) if lr and len(analysis["surfaces"]) > 10 else 0,
            **analysis["jread_components"],
            **analysis["jgri_raw"],
            **analysis["script_dist"]
        }
        raw_data_list.append(entry)

    # Calculate JGRI across the corpus
    df_final = calculate_jgri(pd.DataFrame(raw_data_list))

    # --- Sidebar N-Gram Expander ---
    with st.sidebar.expander("🔍 N-Gram Frequency Expander", expanded=False):
        n_size = st.slider("N-Gram Size", 1, 5, 1)
        ngrams = [" ".join(global_tokens[i:i+n_size]) for i in range(len(global_tokens)-n_size+1)]
        st.dataframe(pd.DataFrame(Counter(ngrams).most_common(20), columns=['Sequence', 'Freq']), hide_index=True)

    # --- Main Analysis Matrix ---
    st.header("Analysis Matrix")
    
    # Organize columns for readability
    display_cols = [
        "File", "Tokens (N)", "Types (V)", "TTR", "MTLD", 
        "Score", "JGRI",  # Main scores
        "Avg Char/Sent", "% Kango", "% Wago", "% Verbs", "% Particles", # JRead components
        "K", "H", "T", "O" # Scripts
    ]
    
    # Rename Script columns for clarity
    df_display = df_final[display_cols].rename(columns={
        "Score": "JReadability",
        "K": "Kanji %", "H": "Hiragana %", "T": "Katakana %", "O": "Other %"
    })
    
    st.dataframe(df_display, use_container_width=True)

    # JGRI Component Explanation
    st.info("**JGRI (Relative Grammatical Complexity):** Normalized Z-scores based on MMS (Morphemes per Sentence), LD (Lexical Density), VPS (Verbs per Sentence), and MPN (Adverbs per Noun).")

else:
    st.info("Awaiting data input...")
