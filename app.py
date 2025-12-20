import streamlit as st
import pandas as pd
import io
import re
import requests
from collections import Counter
from fugashi import Tagger
from lexicalrichness import LexicalRichness

# ===============================================
# --- 1. JREADABILITY FORMULA ---
# ===============================================

def analyze_jreadability(text, tagged_nodes):
    """Computes JReadability score based on Hasebe & Lee (2015)."""
    sentences = [s for s in re.split(r'[。！？\n]', text.strip()) if s.strip()]
    if not sentences:
        return {"JREAD": 0, "avg_char": 0}

    total_chars = sum(len(s) for s in sentences)
    a = total_chars / len(sentences) # Avg characters per sentence

    valid_nodes = [n for n in tagged_nodes if n.surface]
    total_tokens = len(valid_nodes)
    if total_tokens == 0:
        return {"JREAD": 0, "avg_char": round(a, 2)}

    verbs = sum(1 for n in valid_nodes if n.feature.pos1 == "動詞")
    particles = sum(1 for n in valid_nodes if n.feature.pos1 == "助詞")
    kango = sum(1 for n in valid_nodes if re.fullmatch(r"[\u4E00-\u9FFF]+", n.surface))
    wago = sum(1 for n in valid_nodes if re.fullmatch(r"[\u3040-\u309F]+", n.surface))

    b = (kango / total_tokens) * 100
    c = (wago / total_tokens) * 100
    d = (verbs / total_tokens) * 100
    e = (particles / total_tokens) * 100

    # Regression constant and weights
    score = 11.724 - (0.056 * a) - (0.126 * b) - (0.042 * c) - (0.145 * d) - (0.044 * e)

    return {
        "JREAD": round(score, 3),
        "avg_char": round(a, 2),
        "pct_kango": round(b, 2),
        "pct_wago": round(c, 2)
    }

# ===============================================
# --- 2. PRELOADED CORPORA CONFIGURATION ---
# ===============================================

PRELOADED_CORPORA = {
    "DICO-JALF ALL": "https://raw.githubusercontent.com/prihantoro-corpus/vocabulary-profiler/main/DICO-JALF%20all.xlsx",
    "DICO-JALF 30 Files Only": "https://raw.githubusercontent.com/prihantoro-corpus/vocabulary-profiler/main/DICO-JALF%2030%20files%20only.xlsx",
}

class MockFile:
    def __init__(self, name, content):
        self.name = name
        self.content = content
    def read(self):
        return self.content.encode('utf-8')

def fetch_corpus(url, name):
    try:
        response = requests.get(url)
        response.raise_for_status()
        df = pd.read_excel(io.BytesIO(response.content), header=None)
        files = []
        for _, row in df.iterrows():
            files.append(MockFile(str(row[0]), str(row[1])))
        return files
    except Exception as e:
        st.error(f"Error loading {name}: {e}")
        return []

# ===============================================
# --- 3. UI AND MAIN LOGIC ---
# ===============================================

st.set_page_config(layout="wide", page_title="Japanese Vocabulary Profiler")

# Sidebar Auth
password = st.sidebar.text_input("Password", type="password")
if password != "290683":
    st.info("Please enter the password '290683' in the sidebar.")
    st.stop()

st.title("📖 Japanese Text Vocabulary Profiler")

# Input Method Selection
source_option = st.sidebar.selectbox("1. Data Source", 
    ["Upload Files", "DICO-JALF 30", "DICO-JALF ALL"])

selected_files = []
if source_option == "Upload Files":
    uploaded = st.sidebar.file_uploader("Upload .txt or .xlsx", accept_multiple_files=True)
    if uploaded:
        selected_files = uploaded
elif source_option == "DICO-JALF 30":
    selected_files = fetch_corpus(PRELOADED_CORPORA["DICO-JALF 30 Files Only"], "DICO-JALF 30")
elif source_option == "DICO-JALF ALL":
    selected_files = fetch_corpus(PRELOADED_CORPORA["DICO-JALF ALL"], "DICO-JALF ALL")

if selected_files:
    tagger = Tagger()
    all_results = []
    
    for f in selected_files:
        raw_text = f.read().decode('utf-8')
        nodes = tagger(raw_text)
        
        # 1. JReadability
        j_data = analyze_jreadability(raw_text, nodes)
        
        # 2. Lexical Richness (using surface tokens)
        surfaces = [n.surface for n in nodes if n.surface and n.feature.pos1 != "補助記号"]
        text_for_lr = " ".join(surfaces)
        lr = LexicalRichness(text_for_lr) if surfaces else None
        
        # 3. Script Distribution
        scripts = []
        for n in nodes:
            if re.search(r'[\u4e00-\u9faf]', n.surface): scripts.append("Kanji")
            elif re.search(r'[\u3040-\u309f]', n.surface): scripts.append("Hiragana")
            elif re.search(r'[\u30a0-\u30ff]', n.surface): scripts.append("Katakana")
            else: scripts.append("Other")
        
        script_counts = Counter(scripts)
        total_s = sum(script_counts.values())
        
        all_results.append({
            "File": f.name,
            "Tokens (N)": len(surfaces),
            "Types (V)": len(set(surfaces)),
            "TTR": round(len(set(surfaces))/len(surfaces), 3) if surfaces else 0,
            "MTLD": round(lr.mtld(), 2) if lr and len(surfaces) > 10 else 0,
            "JReadability": j_data['JREAD'],
            "Avg Char/Sent": j_data['avg_char'],
            "Kanji %": round((script_counts['Kanji']/total_s)*100, 1) if total_s > 0 else 0
        })

    # --- RESULTS TABLE ---
    st.header("2.3 Analysis Matrix")
    df_results = pd.DataFrame(all_results)
    st.dataframe(df_results, use_container_width=True)

    # --- N-GRAMS ---
    st.header("2.1 N-GRAM Frequency")
    n_size = st.slider("Select N-Gram Size", 1, 5, 1)
    
    # Flatten all tokens across files for global N-Gram
    global_tokens = []
    for f in selected_files:
        f.read() # Reset/dummy read if needed
        t_nodes = tagger(raw_text)
        global_tokens.extend([n.surface for n in t_nodes if n.surface and n.feature.pos1 != "補助記号"])
    
    ngrams = [" ".join(global_tokens[i:i+n_size]) for i in range(len(global_tokens)-n_size+1)]
    ngram_df = pd.DataFrame(Counter(ngrams).most_common(20), columns=['Sequence', 'Freq'])
    st.table(ngram_df)

else:
    st.info("Awaiting data input...")
