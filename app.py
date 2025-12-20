import streamlit as st
import pandas as pd
import io
import os
import re
from collections import Counter
import numpy as np
import matplotlib.pyplot as plt 
import matplotlib.ticker as ticker 
import requests 

# Specialized Imports
try:
    from wordcloud import WordCloud
    from PIL import Image
    from lexicalrichness import LexicalRichness
    from fugashi import Tagger
except ImportError as e:
    st.error(f"Missing package: {e}. Please check your requirements.txt.")
    st.stop()

# ===============================================
# --- 1. JREADABILITY (Hasebe & Lee 2015) ---
# ===============================================

def analyze_jreadability(text, tagged_nodes):
    """Computes JReadability score based on regression coefficients."""
    # Split text into sentences for average calculation [cite: 10, 11]
    sentences = [s for s in re.split(r'[。！？\n]', text.strip()) if s.strip()]
    if not sentences:
        return {"a": 0, "b": 0, "c": 0, "d": 0, "e": 0, "JREAD": None}

    total_chars = sum(len(s) for s in sentences)
    a = total_chars / len(sentences) # Avg characters per sentence [cite: 11]

    valid_nodes = [n for n in tagged_nodes if n.surface]
    total_tokens = len(valid_nodes)
    if total_tokens == 0:
        return {"a": round(a,2), "b": 0, "c": 0, "d": 0, "e": 0, "JREAD": None}

    # Morphological categories for the formula [cite: 11]
    verbs = sum(1 for n in valid_nodes if n.feature.pos1 == "動詞")
    particles = sum(1 for n in valid_nodes if n.feature.pos1 == "助詞")

    # Script/Etymology approximations [cite: 11]
    kango = sum(1 for n in valid_nodes if re.fullmatch(r"[\u4E00-\u9FFF]+", n.surface))
    wago = sum(1 for n in valid_nodes if re.fullmatch(r"[\u3040-\u309F]+", n.surface))

    b = (kango / total_tokens) * 100
    c = (wago / total_tokens) * 100
    d = (verbs / total_tokens) * 100
    e = (particles / total_tokens) * 100

    # Hasebe & Lee Regression Formula [cite: 10, 11]
    X = 11.724 - (0.056 * a) - (0.126 * b) - (0.042 * c) - (0.145 * d) - (0.044 * e)

    return {
        "a": round(a, 2), "b": round(b, 2), "c": round(c, 2), 
        "d": round(d, 2), "e": round(e, 2), "JREAD": round(X, 3)
    }

# ===============================================
# --- 2. PRELOADED CORPORA CONFIGURATION ---
# ===============================================

PRELOADED_CORPORA = {
    "DICO-JALF ALL": "https://raw.githubusercontent.com/prihantoro-corpus/vocabulary-profiler/main/DICO-JALF%20all.xlsx",
    "DICO-JALF 30 Files Only": "https://raw.githubusercontent.com/prihantoro-corpus/vocabulary-profiler/main/DICO-JALF%2030%20files%20only.xlsx",
}

class MockUploadedFile:
    def __init__(self, name, data_io):
        self.name = name
        self._data_io = data_io
    def read(self):
        self._data_io.seek(0)
        return self._data_io.read()

def load_preloaded_corpus(url, name):
    """Fetches external Excel data and mocks it as uploaded files."""
    try:
        response = requests.get(url, allow_redirects=True)
        response.raise_for_status()
        data_io = io.BytesIO(response.content)
        df = pd.read_excel(data_io, header=None)
        
        mock_files = []
        for _, row in df.iterrows():
            filename = str(row.iloc[0]).strip()
            content = str(row.iloc[1]).strip()
            if filename != 'nan' and content != 'nan':
                mock_files.append(MockUploadedFile(filename, io.BytesIO(content.encode('utf-8'))))
        return mock_files
    except Exception as e:
        st.error(f"Failed to fetch {name}: {e}")
        return []

# ===============================================
# --- 3. MAIN APPLICATION INTERFACE ---
# ===============================================

def main():
    st.sidebar.title("Japanese Lexical Profiler")
    
    # Password Protection [cite: 2]
    password = st.sidebar.text_input("Enter Developer Password", type="password")
    if password != "290683":
        st.warning("Please enter the password to access the system.")
        st.stop()

    # Data Input Selection [cite: 7, 8]
    input_method = st.sidebar.selectbox("1. Load Corpus Source", 
        ["Upload Local Files", "Preloaded: DICO-JALF ALL", "Preloaded: DICO-JALF 30 Files"])

    final_files = []
    if "Upload" in input_method:
        uploaded = st.sidebar.file_uploader("Upload .txt files", accept_multiple_files=True)
        if uploaded: final_files = uploaded
    elif "ALL" in input_method:
        final_files = load_preloaded_corpus(PRELOADED_CORPORA["DICO-JALF ALL"], "DICO-JALF ALL")
    else:
        final_files = load_preloaded_corpus(PRELOADED_CORPORA["DICO-JALF 30 Files Only"], "DICO-JALF 30")

    if not final_files:
        st.info("Select a corpus source to begin analysis.")
        st.stop()

    # Morphological Analysis Initialization
    tagger = Tagger()
    results = []
    all_tokens = []

    # Process files
    for f in final_files:
        content = f.read().decode("utf-8")
        nodes = tagger(content)
        j_read = analyze_jreadability(content, nodes)
        
        surfaces = [n.surface for n in nodes if n.surface]
        all_tokens.extend(surfaces)
        
        # Calculate Diversity Metrics [cite: 12]
        lr = LexicalRichness(content)
        
        results.append({
            "File": f.name,
            "Tokens (N)": len(surfaces),
            "Types (V)": len(set(surfaces)),
            "TTR": round(len(set(surfaces))/len(surfaces), 3) if surfaces else 0,
            "MTLD": round(lr.mtld(), 2) if len(surfaces) > 0 else 0,
            "jReadability": j_read['JREAD']
        })

    # Display Matrix [cite: 9]
    st.header("Analysis Matrix")
    df_results = pd.DataFrame(results)
    st.dataframe(df_results)

    # Word Cloud Visualization
    st.header("Word Cloud")
    if all_tokens:
        wc = WordCloud(width=800, height=400, background_color='white').generate(" ".join(all_tokens))
        st.image(wc.to_array())

if __name__ == "__main__":
    main()
