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
# --- 1. CONFIGURATION & JLPT LOADING ---
# ===============================================

GITHUB_BASE_URL = "https://raw.githubusercontent.com/prihantoro-corpus/vocabulary-profiler/main/"
JLPT_FILES = {
    "N1": "unknown_source_N1.csv",
    "N2": "unknown_source_N2.csv",
    "N3": "unknown_source_N3.csv",
    "N4": "unknown_source_N4.csv",
    "N5": "unknown_source_N5.csv"
}

@st.cache_data
def load_jlpt_wordlists():
    """Fetches JLPT wordlists from GitHub and returns a dict of sets."""
    wordlists = {}
    for level, filename in JLPT_FILES.items():
        try:
            url = GITHUB_BASE_URL + filename
            df = pd.read_csv(url)
            # Assuming the CSV has a column named 'word' or it's the first column
            wordlists[level] = set(df.iloc[:, 0].astype(str).tolist())
        except Exception as e:
            st.error(f"Error loading {level} list: {e}")
            wordlists[level] = set()
    return wordlists

# ===============================================
# --- 2. INTERPRETATION & TOOLTIPS ---
# ===============================================

TOOLTIPS = {
    "Tokens": "Corpus size: The total number of all morphemes/words detected by the tokenizer (including repetitions).",
    "TTR": "Type-Token Ratio (V/N). Thresholds: < 0.45: Low diversity | 0.45 - 0.65: Moderate | > 0.65: High.",
    "MTLD": "Measuring Textual Lexical Diversity. Thresholds: < 40: Basic | 40 - 80: Intermediate | > 80: Advanced.",
    "Readability": "JReadability Score. Scale: 0.5-1.5: Upper-advanced | 2.5-3.5: Upper-intermediate | 4.5-5.5: Upper-elementary.",
    "JGRI": "Japanese Grammar Readability Index (Relative Complexity). Positive scores are more complex than the corpus average.",
    "JLPT": "Distribution of words based on JLPT levels. 'NA' indicates words not found in N1-N5 lists."
}

def get_jread_level(score):
    if 0.5 <= score < 1.5: return "Upper-advanced"
    elif 1.5 <= score < 2.5: return "Lower-advanced"
    elif 2.5 <= score < 3.5: return "Upper-intermediate"
    elif 3.5 <= score < 4.5: return "Lower-intermediate"
    elif 4.5 <= score < 5.5: return "Upper-elementary"
    elif 5.5 <= score < 6.5: return "Lower-elementary"
    else: return "Other"

def get_jgri_interpretation(val):
    if val < -1.0: return "Very easy"
    elif -1.0 <= val < 0: return "Relatively easy"
    elif 0 <= val < 1.0: return "Medium complexity"
    else: return "High complexity"

# ===============================================
# --- 3. LINGUISTIC ANALYSIS ENGINE ---
# ===============================================

def analyze_text(text, tagger, jlpt_lists):
    nodes = tagger(text)
    # Filter nodes to exclude punctuation
    valid_nodes = [n for n in nodes if n.surface and n.feature.pos1 != "補助記号"]
    sentences = [s for s in re.split(r'[。！？\n]', text.strip()) if s.strip()]
    num_sentences = len(sentences) if sentences else 1
    
    # POS/Script counts for JReadability
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
    
    # JLPT Mapping
    jlpt_counts = {level: 0 for level in jlpt_lists.keys()}
    jlpt_counts["NA"] = 0
    
    for n in valid_nodes:
        lemma = n.feature.orth if hasattr(n.feature, 'orth') else n.surface
        found = False
        for level, words in jlpt_lists.items():
            if lemma in words:
                jlpt_counts[level] += 1
                found = True
                break
        if not found:
            jlpt_counts["NA"] += 1

    # Formulas
    total_tokens = len(valid_nodes)
    wps = total_tokens / num_sentences
    pk, pw, pv, pp = [(x / total_tokens * 100) if total_tokens > 0 else 0 
                      for x in [scripts["K"], scripts["H"], len(verbs), len(particles)]]
    
    jread_score = (11.724 + (wps * -0.056) + (pk * -0.126) + (pw * -0.042) + (pv * -0.145) + (pp * -0.044))

    return {
        "surfaces": [n.surface for n in valid_nodes],
        "metrics": {
            "Tokens": total_tokens,
            "Readability": round(jread_score, 3),
            "WPS": round(wps, 2), "K_Full": round(pk, 2), "W_Full": round(pw, 2), 
            "V_Full": round(pv, 2), "P_Full": round(pp, 2),
            "MMS": total_tokens / num_sentences, 
            "LD": len(content_words) / total_tokens if total_tokens > 0 else 0,
            "VPS": len(verbs) / num_sentences, 
            "MPN": len(adverbs) / len(nouns) if len(nouns) > 0 else 0
        },
        "scripts": {k: round((v/total_tokens)*100, 1) if total_tokens > 0 else 0 for k, v in scripts.items()},
        "jlpt": jlpt_counts
    }

# ===============================================
# --- 4. STREAMLIT UI ---
# ===============================================

st.set_page_config(layout="wide", page_title="Japanese Lexical Profiler")

# Auth
pwd = st.sidebar.text_input("If you are a developer, tester, or reviewer, enter password", type="password")
if pwd != "290683":
    st.info("Please enter the password in the sidebar to proceed.")
    st.stop()

# Initialize
tagger = Tagger()
jlpt_wordlists = load_jlpt_wordlists()

st.title("📖 Japanese Text Vocabulary Profiler")

# Sidebar Config
st.sidebar.header("N-Gram Config")
n_exact = st.sidebar.number_input("Exact N value", 1, 5, 1)
n_range = st.sidebar.text_input("Range N values (e.g., 2, 4)", "")

source = st.sidebar.selectbox("Data Source", ["Upload Files", "DICO-JALF 30", "DICO-JALF ALL"])
corpus = []

# Data Loading Logic (Simplified for brevity)
if source == "Upload Files":
    up = st.sidebar.file_uploader("Upload .txt or .xlsx", accept_multiple_files=True)
    if up:
        for f in up: corpus.append({"name": f.name, "text": f.read().decode('utf-8')})
else:
    # URL mapping for DICO-JALF
    dico_url = "https://raw.githubusercontent.com/prihantoro-corpus/vocabulary-profiler/main/DICO-JALF%20all.xlsx" if "ALL" in source else "https://raw.githubusercontent.com/prihantoro-corpus/vocabulary-profiler/main/DICO-JALF%2030%20files%20only.xlsx"
    resp = requests.get(dico_url)
    df_pre = pd.read_excel(io.BytesIO(resp.content), header=None)
    corpus = [{"name": str(r[0]), "text": str(r[1])} for _, r in df_pre.iterrows()]

if corpus:
    results, global_tokens = [], []

    for item in corpus:
        data = analyze_text(item['text'], tagger, jlpt_wordlists)
        global_tokens.extend(data["surfaces"])
        lr = LexicalRichness(" ".join(data["surfaces"])) if data["surfaces"] else None
        
        # Build Result Row
        total = data["metrics"]["Tokens"]
        row = {
            "File": item['name'],
            "Tokens": total,
            "TTR": round(len(set(data['surfaces']))/total, 3) if total > 0 else 0,
            "MTLD": round(lr.mtld(), 2) if lr and total > 10 else 0,
            "Readability": data["metrics"]["Readability"],
            "J-Level": get_jread_level(data["metrics"]["Readability"]),
            "WPS": data["metrics"]["WPS"],
            "Percentage Kango": data["metrics"]["K_Full"],
            "Percentage Wago": data["metrics"]["W_Full"],
            "Percentage Verbs": data["metrics"]["V_Full"],
            "Percentage Particles": data["metrics"]["P_Full"],
            "Kanji%": data["scripts"]["K"], "Hira%": data["scripts"]["H"],
            "MMS": data["metrics"]["MMS"], "LD": data["metrics"]["LD"], 
            "VPS": data["metrics"]["VPS"], "MPN": data["metrics"]["MPN"]
        }
        
        # Add JLPT Columns
        for lvl in ["N1", "N2", "N3", "N4", "N5", "NA"]:
            count = data["jlpt"][lvl]
            row[lvl] = count
            row[f"{lvl}%"] = round((count / total * 100), 1) if total > 0 else 0
            
        results.append(row)

    df = pd.DataFrame(results)
    # Calculate JGRI
    for col in ["MMS", "LD", "VPS", "MPN"]:
        df[f"z_{col}"] = zscore(df[col]) if df[col].std() != 0 else 0
    df["JGRI"] = df[[f"z_{col}" for col in ["MMS", "LD", "VPS", "MPN"]]].mean(axis=1).round(3)
    df["Complexity"] = df["JGRI"].apply(get_jgri_interpretation)

    # --- DISPLAY ---
    st.header("Analysis Matrix")
    
    # Configure tooltips for new columns
    jlpt_cols = []
    for l in ["N1", "N2", "N3", "N4", "N5", "NA"]:
        jlpt_cols.extend([l, f"{l}%"])

    column_config = {c: st.column_config.NumberColumn(c, help=TOOLTIPS["JLPT"]) for c in jlpt_cols}
    column_config.update({
        "Tokens": st.column_config.NumberColumn("Tokens", help=TOOLTIPS["Tokens"]),
        "TTR": st.column_config.NumberColumn("TTR", help=TOOLTIPS["TTR"]),
        "Readability": st.column_config.NumberColumn("Readability", help=TOOLTIPS["Readability"]),
        "JGRI": st.column_config.NumberColumn("JGRI", help=TOOLTIPS["JGRI"])
    })

    main_cols = ["File", "Tokens", "TTR", "MTLD", "Readability", "J-Level", "JGRI", "Complexity"] + jlpt_cols + \
                ["WPS", "Percentage Kango", "Percentage Wago", "Percentage Verbs", "Percentage Particles", "Kanji%", "Hira%"]
    
    st.dataframe(df[main_cols], column_config=column_config, use_container_width=True)

    # --- N-GRAMS ---
    st.divider()
    st.header("N-Gram Frequencies")
    total_corpus_tokens = len(global_tokens)
    
    # Range parsing logic
    req_ns = [n_exact]
    if n_range:
        try:
            parts = [int(x.strip()) for x in n_range.split(",") if x.strip()]
            if len(parts) >= 2: req_ns.extend(list(range(min(parts), max(parts) + 1)))
        except: pass
    
    unique_ns = sorted(list(set([n for n in req_ns if 1 <= n <= 5])))
    cols = st.columns(len(unique_ns))
    for i, n in enumerate(unique_ns):
        with cols[i]:
            st.subheader(f"{n}-Gram")
            grams = [" ".join(global_tokens[j:j+n]) for j in range(len(global_tokens)-n+1)]
            df_g = pd.DataFrame(Counter(grams).most_common(10), columns=['Sequence', 'Raw Freq'])
            df_g['Freq (PMW)'] = df_g['Raw Freq'].apply(lambda x: round((x / total_corpus_tokens) * 1_000_000, 2))
            st.dataframe(df_g, hide_index=True)

else:
    st.info("Awaiting data input...")
