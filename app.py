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
# --- 1. INTERPRETATION & TOOLTIPS ---
# ===============================================

TOOLTIPS = {
    "Tokens": "Corpus size: The total number of all morphemes/words detected by the tokenizer (including repetitions).",
    "TTR": "Type-Token Ratio. The ratio of unique words to total words. Higher ratios indicate more diverse vocabulary.",
    "MTLD": "Measuring Textual Lexical Diversity. A robust measure of vocabulary variety that is not affected by text length.",
    "Readability": "JReadability Score: A formula-based assessment of how easy or difficult the text is for Japanese learners.",
    "JGRI": "Japanese Grammar Readability Index: Measures relative grammatical complexity. Positive scores are more complex than the corpus average.",
    "WPS": "Words Per Sentence: The average number of tokens (words/morphemes) found in each sentence.",
    "Kango": "Percentage of Sino-Japanese words (words written in Kanji). High density usually indicates academic or formal text.",
    "Wago": "Percentage of Native Japanese words (usually written in Hiragana). High density indicates conversational or simpler text.",
    "Verbs": "Percentage of verbs in the text. Verbs often increase the dynamic nature but also the structural complexity of a sentence.",
    "Particles": "Percentage of grammatical particles (助詞). These are essential for Japanese syntax and indicate grammatical density."
}

def get_jread_level(score):
    if score is None: return "N/A"
    if 0.5 <= score < 1.5: return "Upper-advanced"
    elif 1.5 <= score < 2.5: return "Lower-advanced"
    elif 2.5 <= score < 3.5: return "Upper-intermediate"
    elif 3.5 <= score < 4.5: return "Lower-intermediate"
    elif 4.5 <= score < 5.5: return "Upper-elementary"
    elif 5.5 <= score < 6.5: return "Lower-elementary"
    else: return "Beginner"

def get_jgri_interpretation(val):
    if val < -1.0: return "Very easy / Conversational"
    elif -1.0 <= val < 0: return "Relatively easy"
    elif 0 <= val < 1.0: return "Medium complexity"
    else: return "High complexity"

def get_ttr_interpretation(val):
    if val < 0.45: return "Low diversity (Repetitive)"
    elif 0.45 <= val < 0.65: return "Moderate diversity"
    else: return "High diversity (Varied)"

def get_mtld_interpretation(val):
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
    
    # POS & Scripts
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
    
    # Formula Calculation (11.724 Constant)
    total_tokens = len(valid_nodes)
    wps = total_tokens / num_sentences
    pk = (scripts["K"] / total_tokens * 100) if total_tokens > 0 else 0
    pw = (scripts["H"] / total_tokens * 100) if total_tokens > 0 else 0
    pv = (len(verbs) / total_tokens * 100) if total_tokens > 0 else 0
    pp = (len(particles) / total_tokens * 100) if total_tokens > 0 else 0
    
    jread_score = (11.724 + (wps * -0.056) + (pk * -0.126) + 
                   (pw * -0.042) + (pv * -0.145) + (pp * -0.044))

    return {
        "tokens": [n.surface for n in valid_nodes if n.feature.pos1 != "補助記号"],
        "jread": {
            "Score": round(jread_score, 3), "WPS": round(wps, 2), 
            "K_Full": round(pk, 2), "W_Full": round(pw, 2), 
            "V_Full": round(pv, 2), "P_Full": round(pp, 2)
        },
        "jgri_raw": {"MMS": total_tokens / num_sentences, "LD": len(content_words) / total_tokens, "VPS": len(verbs) / num_sentences, "MPN": len(adverbs) / len(nouns) if len(nouns) > 0 else 0},
        "scripts": {k: round((v/sum(scripts.values()))*100, 1) for k, v in scripts.items()}
    }

# ===============================================
# --- 3. UI LAYOUT & DATA FETCHING ---
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

PRELOADED_CORPORA = {
    "DICO-JALF ALL": "https://raw.githubusercontent.com/prihantoro-corpus/vocabulary-profiler/main/DICO-JALF%20all.xlsx",
    "DICO-JALF 30 Files Only": "https://raw.githubusercontent.com/prihantoro-corpus/vocabulary-profiler/main/DICO-JALF%2030%20files%20only.xlsx",
}

source = st.sidebar.selectbox("Data Source", ["Upload Files", "DICO-JALF 30", "DICO-JALF ALL"])
corpus = []
if source == "Upload Files":
    up = st.sidebar.file_uploader("Upload .txt or .xlsx", accept_multiple_files=True)
    if up:
        for f in up: corpus.append({"name": f.name, "text": f.read().decode('utf-8')})
else:
    try:
        url_key = "DICO-JALF ALL" if "ALL" in source else "DICO-JALF 30 Files Only"
        resp = requests.get(PRELOADED_CORPORA[url_key])
        df_pre = pd.read_excel(io.BytesIO(resp.content), header=None)
        corpus = [{"name": str(r[0]), "text": str(r[1])} for _, r in df_pre.iterrows()]
    except: st.error("Failed to load pre-loaded corpus.")

if corpus:
    tagger = Tagger()
    results, global_tokens = [], []

    for item in corpus:
        data = analyze_text(item['text'], tagger)
        global_tokens.extend(data["tokens"])
        
        lr = LexicalRichness(" ".join(data["tokens"])) if data["tokens"] else None
        ttr_val = round(len(set(data['tokens']))/len(data['tokens']), 3) if data['tokens'] else 0
        mtld_val = round(lr.mtld(), 2) if lr and len(data["tokens"]) > 10 else 0
        
        results.append({
            "File": item['name'],
            "Tokens": len(data["tokens"]),
            "TTR": ttr_val,
            "TTR Interpretation": get_ttr_interpretation(ttr_val),
            "MTLD": mtld_val,
            "MTLD Interpretation": get_mtld_interpretation(mtld_val),
            "Readability": data["jread"]["Score"],
            "J-Level": get_jread_level(data["jread"]["Score"]),
            "WPS": data["jread"]["WPS"],
            "Percentage Kango": data["jread"]["K_Full"],
            "Percentage Wago": data["jread"]["W_Full"],
            "Percentage Verbs": data["jread"]["V_Full"],
            "Percentage Particles": data["jread"]["P_Full"],
            **data["jgri_raw"], **data["scripts"]
        })

    df = pd.DataFrame(results)
    for col in ["MMS", "LD", "VPS", "MPN"]:
        df[f"z_{col}"] = zscore(df[col]) if df[col].std() != 0 else 0
    df["JGRI"] = df[[f"z_{col}" for col in ["MMS", "LD", "VPS", "MPN"]]].mean(axis=1).round(3)
    df["Complexity"] = df["JGRI"].apply(get_jgri_interpretation)

    # --- MAIN TABLE WITH TOOLTIPS ---
    st.header("Analysis Matrix")
    
    column_config = {
        "Tokens": st.column_config.NumberColumn("Tokens", help=TOOLTIPS["Tokens"]),
        "TTR": st.column_config.NumberColumn("TTR", help=TOOLTIPS["TTR"]),
        "MTLD": st.column_config.NumberColumn("MTLD", help=TOOLTIPS["MTLD"]),
        "Readability": st.column_config.NumberColumn("Readability", help=TOOLTIPS["Readability"]),
        "JGRI": st.column_config.NumberColumn("JGRI", help=TOOLTIPS["JGRI"]),
        "WPS": st.column_config.NumberColumn("WPS", help=TOOLTIPS["WPS"]),
        "Percentage Kango": st.column_config.NumberColumn("Percentage Kango", help=TOOLTIPS["Kango"]),
        "Percentage Wago": st.column_config.NumberColumn("Percentage Wago", help=TOOLTIPS["Wago"]),
        "Percentage Verbs": st.column_config.NumberColumn("Percentage Verbs", help=TOOLTIPS["Verbs"]),
        "Percentage Particles": st.column_config.NumberColumn("Percentage Particles", help=TOOLTIPS["Particles"]),
    }

    display_cols = ["File", "Tokens", "TTR", "TTR Interpretation", "MTLD", "MTLD Interpretation", "Readability", "J-Level", "JGRI", "Complexity", "WPS", "Percentage Kango", "Percentage Wago", "Percentage Verbs", "Percentage Particles", "K", "H", "T", "O"]
    st.dataframe(df[display_cols].rename(columns={"K": "Kanji%", "H": "Hira%", "T": "Kata%", "O": "Other%"}), 
                 column_config=column_config, use_container_width=True)

    # --- N-GRAM TABLES ---
    st.divider()
    st.header("N-Gram Frequencies")
    
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
            st.dataframe(pd.DataFrame(Counter(grams).most_common(10), columns=['Sequence', 'Freq']), hide_index=True)

else:
    st.info("Awaiting data input...")
