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
    "Tokens": "Corpus size: Total number of all morphemes/words detected by the tokenizer.",
    "TTR": "Type-Token Ratio. Thresholds: < 0.45: Repetitive | 0.45-0.65: Moderate | > 0.65: Varied.",
    "MTLD": "Lexical Diversity. Thresholds: < 40: Basic | 40-80: Intermediate | > 80: Advanced.",
    "Readability": "JReadability (Hasebe & Lee 2015). Constant: 11.724. Lower scores = Advanced.",
    "JGRI": "Relative Complexity: < -1.0: Easy | 0 to +1.0: Medium | > +1.0: High complexity.",
    "JLPT": "Distribution based on N1-N5 wordlists from GitHub repository.",
    "POS": "Part of Speech Distribution: English and Japanese labels included."
}

# POS Options for the N-Gram Selectbox
POS_OPTIONS = ["Any (*)", "名詞 (Noun)", "動詞 (Verb)", "助詞 (Particle)", "副詞 (Adverb)", "形容詞 (Adjective)", "助動詞 (Auxiliary)", "接続詞 (Conjunction)", "代名詞 (Pronoun)", "連体詞 (Determiner)", "感動詞 (Interjection)"]

# ===============================================
# --- 2. LINGUISTIC ENGINE ---
# ===============================================

@st.cache_data
def load_jlpt_wordlists():
    base = "https://raw.githubusercontent.com/prihantoro-corpus/vocabulary-profiler/main/"
    files = {"N1": "unknown_source_N1.csv", "N2": "unknown_source_N2.csv", "N3": "unknown_source_N3.csv", "N4": "unknown_source_N4.csv", "N5": "unknown_source_N5.csv"}
    lists = {}
    for lvl, f in files.items():
        try:
            df = pd.read_csv(base + f)
            lists[lvl] = set(df.iloc[:, 0].astype(str).tolist())
        except: lists[lvl] = set()
    return lists

def analyze_text(text, tagger, jlpt_lists):
    nodes = tagger(text)
    valid_nodes = []
    for n in nodes:
        if n.surface and n.feature.pos1 != "補助記号":
            valid_nodes.append({
                "surface": n.surface,
                "lemma": n.feature.orth if hasattr(n.feature, 'orth') else n.surface,
                "pos": n.feature.pos1
            })
    
    sentences = [s for s in re.split(r'[。！？\n]', text.strip()) if s.strip()]
    num_sentences = len(sentences) if sentences else 1
    total_tokens = len(valid_nodes)
    
    scripts = {"K": 0, "H": 0, "T": 0, "NA": 0}
    for n in valid_nodes:
        if re.search(r'[\u4e00-\u9faf]', n['surface']): scripts["K"] += 1
        elif re.search(r'[\u3040-\u309f]', n['surface']): scripts["H"] += 1
        elif re.search(r'[\u30a0-\u30ff]', n['surface']): scripts["T"] += 1
        else: scripts["NA"] += 1

    pos_names = {"Noun (名詞)": "名詞", "Verb (動詞)": "動詞", "Particle (助詞)": "助詞", "Adverb (副詞)": "副詞", "Adjective (形容詞)": "形容詞", "Auxiliary (助動詞)": "助動詞", "Conjunction (接続詞)": "接続詞"}
    pos_counts = {k: sum(1 for n in valid_nodes if n['pos'] == v) for k, v in pos_names.items()}
    
    jlpt_counts = {lvl: 0 for lvl in ["N1", "N2", "N3", "N4", "N5", "NA"]}
    for n in valid_nodes:
        found = False
        for lvl in ["N1", "N2", "N3", "N4", "N5"]:
            if n['lemma'] in jlpt_lists[lvl]:
                jlpt_counts[lvl] += 1
                found = True
                break
        if not found: jlpt_counts["NA"] += 1

    # JReadability Formula
    wps = total_tokens / num_sentences
    pk, ph, pv, pp = [(x/total_tokens*100) if total_tokens > 0 else 0 for x in [scripts["K"], scripts["H"], pos_counts["Verb (動詞)"], pos_counts["Particle (助詞)"]]]
    jread_score = (11.724 + (wps * -0.056) + (pk * -0.126) + (ph * -0.042) + (pv * -0.145) + (pp * -0.044))

    return {
        "tokens": valid_nodes,
        "stats": {"Tokens": total_tokens, "WPS": round(wps, 2), "Readability": round(jread_score, 3), "K_Raw": scripts["K"], "H_Raw": scripts["H"], "T_Raw": scripts["T"], "O_Raw": scripts["NA"], "V_Raw": pos_counts["Verb (動詞)"], "P_Raw": pos_counts["Particle (助詞)"]},
        "pos": pos_counts, "jlpt": jlpt_counts,
        "jgri": {"MMS": total_tokens/num_sentences, "LD": sum(pos_counts.values())/total_tokens if total_tokens > 0 else 0, "VPS": pos_counts["Verb (動詞)"]/num_sentences, "MPN": pos_counts["Adverb (副詞)"]/pos_counts["Noun (名詞)"] if pos_counts["Noun (名詞)"] > 0 else 0}
    }

# ===============================================
# --- 3. UI LAYOUT ---
# ===============================================

st.set_page_config(layout="wide", page_title="Japanese Lexical Profiler")

pwd = st.sidebar.text_input("If you are a developer, tester, or reviewer, enter password", type="password")
if pwd != "290683":
    st.info("If you are a developer, tester, or reviewer, enter the password in the sidebar.")
    st.stop()

tagger, jlpt_lists = Tagger(), load_jlpt_wordlists()
st.title("📖 Japanese Text Vocabulary Profiler")

# Sidebar: Advanced N-Gram Search with Dual Boxes
st.sidebar.header("Advanced N-Gram Pattern")
n_val = st.sidebar.number_input("N-Gram Size", 1, 5, 2)

pattern_words = []
pattern_pos = []

for i in range(n_val):
    st.sidebar.write(f"**Position {i+1}**")
    col1, col2 = st.sidebar.columns(2)
    with col1:
        w = st.text_input("Word/*", value="*", key=f"w_{i}")
        pattern_words.append(w)
    with col2:
        p = st.selectbox("POS Tag", options=POS_OPTIONS, key=f"p_{i}")
        pattern_pos.append(p.split(" ")[0]) # Extract '名詞' from '名詞 (Noun)'

source = st.sidebar.selectbox("Data Source", ["Upload Files", "DICO-JALF 30", "DICO-JALF ALL"])
corpus = []
if source == "Upload Files":
    up = st.sidebar.file_uploader("Upload .txt or .xlsx", accept_multiple_files=True)
    if up:
        for f in up: corpus.append({"name": f.name, "text": f.read().decode('utf-8')})
else:
    url = "https://raw.githubusercontent.com/prihantoro-corpus/vocabulary-profiler/main/DICO-JALF%20all.xlsx" if "ALL" in source else "https://raw.githubusercontent.com/prihantoro-corpus/vocabulary-profiler/main/DICO-JALF%2030%20files%20only.xlsx"
    df_pre = pd.read_excel(io.BytesIO(requests.get(url).content), header=None)
    corpus = [{"name": str(r[0]), "text": str(r[1])} for _, r in df_pre.iterrows()]

if corpus:
    results, pos_results, global_tokens = [], [], []
    for item in corpus:
        data = analyze_text(item['text'], tagger, jlpt_lists)
        global_tokens.extend(data["tokens"])
        total = data["stats"]["Tokens"]
        lr = LexicalRichness(" ".join([t['surface'] for t in data["tokens"]])) if total > 10 else None
        
        row = {
            "File": item['name'], "Tokens": total, "TTR": round(len(set([t['lemma'] for t in data["tokens"]]))/total, 3) if total > 0 else 0,
            "MTLD": round(lr.mtld(), 2) if lr else 0, "Readability": data["stats"]["Readability"], "WPS": data["stats"]["WPS"],
            "Percentage Kango": round(data["stats"]["K_Raw"]/total*100, 2) if total > 0 else 0,
            "Percentage Wago": round(data["stats"]["H_Raw"]/total*100, 2) if total > 0 else 0,
            "Percentage Verbs": round(data["stats"]["V_Raw"]/total*100, 2) if total > 0 else 0,
            "Percentage Particles": round(data["stats"]["P_Raw"]/total*100, 2) if total > 0 else 0,
            "K%": round(data["stats"]["K_Raw"]/total*100, 1) if total > 0 else 0, "H%": round(data["stats"]["H_Raw"]/total*100, 1) if total > 0 else 0,
            "T%": round(data["stats"]["T_Raw"]/total*100, 1) if total > 0 else 0, "Other%": round(data["stats"]["O_Raw"]/total*100, 1) if total > 0 else 0,
            **data["jgri"]
        }
        for lvl in ["N1", "N2", "N3", "N4", "N5", "NA"]:
            row[lvl], row[f"{lvl}%"] = data["jlpt"][lvl], round((data["jlpt"][lvl]/total*100), 1) if total > 0 else 0
        results.append(row)
        
        p_row = {"File": item['name'], "Tokens": total}
        for p_lab, count in data["pos"].items():
            p_row[f"{p_lab} (Raw)"], p_row[f"{p_lab} (%)"] = count, round((count/total*100), 2) if total > 0 else 0
        pos_results.append(p_row)

    df = pd.DataFrame(results)
    for c in ["MMS", "LD", "VPS", "MPN"]:
        df[f"z_{c}"] = zscore(df[c]) if df[c].std() != 0 else 0
    df["JGRI"] = df[[f"z_{c}" for c in ["MMS", "LD", "VPS", "MPN"]]].mean(axis=1).round(3)

    tab1, tab2 = st.tabs(["📊 General Analysis", "📝 POS Distribution"])
    with tab1:
        st.header("Analysis Matrix")
        cfg = {k: st.column_config.NumberColumn(k, help=v) for k, v in TOOLTIPS.items()}
        disp = ["File", "Tokens", "TTR", "MTLD", "Readability", "JGRI", "WPS", "Percentage Kango", "Percentage Wago", "Percentage Verbs", "Percentage Particles", "K%", "H%", "T%", "Other%"] + [f"{l}{s}" for l in ["N1","N2","N3","N4","N5","NA"] for s in ["", "%"]]
        st.dataframe(df[disp], column_config=cfg, use_container_width=True)
    with tab2:
        st.header("POS Distribution")
        st.dataframe(pd.DataFrame(pos_results), use_container_width=True)

    # --- ADVANCED N-GRAM PATTERN MATCHING ---
    st.divider()
    st.header(f"N-Gram Pattern Results")
    matches = []
    for j in range(len(global_tokens) - n_val + 1):
        window, match = global_tokens[j : j + n_val], True
        for idx in range(n_val):
            w_pat, p_pat = pattern_words[idx].strip(), pattern_pos[idx]
            token = window[idx]
            if w_pat != "*" and w_pat != "" and token['surface'] != w_pat and token['lemma'] != w_pat: match = False; break
            if p_pat != "Any" and token['pos'] != p_pat: match = False; break
        if match: matches.append(" ".join([t['surface'] for t in window]))
    
    if matches:
        df_g = pd.DataFrame(Counter(matches).most_common(10), columns=['Sequence', 'Raw Freq'])
        df_g['Freq (PMW)'] = df_g['Raw Freq'].apply(lambda x: round((x / len(global_tokens)) * 1_000_000, 2))
        st.dataframe(df_g, use_container_width=True)
    else: st.warning("No patterns matched.")
