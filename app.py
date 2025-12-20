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
# --- 1. CONFIGURATION & TOOLTIPS ---
# ===============================================

GITHUB_BASE = "https://raw.githubusercontent.com/prihantoro-corpus/vocabulary-profiler/main/"
JLPT_FILES = {"N1": "unknown_source_N1.csv", "N2": "unknown_source_N2.csv", "N3": "unknown_source_N3.csv", "N4": "unknown_source_N4.csv", "N5": "unknown_source_N5.csv"}

TOOLTIPS = {
    "Tokens": "Corpus size: Total number of all morphemes/words detected by the tokenizer.",
    "TTR": "Type-Token Ratio. Thresholds:\n- < 0.45: Repetitive\n- 0.45-0.65: Moderate\n- > 0.65: Varied/Diverse",
    "MTLD": "Lexical Diversity (Length-independent). Thresholds:\n- < 40: Basic\n- 40-80: Intermediate\n- > 80: Advanced",
    "Readability": "JReadability (Hasebe & Lee 2015). Thresholds:\n- 0.5-1.5: Upper-advanced\n- 2.5-3.5: Upper-intermediate\n- 4.5-5.5: Upper-elementary",
    "JGRI": "Japanese Grammar Readability Index (Relative Complexity):\n- < -1.0: Very easy / Conversational\n- 0 to +1.0: Medium complexity\n- > +1.0: High complexity"
}

# ===============================================
# --- 2. LINGUISTIC ENGINE ---
# ===============================================

@st.cache_data
def load_jlpt_wordlists():
    wordlists = {}
    for lvl, f in JLPT_FILES.items():
        try:
            df = pd.read_csv(GITHUB_BASE + f)
            wordlists[lvl] = set(df.iloc[:, 0].astype(str).tolist())
        except: wordlists[lvl] = set()
    return wordlists

def get_jread_level(score):
    if 0.5 <= score < 1.5: return "Upper-advanced"
    elif 1.5 <= score < 2.5: return "Lower-advanced"
    elif 2.5 <= score < 3.5: return "Upper-intermediate"
    elif 3.5 <= score < 4.5: return "Lower-intermediate"
    elif 4.5 <= score < 5.5: return "Upper-elementary"
    elif 5.5 <= score < 6.5: return "Lower-elementary"
    else: return "Other"

def analyze_text(text, tagger, jlpt_lists):
    nodes = tagger(text)
    # Filter nodes for analysis (excluding specific punctuation)
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
    
    # 2.1 Script Detection
    scripts = {"K": 0, "H": 0, "T": 0, "NA": 0}
    for n in valid_nodes:
        if re.search(r'[\u4e00-\u9faf]', n['surface']): scripts["K"] += 1
        elif re.search(r'[\u3040-\u309f]', n['surface']): scripts["H"] += 1
        elif re.search(r'[\u30a0-\u30ff]', n['surface']): scripts["T"] += 1
        else: scripts["NA"] += 1

    # 2.2 POS & JLPT Mapping
    pos_map = {"Noun (名詞)": "名詞", "Verb (動詞)": "動詞", "Particle (助詞)": "助詞", "Adverb (副詞)": "副詞", "Adjective (形容詞)": "形容詞"}
    pos_counts = {k: sum(1 for n in valid_nodes if n['pos'] == v) for k, v in pos_map.items()}
    
    jlpt_counts = {lvl: 0 for lvl in ["N1", "N2", "N3", "N4", "N5", "NA"]}
    for n in valid_nodes:
        found = False
        for lvl in ["N1", "N2", "N3", "N4", "N5"]:
            if n['lemma'] in jlpt_lists[lvl]:
                jlpt_counts[lvl] += 1
                found = True
                break
        if not found: jlpt_counts["NA"] += 1

    # 2.3 Formula components
    wps = total_tokens / num_sentences
    pk, ph, pv, pp = [(x/total_tokens*100) if total_tokens > 0 else 0 for x in [scripts["K"], scripts["H"], pos_counts["Verb (動詞)"], pos_counts["Particle (助詞)"]]]
    jread_score = (11.724 + (wps * -0.056) + (pk * -0.126) + (ph * -0.042) + (pv * -0.145) + (pp * -0.044))

    return {
        "tokens": valid_nodes,
        "raw_stats": {"Tokens": total_tokens, "WPS": round(wps, 2), "Readability": round(jread_score, 3), "K_Raw": scripts["K"], "H_Raw": scripts["H"], "T_Raw": scripts["T"], "O_Raw": scripts["NA"], "V_Raw": pos_counts["Verb (動詞)"], "P_Raw": pos_counts["Particle (助詞)"]},
        "pos_dist": pos_counts,
        "jlpt": jlpt_counts,
        "jgri_base": {"MMS": total_tokens/num_sentences, "LD": sum(pos_counts.values())/total_tokens if total_tokens > 0 else 0, "VPS": pos_counts["Verb (動詞)"]/num_sentences, "MPN": pos_counts["Adverb (副詞)"]/pos_counts["Noun (名詞)"] if pos_counts["Noun (名詞)"] > 0 else 0}
    }

# ===============================================
# --- 3. UI & SIDEBAR ---
# ===============================================

st.set_page_config(layout="wide", page_title="Japanese Lexical Profiler")

pwd = st.sidebar.text_input("If you are a developer, tester, or reviewer, enter password", type="password")
if pwd != "290683":
    st.info("Please enter the password in the sidebar to proceed.")
    st.stop()

tagger, jlpt_lists = Tagger(), load_jlpt_wordlists()

# Sidebar: Advanced N-Gram Search
st.sidebar.header("Advanced N-Gram Pattern")
n_val = st.sidebar.number_input("N-Gram Size", 1, 5, 2)
pattern = []
for i in range(n_val):
    pattern.append(st.sidebar.text_input(f"Position {i+1} (word, POS:名詞, or *)", value="*", key=f"p_{i}"))

st.title("📖 Japanese Text Vocabulary Profiler")

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
        total = data["raw_stats"]["Tokens"]
        lr = LexicalRichness(" ".join([t['surface'] for t in data["tokens"]])) if total > 10 else None
        
        # General Row
        row = {
            "File": item['name'], "Tokens": total, 
            "TTR": round(len(set([t['lemma'] for t in data["tokens"]]))/total, 3) if total > 0 else 0,
            "MTLD": round(lr.mtld(), 2) if lr else 0,
            "Readability": data["raw_stats"]["Readability"], "J-Level": get_jread_level(data["raw_stats"]["Readability"]),
            "WPS": data["raw_stats"]["WPS"], "Percentage Kango": round(data["raw_stats"]["K_Raw"]/total*100, 2) if total > 0 else 0,
            "Percentage Wago": round(data["raw_stats"]["H_Raw"]/total*100, 2) if total > 0 else 0,
            "Percentage Verbs": round(data["raw_stats"]["V_Raw"]/total*100, 2) if total > 0 else 0,
            "Percentage Particles": round(data["raw_stats"]["P_Raw"]/total*100, 2) if total > 0 else 0,
            "K%": round(data["raw_stats"]["K_Raw"]/total*100, 1) if total > 0 else 0, "H%": round(data["raw_stats"]["H_Raw"]/total*100, 1) if total > 0 else 0,
            "T%": round(data["raw_stats"]["T_Raw"]/total*100, 1) if total > 0 else 0, "Other%": round(data["raw_stats"]["O_Raw"]/total*100, 1) if total > 0 else 0,
            **data["jgri_base"]
        }
        for lvl in ["N1", "N2", "N3", "N4", "N5", "NA"]:
            row[lvl], row[f"{lvl}%"] = data["jlpt"][lvl], round((data["jlpt"][lvl]/total*100), 1) if total > 0 else 0
        results.append(row)

        # POS Row
        p_row = {"File": item['name'], "Tokens": total}
        for p_lab, count in data["pos_dist"].items():
            p_row[f"{p_lab} (Raw)"], p_row[f"{p_lab} (%)"] = count, round((count/total*100), 2) if total > 0 else 0
        pos_results.append(p_row)

    df = pd.DataFrame(results)
    for c in ["MMS", "LD", "VPS", "MPN"]:
        df[f"z_{c}"] = zscore(df[c]) if df[c].std() != 0 else 0
    df["JGRI"] = df[[f"z_{c}" for c in ["MMS", "LD", "VPS", "MPN"]]].mean(axis=1).round(3)
    df["Complexity"] = df["JGRI"].apply(lambda v: "High" if v > 1 else ("Medium" if v >= 0 else "Easy"))

    # --- OUTPUT TABS ---
    tab1, tab2 = st.tabs(["📊 General Analysis", "📝 POS Distribution"])
    
    with tab1:
        st.header("Analysis Matrix")
        cfg = {k: st.column_config.NumberColumn(k, help=v) for k, v in TOOLTIPS.items()}
        disp = ["File", "Tokens", "TTR", "MTLD", "Readability", "J-Level", "JGRI", "Complexity", "WPS", "Percentage Kango", "Percentage Wago", "Percentage Verbs", "Percentage Particles", "K%", "H%", "T%", "Other%"] + [f"{l}{s}" for l in ["N1","N2","N3","N4","N5","NA"] for s in ["", "%"]]
        st.dataframe(df[disp], column_config=cfg, use_container_width=True)

    with tab2:
        st.header("POS Distribution (English & 日本語)")
        st.dataframe(pd.DataFrame(pos_results), use_container_width=True)

    # --- ADVANCED N-GRAM RESULTS ---
    st.divider()
    st.header(f"Advanced {n_val}-Gram Search")
    matches = []
    for j in range(len(global_tokens) - n_val + 1):
        window = global_tokens[j : j + n_val]
        match = True
        for idx, p in enumerate(pattern):
            p = p.strip()
            if p == "*" or p == "": continue
            elif p.startswith("POS:"):
                if window[idx]['pos'] != p.split(":")[1]: match = False; break
            elif window[idx]['surface'] != p and window[idx]['lemma'] != p:
                match = False; break
        if match: matches.append(" ".join([t['surface'] for t in window]))
    
    if matches:
        st.write(f"Pattern matched `{len(matches)}` times.")
        df_g = pd.DataFrame(Counter(matches).most_common(10), columns=['Sequence', 'Raw Freq'])
        df_g['Freq (PMW)'] = df_g['Raw Freq'].apply(lambda x: round((x / len(global_tokens)) * 1_000_000, 2))
        st.dataframe(df_g, use_container_width=True)
    else: st.warning("No sequences matched the specified pattern.")

else: st.info("Awaiting data input...")
