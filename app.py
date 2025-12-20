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
    "MTLD": "Lexical Diversity (Length-independent). Thresholds: < 40: Basic | 40-80: Intermediate | > 80: Advanced.",
    "Readability": "JReadability: 0.5-1.5: Upper-adv | 1.5-2.5: Lower-adv | 2.5-3.5: Upper-int | 3.5-4.5: Lower-int | 4.5-5.5: Upper-elem | 5.5-6.5: Lower-elem.",
    "JGRI": "Relative Complexity: < -1.0: Very easy | 0 to +1.0: Medium | > +1.0: High complexity.",
    "Scripts": "Distribution: Kanji (K), Hiragana (H), Katakana (T), and NA (Other symbols).",
    "JLPT": "Distribution based on N1-N5 lists. NA = Not found in standard JLPT lists."
}

def get_jread_level(score):
    if score is None: return "N/A"
    if 0.5 <= score < 1.5: return "Upper-advanced"
    elif 1.5 <= score < 2.5: return "Lower-advanced"
    elif 2.5 <= score < 3.5: return "Upper-intermediate"
    elif 3.5 <= score < 4.5: return "Lower-intermediate"
    elif 4.5 <= score < 5.5: return "Upper-elementary"
    elif 5.5 <= score < 6.5: return "Lower-elementary"
    else: return "Beginner/Other"

def get_jgri_interpretation(val):
    if val < -1.0: return "Very easy / Conversational"
    elif -1.0 <= val < 0: return "Relatively easy"
    elif 0 <= val < 1.0: return "Medium complexity"
    else: return "High complexity"

# ===============================================
# --- 2. DATA FETCHING ---
# ===============================================

GITHUB_BASE = "https://raw.githubusercontent.com/prihantoro-corpus/vocabulary-profiler/main/"
JLPT_FILES = {"N1": "unknown_source_N1.csv", "N2": "unknown_source_N2.csv", "N3": "unknown_source_N3.csv", "N4": "unknown_source_N4.csv", "N5": "unknown_source_N5.csv"}

@st.cache_data
def load_jlpt():
    wordlists = {}
    for lvl, f in JLPT_FILES.items():
        try:
            df = pd.read_csv(GITHUB_BASE + f)
            wordlists[lvl] = set(df.iloc[:, 0].astype(str).tolist())
        except: wordlists[lvl] = set()
    return wordlists

# ===============================================
# --- 3. LINGUISTIC ENGINE ---
# ===============================================

def analyze_text(text, tagger, jlpt_lists):
    nodes = tagger(text)
    # Filter nodes to exclude Japanese punctuation (補助記号)
    valid_nodes = [n for n in nodes if n.surface and n.feature.pos1 != "補助記号"]
    sentences = [s for s in re.split(r'[。！？\n]', text.strip()) if s.strip()]
    num_sentences = len(sentences) if sentences else 1
    total_tokens = len(valid_nodes)
    
    # POS Mapping (Bilingual)
    pos_map = {
        "Noun (名詞)": "名詞", "Verb (動詞)": "動詞", "Particle (助詞)": "助詞", 
        "Adverb (副詞)": "副詞", "Adjective (形容詞)": "形容詞", 
        "Auxiliary (助動詞)": "助動詞", "Conjunction (接続詞)": "接続詞", "Pronoun (代名詞)": "代名詞"
    }
    pos_counts = {k: sum(1 for n in valid_nodes if n.feature.pos1 == v) for k, v in pos_map.items()}
    content_count = sum(1 for n in valid_nodes if n.feature.pos1 in ["名詞", "動詞", "形容詞", "副詞"])
    
    # Scripts
    scripts = {"K": 0, "H": 0, "T": 0, "NA": 0}
    for n in valid_nodes:
        if re.search(r'[\u4e00-\u9faf]', n.surface): scripts["K"] += 1
        elif re.search(r'[\u3040-\u309f]', n.surface): scripts["H"] += 1
        elif re.search(r'[\u30a0-\u30ff]', n.surface): scripts["T"] += 1
        else: scripts["NA"] += 1

    # JLPT Mapping
    jlpt_counts = {lvl: 0 for lvl in ["N1", "N2", "N3", "N4", "N5", "NA"]}
    for n in valid_nodes:
        lemma = n.feature.orth if hasattr(n.feature, 'orth') else n.surface
        found = False
        for lvl in ["N1", "N2", "N3", "N4", "N5"]:
            if lemma in jlpt_lists[lvl]:
                jlpt_counts[lvl] += 1
                found = True
                break
        if not found: jlpt_counts["NA"] += 1

    # JReadability (11.724 + WPS*-0.056 + K%*-0.126 + W%*-0.042 + V%*-0.145 + P%*-0.044)
    wps = total_tokens / num_sentences
    pk, ph, pv, pp = [(x / total_tokens * 100) if total_tokens > 0 else 0 
                      for x in [scripts["K"], scripts["H"], pos_counts["Verb (動詞)"], pos_counts["Particle (助詞)"]]]
    jread_score = (11.724 + (wps * -0.056) + (pk * -0.126) + (ph * -0.042) + (pv * -0.145) + (pp * -0.044))

    return {
        "tokens_list": [n.surface for n in valid_nodes],
        "stats": {
            "Tokens": total_tokens, "WPS": round(wps, 2), "Readability": round(jread_score, 3),
            "K_Raw": scripts["K"], "H_Raw": scripts["H"], "T_Raw": scripts["T"], "Other_Raw": scripts["NA"],
            "V_Raw": pos_counts["Verb (動詞)"], "P_Raw": pos_counts["Particle (助詞)"]
        },
        "pos": pos_counts,
        "jgri_raw": {
            "MMS": total_tokens / num_sentences, 
            "LD": content_count/total_tokens if total_tokens > 0 else 0, 
            "VPS": pos_counts["Verb (動詞)"]/num_sentences, 
            "MPN": pos_counts["Adverb (副詞)"]/pos_counts["Noun (名詞)"] if pos_counts["Noun (名詞)"] > 0 else 0
        },
        "jlpt": jlpt_counts
    }

# ===============================================
# --- 4. UI & MAIN LOGIC ---
# ===============================================

st.set_page_config(layout="wide", page_title="Japanese Lexical Profiler")

# Password Authentication
pwd = st.sidebar.text_input("If you are a developer, tester, or reviewer, enter password", type="password")
if pwd != "290683":
    st.info("Please enter the password in the sidebar to proceed.")
    st.stop()

tagger, jlpt_lists = Tagger(), load_jlpt()
st.title("📖 Japanese Text Vocabulary Profiler")

# Sidebar
st.sidebar.header("N-Gram Config")
n_exact = st.sidebar.number_input("Exact N value", 1, 5, 1)
n_range = st.sidebar.text_input("Range N values (e.g., 2, 4)", "")

source = st.sidebar.selectbox("Data Source", ["Upload Files", "DICO-JALF 30", "DICO-JALF ALL"])
corpus = []

if source == "Upload Files":
    up = st.sidebar.file_uploader("Upload files", accept_multiple_files=True)
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
        global_tokens.extend(data["tokens_list"])
        total = data["stats"]["Tokens"]
        lr = LexicalRichness(" ".join(data["tokens_list"])) if total > 0 else None
        
        # 1. Build General Matrix Row
        row = {
            "File": item['name'], "Tokens": total,
            "TTR": round(len(set(data['tokens_list']))/total, 3) if total > 0 else 0,
            "MTLD": round(lr.mtld(), 2) if lr and total > 10 else 0,
            "Readability": data["stats"]["Readability"], "J-Level": get_jread_level(data["stats"]["Readability"]),
            "WPS": data["stats"]["WPS"], 
            "Kango Count": data["stats"]["K_Raw"], "Percentage Kango": round(data["stats"]["K_Raw"]/total*100, 2) if total > 0 else 0,
            "Wago Count": data["stats"]["H_Raw"], "Percentage Wago": round(data["stats"]["H_Raw"]/total*100, 2) if total > 0 else 0,
            "Verbs Count": data["stats"]["V_Raw"], "Percentage Verbs": round(data["stats"]["V_Raw"]/total*100, 2) if total > 0 else 0,
            "Particles Count": data["stats"]["P_Raw"], "Percentage Particles": round(data["stats"]["P_Raw"]/total*100, 2) if total > 0 else 0,
            "Kanji Count": data["stats"]["K_Raw"], "Kanji%": round(data["stats"]["K_Raw"]/total*100, 1) if total > 0 else 0,
            "Hira Count": data["stats"]["H_Raw"], "Hira%": round(data["stats"]["H_Raw"]/total*100, 1) if total > 0 else 0,
            "Kata Count": data["stats"]["T_Raw"], "Kata%": round(data["stats"]["T_Raw"]/total*100, 1) if total > 0 else 0,
            "Other Count": data["stats"]["Other_Raw"], "Other%": round(data["stats"]["Other_Raw"]/total*100, 1) if total > 0 else 0,
            **data["jgri_raw"]
        }
        for lvl in ["N1", "N2", "N3", "N4", "N5", "NA"]:
            row[lvl], row[f"{lvl}%"] = data["jlpt"][lvl], round((data["jlpt"][lvl]/total*100), 1) if total > 0 else 0
        results.append(row)

        # 2. Build POS Matrix Row
        p_row = {"File": item['name'], "Tokens": total}
        for pos_label, count in data["pos"].items():
            p_row[f"{pos_label} (Raw)"] = count
            p_row[f"{pos_label} (%)"] = round((count/total*100), 2) if total > 0 else 0
        pos_results.append(p_row)

    # Calculate JGRI Relative Normalization
    df = pd.DataFrame(results)
    for col in ["MMS", "LD", "VPS", "MPN"]:
        df[f"z_{col}"] = zscore(df[col]) if df[col].std() != 0 else 0
    df["JGRI"] = df[[f"z_{col}" for col in ["MMS", "LD", "VPS", "MPN"]]].mean(axis=1).round(3)
    df["Complexity"] = df["JGRI"].apply(get_jgri_interpretation)

    # --- TABS FOR MATRICES ---
    tab_gen, tab_pos = st.tabs(["📊 General Analysis", "📝 POS Distribution"])

    with tab_gen:
        st.header("General Lexical Profile")
        col_cfg = {
            "Tokens": st.column_config.NumberColumn("Tokens", help=TOOLTIPS["Tokens"]),
            "TTR": st.column_config.NumberColumn("TTR", help=TOOLTIPS["TTR"]),
            "MTLD": st.column_config.NumberColumn("MTLD", help=TOOLTIPS["MTLD"]),
            "Readability": st.column_config.NumberColumn("Readability", help=TOOLTIPS["Readability"]),
            "JGRI": st.column_config.NumberColumn("JGRI", help=TOOLTIPS["JGRI"])
        }
        disp_cols = ["File", "Tokens", "TTR", "MTLD", "Readability", "J-Level", "JGRI", "Complexity", "WPS", "Kango Count", "Percentage Kango", "Wago Count", "Percentage Wago", "Verbs Count", "Percentage Verbs", "Particles Count", "Percentage Particles", "Kanji Count", "Kanji%", "Hira Count", "Hira%", "Kata Count", "Kata%", "Other Count", "Other%"] + \
                    ["N1", "N1%", "N2", "N2%", "N3", "N3%", "N4", "N4%", "N5", "N5%", "NA", "NA%"]
        st.dataframe(df[disp_cols], column_config=col_cfg, use_container_width=True)

    with tab_pos:
        st.header("Part-of-Speech Distribution (English & 日本語)")
        st.dataframe(pd.DataFrame(pos_results), use_container_width=True)

    # --- N-GRAM TABLES ---
    st.divider()
    st.header("N-Gram Frequencies (Global PMW)")
    total_corpus_tokens = len(global_tokens)
    req_ns = [n_exact]
    if n_range:
        try:
            parts = [int(x.strip()) for x in n_range.split(",") if x.strip()]
            if len(parts) >= 2: req_ns.extend(list(range(min(parts), max(parts) + 1)))
        except: pass
    
    unique_ns = sorted(list(set([n for n in req_ns if 1 <= n <= 5])))
    st_cols = st.columns(len(unique_ns))
    for i, n in enumerate(unique_ns):
        with st_cols[i]:
            st.subheader(f"{n}-Gram")
            grams = [" ".join(global_tokens[j:j+n]) for j in range(len(global_tokens)-n+1)]
            df_g = pd.DataFrame(Counter(grams).most_common(10), columns=['Sequence', 'Raw Freq'])
            df_g['PMW'] = df_g['Raw Freq'].apply(lambda x: round((x / total_corpus_tokens) * 1_000_000, 2))
            st.dataframe(df_g.rename(columns={'PMW': 'Freq (per Million)'}), hide_index=True)
else:
    st.info("Awaiting data input...")
