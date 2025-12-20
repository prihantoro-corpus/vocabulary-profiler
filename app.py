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
    "Tokens": "Corpus size: Total number of all morphemes/words detected (including repetitions).",
    "TTR": "Type-Token Ratio (V/N). Thresholds:\n- < 0.45: Low diversity\n- 0.45 - 0.65: Moderate\n- > 0.65: High",
    "MTLD": "Measuring Textual Lexical Diversity. Thresholds:\n- < 40: Basic\n- 40 - 80: Intermediate\n- > 80: Advanced",
    "Readability": "JReadability Score. Thresholds:\n- 0.5-1.5: Upper-advanced\n- 2.5-3.5: Upper-intermediate\n- 4.5-5.5: Upper-elementary",
    "JGRI": "Japanese Grammar Readability Index (Relative Complexity):\n- < -1.0: Very easy\n- 0 to +1.0: Medium complexity\n- > +1.0: High complexity",
    "WPS": "Words Per Sentence: Average tokens per sentence.",
    "JLPT": "Distribution based on N1-N5 wordlists. NA = Not found in standard JLPT lists.",
    "Scripts": "Distribution of writing systems: Kanji (K), Hiragana (H), Katakana (T), and NA (Other characters/symbols)."
}

def get_jread_level(score):
    if score is None: return "N/A"
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
    valid_nodes = [n for n in nodes if n.surface and n.feature.pos1 != "補助記号"]
    sentences = [s for s in re.split(r'[。！？\n]', text.strip()) if s.strip()]
    num_sentences = len(sentences) if sentences else 1
    total_tokens = len(valid_nodes)
    
    # 3.1 Script and POS Extraction
    verbs = [n for n in valid_nodes if n.feature.pos1 == "動詞"]
    particles = [n for n in valid_nodes if n.feature.pos1 == "助詞"]
    nouns = [n for n in valid_nodes if n.feature.pos1 == "名詞"]
    adverbs = [n for n in valid_nodes if n.feature.pos1 == "副詞"]
    content_words = [n for n in valid_nodes if n.feature.pos1 in ["名詞", "動詞", "形容詞", "副詞"]]
    
    scripts = {"Kanji": 0, "Hiragana": 0, "Katakana": 0, "NA_Script": 0}
    for n in valid_nodes:
        if re.search(r'[\u4e00-\u9faf]', n.surface): scripts["Kanji"] += 1
        elif re.search(r'[\u3040-\u309f]', n.surface): scripts["Hiragana"] += 1
        elif re.search(r'[\u30a0-\u30ff]', n.surface): scripts["Katakana"] += 1
        else: scripts["NA_Script"] += 1

    # 3.2 JLPT Mapping
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

    # 3.3 Formula Vars
    wps = total_tokens / num_sentences
    pk = (scripts["Kanji"] / total_tokens * 100) if total_tokens > 0 else 0
    pw = (scripts["Hiragana"] / total_tokens * 100) if total_tokens > 0 else 0
    pv = (len(verbs) / total_tokens * 100) if total_tokens > 0 else 0
    pp = (len(particles) / total_tokens * 100) if total_tokens > 0 else 0
    jread_score = (11.724 + (wps * -0.056) + (pk * -0.126) + (pw * -0.042) + (pv * -0.145) + (pp * -0.044))

    return {
        "tokens_list": [n.surface for n in valid_nodes],
        "raw": {
            "Tokens": total_tokens, "WPS": round(wps, 2), "Readability": round(jread_score, 3),
            "Kango_Raw": scripts["Kanji"], "Wago_Raw": scripts["Hiragana"],
            "Verbs_Raw": len(verbs), "Particles_Raw": len(particles),
            "Kanji_Raw": scripts["Kanji"], "Hiragana_Raw": scripts["Hiragana"],
            "Katakana_Raw": scripts["Katakana"], "Other_Script_Raw": scripts["NA_Script"]
        },
        "pct": {
            "Percentage Kango": round(pk, 2), "Percentage Wago": round(pw, 2),
            "Percentage Verbs": round(pv, 2), "Percentage Particles": round(pp, 2),
            "Kanji%": round(pk, 1), "Hiragana%": round(pw, 1),
            "Katakana%": round((scripts["Katakana"]/total_tokens*100), 1) if total_tokens > 0 else 0,
            "Other_Script%": round((scripts["NA_Script"]/total_tokens*100), 1) if total_tokens > 0 else 0
        },
        "jgri": {"MMS": total_tokens / num_sentences, "LD": len(content_words)/total_tokens if total_tokens > 0 else 0, "VPS": len(verbs)/num_sentences, "MPN": len(adverbs)/len(nouns) if len(nouns) > 0 else 0},
        "jlpt": jlpt_counts
    }

# ===============================================
# --- 4. STREAMLIT UI ---
# ===============================================

st.set_page_config(layout="wide", page_title="Japanese Lexical Profiler")

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
    results, global_tokens = [], []
    for item in corpus:
        data = analyze_text(item['text'], tagger, jlpt_lists)
        global_tokens.extend(data["tokens_list"])
        lr = LexicalRichness(" ".join(data["tokens_list"])) if data["tokens_list"] else None
        
        row = {
            "File": item['name'], "Tokens": data["raw"]["Tokens"],
            "TTR": round(len(set(data['tokens_list']))/data["raw"]["Tokens"], 3) if data["raw"]["Tokens"] > 0 else 0,
            "MTLD": round(lr.mtld(), 2) if lr and data["raw"]["Tokens"] > 10 else 0,
            "Readability": data["raw"]["Readability"], "J-Level": get_jread_level(data["raw"]["Readability"]),
            "WPS": data["raw"]["WPS"],
            "Kango Count": data["raw"]["Kango_Raw"], "Percentage Kango": data["pct"]["Percentage Kango"],
            "Wago Count": data["raw"]["Wago_Raw"], "Percentage Wago": data["pct"]["Percentage Wago"],
            "Verbs Count": data["raw"]["Verbs_Raw"], "Percentage Verbs": data["pct"]["Percentage Verbs"],
            "Particles Count": data["raw"]["Particles_Raw"], "Percentage Particles": data["pct"]["Percentage Particles"],
            "Kanji Count": data["raw"]["Kanji_Raw"], "Kanji%": data["pct"]["Kanji%"],
            "Hiragana Count": data["raw"]["Hiragana_Raw"], "Hiragana%": data["pct"]["Hiragana%"],
            "Katakana Count": data["raw"]["Katakana_Raw"], "Katakana%": data["pct"]["Katakana%"],
            "Other Scripts Count": data["raw"]["Other_Script_Raw"], "Other Scripts%": data["pct"]["Other_Script%"],
            **data["jgri"]
        }
        for lvl in ["N1", "N2", "N3", "N4", "N5", "NA"]:
            row[lvl], row[f"{lvl}%"] = data["jlpt"][lvl], round((data["jlpt"][lvl]/data["raw"]["Tokens"]*100), 1) if data["raw"]["Tokens"] > 0 else 0
        results.append(row)

    df = pd.DataFrame(results)
    for col in ["MMS", "LD", "VPS", "MPN"]:
        df[f"z_{col}"] = zscore(df[col]) if df[col].std() != 0 else 0
    df["JGRI"] = df[[f"z_{col}" for col in ["MMS", "LD", "VPS", "MPN"]]].mean(axis=1).round(3)
    df["Complexity"] = df["JGRI"].apply(get_jgri_interpretation)

    # --- DISPLAY MATRIX ---
    st.header("Analysis Matrix")
    
    # Tooltip Configuration
    column_config = {
        "Tokens": st.column_config.NumberColumn("Tokens", help=TOOLTIPS["Tokens"]),
        "TTR": st.column_config.NumberColumn("TTR", help=TOOLTIPS["TTR"]),
        "MTLD": st.column_config.NumberColumn("MTLD", help=TOOLTIPS["MTLD"]),
        "Readability": st.column_config.NumberColumn("Readability", help=TOOLTIPS["Readability"]),
        "JGRI": st.column_config.NumberColumn("JGRI", help=TOOLTIPS["JGRI"]),
        "Kanji Count": st.column_config.NumberColumn("Kanji Count", help=TOOLTIPS["Scripts"]),
        "Hiragana Count": st.column_config.NumberColumn("Hiragana Count", help=TOOLTIPS["Scripts"]),
        "Katakana Count": st.column_config.NumberColumn("Katakana Count", help=TOOLTIPS["Scripts"]),
        "Other Scripts Count": st.column_config.NumberColumn("Other Scripts Count", help=TOOLTIPS["Scripts"])
    }

    cols = ["File", "Tokens", "TTR", "MTLD", "Readability", "J-Level", "JGRI", "Complexity", "WPS", 
            "Kango Count", "Percentage Kango", "Wago Count", "Percentage Wago", 
            "Verbs Count", "Percentage Verbs", "Particles Count", "Percentage Particles",
            "Kanji Count", "Kanji%", "Hiragana Count", "Hiragana%", 
            "Katakana Count", "Katakana%", "Other Scripts Count", "Other Scripts%"] + \
           ["N1", "N1%", "N2", "N2%", "N3", "N3%", "N4", "N4%", "N5", "N5%", "NA", "NA%"]
    
    st.dataframe(df[cols], column_config=column_config, use_container_width=True)

    # --- N-GRAM TABLES ---
    st.divider()
    st.header("N-Gram Frequencies")
    total_tokens = len(global_tokens)
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
            df_g['PMW'] = df_g['Raw Freq'].apply(lambda x: round((x / total_tokens) * 1_000_000, 2))
            st.dataframe(df_g.rename(columns={'PMW': 'Freq (per Million)'}), hide_index=True)
else:
    st.info("Awaiting data input...")
