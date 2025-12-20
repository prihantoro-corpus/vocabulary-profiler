import streamlit as st
import pandas as pd
import io
import re
import requests
import numpy as np
import plotly.express as px
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
    "TTR": "Type-Token Ratio (V/N). Thresholds:\n- < 0.45: Repetitive\n- 0.45-0.65: Moderate\n- > 0.65: Varied/Diverse",
    "MTLD": "Measuring Textual Lexical Diversity. Thresholds:\n- < 40: Basic\n- 40-80: Intermediate\n- > 80: Advanced",
    "Readability": "JReadability (Hasebe & Lee 2015). Scale: 0.5 (Advanced) to 6.5 (Elementary).",
    "JGRI": "Japanese Grammar Readability Index (Relative Complexity). Positive = More complex than corpus average."
}

POS_OPTIONS = ["Any (*)", "名詞 (Noun)", "動詞 (Verb)", "助詞 (Particle)", "副詞 (Adverb)", "形容詞 (Adjective)", "助動詞 (Auxiliary)", "接続詞 (Conjunction)", "代名詞 (Pronoun)", "連体詞 (Determiner)", "感動詞 (Interjection)"]

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
    valid_nodes = [n for n in nodes if n.surface and n.feature.pos1 != "補助記号"]
    sentences = [s for s in re.split(r'[。！？\n]', text.strip()) if s.strip()]
    num_sentences = len(sentences) if sentences else 1
    total_tokens = len(valid_nodes)
    
    # Scripts
    scripts = {"K": 0, "H": 0, "T": 0, "NA": 0}
    for n in valid_nodes:
        if re.search(r'[\u4e00-\u9faf]', n.surface): scripts["K"] += 1
        elif re.search(r'[\u3040-\u309f]', n.surface): scripts["H"] += 1
        elif re.search(r'[\u30a0-\u30ff]', n.surface): scripts["T"] += 1
        else: scripts["NA"] += 1

    # Detailed POS (10 Categories)
    pos_map = {
        "Noun (名詞)": "名詞", "Verb (動詞)": "動詞", "Particle (助詞)": "助詞", 
        "Adverb (副詞)": "副詞", "Adjective (形容詞)": "形容詞", "Auxiliary (助動詞)": "助動詞", 
        "Conjunction (接続詞)": "接続詞", "Pronoun (代名詞)": "代名詞", 
        "Determiner (連体詞)": "連体詞", "Interjection (感動詞)": "感動詞"
    }
    pos_counts = {k: sum(1 for n in valid_nodes if n.feature.pos1 == v) for k, v in pos_map.items()}
    
    # JLPT & JReadability
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

    wps = total_tokens / num_sentences
    pk, ph, pv, pp = [(x/total_tokens*100) if total_tokens > 0 else 0 for x in [scripts["K"], scripts["H"], pos_counts["Verb (動詞)"], pos_counts["Particle (助詞)"]]]
    jread_score = (11.724 + (wps * -0.056) + (pk * -0.126) + (ph * -0.042) + (pv * -0.145) + (pp * -0.044))

    return {
        "tokens": [{"surface": n.surface, "lemma": n.feature.orth if hasattr(n.feature, 'orth') else n.surface, "pos": n.feature.pos1} for n in valid_nodes],
        "stats": {"Tokens": total_tokens, "WPS": round(wps, 2), "Readability": round(jread_score, 3), "K_Raw": scripts["K"], "H_Raw": scripts["H"], "T_Raw": scripts["T"], "O_Raw": scripts["NA"]},
        "jlpt": jlpt_counts, "pos_full": pos_counts,
        "jgri": {"MMS": total_tokens/num_sentences, "LD": sum(pos_counts.values())/total_tokens if total_tokens > 0 else 0, "VPS": pos_counts["Verb (動詞)"]/num_sentences, "MPN": pos_counts["Adverb (副詞)"]/pos_counts["Noun (名詞)"] if pos_counts["Noun (名詞)"] > 0 else 0}
    }

# ===============================================
# --- 3. UI LAYOUT ---
# ===============================================

st.set_page_config(layout="wide", page_title="Japanese Lexical Profiler")

if st.sidebar.text_input("If you are a developer, tester, or reviewer, enter password", type="password") != "290683":
    st.info("Please enter the password in the sidebar to proceed.")
    st.stop()

tagger, jlpt_wordlists = Tagger(), load_jlpt_wordlists()
st.title("📖 Japanese Text Vocabulary Profiler")

# Sidebar: Advanced N-Gram Search
st.sidebar.header("Advanced N-Gram Pattern")
n_val = st.sidebar.number_input("N-Gram Size", 1, 5, 2)
p_words, p_pos = [], []
for i in range(n_val):
    st.sidebar.write(f"**Position {i+1}**")
    c1, c2 = st.sidebar.columns(2)
    p_words.append(c1.text_input("Word/*", value="*", key=f"w_{i}"))
    p_pos.append(c2.selectbox("POS Tag", options=POS_OPTIONS, key=f"p_{i}").split(" ")[0])

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
    results, pos_results, all_tokens = [], [], []
    for item in corpus:
        data = analyze_text(item['text'], tagger, jlpt_wordlists)
        all_tokens.extend(data["tokens"])
        total = data["stats"]["Tokens"]
        lr = LexicalRichness(" ".join([t['surface'] for t in data["tokens"]])) if total > 10 else None
        
        row = {
            "File": item['name'], "Tokens": total, "TTR": round(len(set([t['lemma'] for t in data["tokens"]]))/total, 3) if total > 0 else 0,
            "MTLD": round(lr.mtld(), 2) if lr else 0, "Readability": data["stats"]["Readability"], "J-Level": get_jread_level(data["stats"]["Readability"]),
            "WPS": data["stats"]["WPS"], 
            "Percentage Kango": round(data["stats"]["K_Raw"]/total*100, 2) if total > 0 else 0,
            "Percentage Wago": round(data["stats"]["H_Raw"]/total*100, 2) if total > 0 else 0,
            "Percentage Verbs": round(data["stats"]["V_Raw"]/total*100, 2) if total > 0 else 0,
            "Percentage Particles": round(data["stats"]["P_Raw"]/total*100, 2) if total > 0 else 0,
            "Kanji Count": data["stats"]["K_Raw"], "Kanji%": round(data["stats"]["K_Raw"]/total*100, 1) if total > 0 else 0,
            "Hira Count": data["stats"]["H_Raw"], "Hira%": round(data["stats"]["H_Raw"]/total*100, 1) if total > 0 else 0,
            "Kata Count": data["stats"]["T_Raw"], "Kata%": round(data["stats"]["T_Raw"]/total*100, 1) if total > 0 else 0,
            "Other Count": data["stats"]["O_Raw"], "Other%": round(data["stats"]["O_Raw"]/total*100, 1) if total > 0 else 0,
            **data["jgri"]
        }
        for lvl in ["N1", "N2", "N3", "N4", "N5", "NA"]:
            row[lvl], row[f"{lvl}%"] = data["jlpt"][lvl], round((data["jlpt"][lvl]/total*100), 1) if total > 0 else 0
        results.append(row)

        p_row = {"File": item['name'], "Tokens": total}
        for lbl, count in data["pos_full"].items():
            p_row[f"{lbl} (Raw)"] = count
            p_row[f"{lbl} (%)"] = round((count/total*100), 2) if total > 0 else 0
        pos_results.append(p_row)

    df_gen = pd.DataFrame(results)
    for c in ["MMS", "LD", "VPS", "MPN"]:
        df_gen[f"z_{c}"] = zscore(df_gen[c]) if df_gen[c].std() != 0 else 0
    df_gen["JGRI"] = df_gen[[f"z_{c}" for c in ["MMS", "LD", "VPS", "MPN"]]].mean(axis=1).round(3)

    tab1, tab2 = st.tabs(["📊 General Analysis", "📝 POS Distribution"])
    with tab1:
        st.header("Analysis Matrix")
        cfg = {k: st.column_config.NumberColumn(k, help=v) for k, v in TOOLTIPS.items()}
        disp = ["File", "Tokens", "TTR", "MTLD", "Readability", "J-Level", "JGRI", "WPS", "Percentage Kango", "Percentage Wago", "Percentage Verbs", "Percentage Particles", "Kanji Count", "Kanji%", "Hira Count", "Hira%", "Kata Count", "Kata%", "Other Count", "Other%"] + [f"{l}{s}" for l in ["N1","N2","N3","N4","N5","NA"] for s in ["", "%"]]
        st.dataframe(df_gen[disp], column_config=cfg, use_container_width=True)

        st.header("📈 Visualizations")
        st.plotly_chart(px.bar(df_gen, x="File", y="Tokens", title="1. Tokens per File"), use_container_width=True)
        st.plotly_chart(px.bar(df_gen, x="File", y="TTR", title="2. TTR"), use_container_width=True)
        st.plotly_chart(px.bar(df_gen, x="File", y="MTLD", title="3. MTLD"), use_container_width=True)
        st.plotly_chart(px.bar(df_gen, x="File", y="Readability", title="4. JReadability"), use_container_width=True)
        st.plotly_chart(px.bar(df_gen, x="File", y="JGRI", title="5. JGRI"), use_container_width=True)
        
        # Script Stacked
        df_s = df_gen.melt(id_vars=["File"], value_vars=["Kanji%", "Hira%", "Kata%", "Other%"], var_name="Script", value_name="%")
        st.plotly_chart(px.bar(df_s, x="File", y="%", color="Script", title="6. Script Distribution (%)", barmode="stack"), use_container_width=True)
        
        # JLPT Stacked
        df_j = df_gen.melt(id_vars=["File"], value_vars=["N1%", "N2%", "N3%", "N4%", "N5%", "NA%"], var_name="Level", value_name="%")
        st.plotly_chart(px.bar(df_j, x="File", y="%", color="Level", title="7. JLPT Distribution (%)", barmode="stack", category_orders={"Level": ["N1%", "N2%", "N3%", "N4%", "N5%", "NA%"]}), use_container_width=True)

    with tab2:
        st.header("POS Distribution (English & 日本語)")
        st.dataframe(pd.DataFrame(pos_results), use_container_width=True)

    # N-Gram Logic
    st.divider()
    st.header(f"N-Gram Pattern Results")
    matches = []
    for j in range(len(all_tokens) - n_val + 1):
        window, match = all_tokens[j : j + n_val], True
        for idx in range(n_val):
            w_pat, p_pat = p_words[idx].strip(), p_pos[idx]
            regex_str = "^" + w_pat.replace("*", ".*") + "$"
            if w_pat != "*" and not re.search(regex_str, window[idx]['surface']) and not re.search(regex_str, window[idx]['lemma']): match = False; break
            if p_pat != "Any" and window[idx]['pos'] != p_pat: match = False; break
        if match: matches.append(" ".join([t['surface'] for t in window]))
    
    if matches:
        df_g = pd.DataFrame(Counter(matches).most_common(10), columns=['Sequence', 'Raw Freq'])
        df_g['Freq (PMW)'] = df_g['Raw Freq'].apply(lambda x: round((x / len(all_tokens)) * 1_000_000, 2))
        st.dataframe(df_g, use_container_width=True)
else:
    st.info("Awaiting input.")
