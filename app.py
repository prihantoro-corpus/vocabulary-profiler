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
    "TTR": "Type-Token Ratio. Thresholds: < 0.45: Repetitive | 0.45-0.65: Moderate | > 0.65: Varied.",
    "MTLD": "Lexical Diversity (Length-independent). Thresholds: < 40: Basic | 40-80: Intermediate | > 80: Advanced.",
    "Readability": "JReadability: 0.5-1.5: Upper-adv | 1.5-2.5: Lower-adv | 2.5-3.5: Upper-int | 3.5-4.5: Lower-int | 4.5-5.5: Upper-elem | 5.5-6.5: Lower-elem.",
    "JGRI": "Relative Complexity: < -1.0: Very easy | 0 to +1.0: Medium | > +1.0: High complexity.",
    "JLPT": "Distribution based on N1-N5 lists. NA = Not found in standard JLPT lists."
}

# POS Options for N-Gram dropdown
POS_OPTIONS = ["Any (*)", "名詞 (Noun)", "動詞 (Verb)", "助詞 (Particle)", "副詞 (Adverb)", "形容詞 (Adjective)", "助動詞 (Auxiliary)", "接続詞 (Conjunction)", "代名詞 (Pronoun)"]

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

def get_jgri_interpretation(val):
    if val < -1.0: return "Very easy"
    elif -1.0 <= val < 0: return "Relatively easy"
    elif 0 <= val < 1.0: return "Medium complexity"
    else: return "High complexity"

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

    # POS
    pos_map = {"Noun": "名詞", "Verb": "動詞", "Particle": "助詞", "Adverb": "副詞", "Adjective": "形容詞", "Auxiliary": "助動詞", "Conjunction": "接続詞"}
    pos_counts = {k: sum(1 for n in valid_nodes if n.feature.pos1 == v) for k, v in pos_map.items()}
    
    # JLPT
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
    pk, ph, pv, pp = [(x/total_tokens*100) if total_tokens > 0 else 0 for x in [scripts["K"], scripts["H"], pos_counts["Verb"], pos_counts["Particle"]]]
    jread_score = (11.724 + (wps * -0.056) + (pk * -0.126) + (ph * -0.042) + (pv * -0.145) + (pp * -0.044))

    return {
        "tokens": [{"surface": n.surface, "lemma": n.feature.orth if hasattr(n.feature, 'orth') else n.surface, "pos": n.feature.pos1} for n in valid_nodes],
        "stats": {"Tokens": total_tokens, "WPS": round(wps, 2), "Readability": round(jread_score, 3), "K_Raw": scripts["K"], "H_Raw": scripts["H"], "T_Raw": scripts["T"], "O_Raw": scripts["NA"], "V_Raw": pos_counts["Verb"], "P_Raw": pos_counts["Particle"]},
        "jlpt": jlpt_counts,
        "jgri": {"MMS": total_tokens/num_sentences, "LD": sum(pos_counts.values())/total_tokens if total_tokens > 0 else 0, "VPS": pos_counts["Verb"]/num_sentences, "MPN": pos_counts["Adverb"]/pos_counts["Noun"] if pos_counts["Noun"] > 0 else 0},
        "pos_full": pos_counts
    }

# ===============================================
# --- 4. STREAMLIT UI ---
# ===============================================

st.set_page_config(layout="wide", page_title="Japanese Lexical Profiler")

pwd = st.sidebar.text_input("If you are a developer, tester, or reviewer, enter password", type="password")
if pwd != "290683":
    st.info("If you are a developer, tester, or reviewer, enter the password in the sidebar.")
    st.stop()

tagger, jlpt_wordlists = Tagger(), load_jlpt_wordlists()
st.title("📖 Japanese Text Vocabulary Profiler")

# Sidebar: Advanced N-Gram Search
st.sidebar.header("Advanced N-Gram Pattern")
n_gram_val = st.sidebar.number_input("N-Gram Size", 1, 5, 2)
p_words, p_pos = [], []
for i in range(n_gram_val):
    st.sidebar.write(f"**Position {i+1}**")
    c1, c2 = st.sidebar.columns(2)
    p_words.append(c1.text_input("Regex/Word", value="*", key=f"w_{i}"))
    p_pos.append(c2.selectbox("POS Tag", options=POS_OPTIONS, key=f"p_{i}").split(" ")[0])

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
    results, pos_results, all_global_tokens = [], [], []
    for item in corpus:
        data = analyze_text(item['text'], tagger, jlpt_wordlists)
        all_global_tokens.extend(data["tokens"])
        total = data["stats"]["Tokens"]
        lr = LexicalRichness(" ".join([t['surface'] for t in data["tokens"]])) if total > 10 else None
        
        row = {
            "File": item['name'], "Tokens": total,
            "TTR": round(len(set([t['lemma'] for t in data["tokens"]]))/total, 3) if total > 0 else 0,
            "MTLD": round(lr.mtld(), 2) if lr else 0, "Readability": data["stats"]["Readability"], "J-Level": get_jread_level(data["stats"]["Readability"]),
            "WPS": data["stats"]["WPS"], "Percentage Kango": round(data["stats"]["K_Raw"]/total*100, 2) if total > 0 else 0,
            "Percentage Wago": round(data["stats"]["H_Raw"]/total*100, 2) if total > 0 else 0,
            "Percentage Verbs": round(data["stats"]["V_Raw"]/total*100, 2) if total > 0 else 0,
            "Percentage Particles": round(data["stats"]["P_Raw"]/total*100, 2) if total > 0 else 0,
            "Kanji%": round(data["stats"]["K_Raw"]/total*100, 1) if total > 0 else 0, "Hira%": round(data["stats"]["H_Raw"]/total*100, 1) if total > 0 else 0,
            "Kata%": round(data["stats"]["T_Raw"]/total*100, 1) if total > 0 else 0, "Other%": round(data["stats"]["O_Raw"]/total*100, 1) if total > 0 else 0,
            **data["jgri"]
        }
        for lvl in ["N1", "N2", "N3", "N4", "N5", "NA"]:
            row[lvl], row[f"{lvl}%"] = data["jlpt"][lvl], round((data["jlpt"][lvl]/total*100), 1) if total > 0 else 0
        results.append(row)

        p_row = {"File": item['name'], "Tokens": total}
        for pos_label, count in data["pos_full"].items():
            p_row[f"{pos_label} (%)"] = round((count/total*100), 2) if total > 0 else 0
        pos_results.append(p_row)

    df = pd.DataFrame(results)
    for col in ["MMS", "LD", "VPS", "MPN"]:
        df[f"z_{col}"] = zscore(df[col]) if df[col].std() != 0 else 0
    df["JGRI"] = df[[f"z_{col}" for col in ["MMS", "LD", "VPS", "MPN"]]].mean(axis=1).round(3)
    df["Complexity"] = df["JGRI"].apply(get_jgri_interpretation)

    tab_mat, tab_pos = st.tabs(["📊 General Analysis", "📝 POS Distribution"])
    with tab_mat:
        st.header("Analysis Matrix")
        cfg = {k: st.column_config.NumberColumn(k, help=v) for k, v in TOOLTIPS.items()}
        disp = ["File", "Tokens", "TTR", "MTLD", "Readability", "J-Level", "JGRI", "Complexity", "WPS", "Percentage Kango", "Percentage Wago", "Percentage Verbs", "Percentage Particles", "Kanji%", "Hira%", "Kata%", "Other%"] + [f"{l}{s}" for l in ["N1","N2","N3","N4","N5","NA"] for s in ["", "%"]]
        st.dataframe(df[disp], column_config=cfg, use_container_width=True)

        # --- 📈 Visualizations ---
        st.divider()
        st.header("📈 Visualizations")
        
        c1, c2 = st.columns(2)
        c1.plotly_chart(px.bar(df, x="File", y="Tokens", title="1. Tokens per File"), use_container_width=True)
        c2.plotly_chart(px.bar(df, x="File", y="TTR", title="2. TTR"), use_container_width=True)
        
        c3, c4 = st.columns(2)
        c3.plotly_chart(px.bar(df, x="File", y="MTLD", title="3. MTLD"), use_container_width=True)
        c4.plotly_chart(px.bar(df, x="File", y="Readability", title="4. JReadability"), use_container_width=True)
        
        st.plotly_chart(px.bar(df, x="File", y="JGRI", title="5. JGRI"), use_container_width=True)
        
        # 6. Script Distribution (Stacked)
        df_script = df.melt(id_vars=["File"], value_vars=["Kanji%", "Hira%", "Kata%", "Other%"], var_name="Script", value_name="%")
        st.plotly_chart(px.bar(df_script, x="File", y="%", color="Script", title="6. Script Distribution (%)", barmode="stack"), use_container_width=True)
        
        # 7. JLPT Distribution (Stacked)
        df_jlpt = df.melt(id_vars=["File"], value_vars=["N1%", "N2%", "N3%", "N4%", "N5%", "NA%"], var_name="Level", value_name="%")
        st.plotly_chart(px.bar(df_jlpt, x="File", y="%", color="Level", title="7. JLPT Distribution (%)", barmode="stack", category_orders={"Level": ["N1%", "N2%", "N3%", "N4%", "N5%", "NA%"]}), use_container_width=True)

    with tab_pos:
        st.header("POS Distribution (%)")
        st.dataframe(pd.DataFrame(pos_results), use_container_width=True)

    # --- N-GRAM PATTERN MATCHING ---
    st.divider()
    st.header(f"N-Gram Pattern Results")
    matches = []
    for j in range(len(all_global_tokens) - n_gram_val + 1):
        window, match = all_global_tokens[j : j + n_gram_val], True
        for idx in range(n_gram_val):
            w_pat, p_pat = p_words[idx].strip(), p_pos[idx]
            regex_str = "^" + w_pat.replace("*", ".*") + "$"
            if w_pat != "*" and not re.search(regex_str, window[idx]['surface']) and not re.search(regex_str, window[idx]['lemma']): match = False; break
            if p_pat != "Any" and window[idx]['pos'] != p_pat: match = False; break
        if match: matches.append(" ".join([t['surface'] for t in window]))
    
    if matches:
        df_g = pd.DataFrame(Counter(matches).most_common(10), columns=['Sequence', 'Raw Freq'])
        df_g['Freq (PMW)'] = df_g['Raw Freq'].apply(lambda x: round((x / len(all_global_tokens)) * 1_000_000, 2))
        st.dataframe(df_g, use_container_width=True)
    else: st.warning("No patterns matched.")
else: st.info("Awaiting data input...")
