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
    "Tokens": "Corpus size: Total tokens detected.",
    "TTR": "Type-Token Ratio (V/N). Higher = More diverse.",
    "MTLD": "Lexical Diversity (Length-independent). > 80 is Advanced.",
    "Readability": "JReadability (Hasebe & Lee 2015). Lower = More advanced text.",
    "JGRI": "Relative Complexity: Z-score average of MMS, LD, VPS, and MPN."
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

    # Detailed POS Extraction (10 Categories)
    pos_map = {
        "Noun (名詞)": "名詞", "Verb (動詞)": "動詞", "Particle (助詞)": "助詞", 
        "Adverb (副詞)": "副詞", "Adjective (形容詞)": "形容詞", "Auxiliary (助動詞)": "助動詞", 
        "Conjunction (接続詞)": "接続詞", "Pronoun (代名詞)": "代名詞", 
        "Determiner (連体詞)": "連体詞", "Interjection (感動詞)": "感動詞"
    }
    pos_counts = {k: sum(1 for n in valid_nodes if n.feature.pos1 == v) for k, v in pos_map.items()}
    
    # JLPT & Formulas
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
    jread = (11.724 + (wps * -0.056) + (pk * -0.126) + (ph * -0.042) + (pv * -0.145) + (pp * -0.044))

    return {
        "tokens": [{"surface": n.surface, "lemma": n.feature.orth if hasattr(n.feature, 'orth') else n.surface, "pos": n.feature.pos1} for n in valid_nodes],
        "stats": {"Tokens": total_tokens, "WPS": round(wps, 2), "Readability": round(jread, 3), "K_Raw": scripts["K"], "H_Raw": scripts["H"], "T_Raw": scripts["T"], "O_Raw": scripts["NA"]},
        "jlpt": jlpt_counts, "pos_full": pos_counts,
        "jgri": {"MMS": total_tokens/num_sentences, "LD": sum(pos_counts.values())/total_tokens if total_tokens > 0 else 0, "VPS": pos_counts["Verb (動詞)"]/num_sentences, "MPN": pos_counts["Adverb (副詞)"]/pos_counts["Noun (名詞)"] if pos_counts["Noun (名詞)"] > 0 else 0}
    }

# ===============================================
# --- 3. UI HELPER ---
# ===============================================

def add_download_button(fig, filename):
    img_bytes = fig.to_image(format="png")
    st.download_button(label=f"📥 Download {filename}", data=img_bytes, file_name=f"{filename}.png", mime="image/png")

# ===============================================
# --- 4. MAIN APP ---
# ===============================================

st.set_page_config(layout="wide", page_title="Japanese Lexical Profiler")

if st.sidebar.text_input("Developer Password", type="password") != "290683":
    st.info("Please enter the password in the sidebar to proceed.")
    st.stop()

tagger, jlpt_wordlists = Tagger(), load_jlpt_wordlists()
st.title("📖 Japanese Text Vocabulary Profiler")

# Sidebar N-Gram Pattern
st.sidebar.header("Advanced N-Gram Pattern")
n_gram_size = st.sidebar.number_input("N-Gram Size", 1, 5, 2)
p_words, p_pos = [], []
for i in range(n_gram_size):
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
    results, pos_results, all_tokens = [], [], []
    for item in corpus:
        data = analyze_text(item['text'], tagger, jlpt_wordlists)
        all_tokens.extend(data["tokens"])
        total = data["stats"]["Tokens"]
        lr = LexicalRichness(" ".join([t['surface'] for t in data["tokens"]])) if total > 10 else None
        
        row = {"File": item['name'], "Tokens": total, "TTR": round(len(set([t['lemma'] for t in data["tokens"]]))/total, 3) if total > 0 else 0, "MTLD": round(lr.mtld(), 2) if lr else 0, "Readability": data["stats"]["Readability"], "WPS": data["stats"]["WPS"], "Kanji%": round(data["stats"]["K_Raw"]/total*100, 1) if total > 0 else 0, "Hira%": round(data["stats"]["H_Raw"]/total*100, 1) if total > 0 else 0, "Kata%": round(data["stats"]["T_Raw"]/total*100, 1) if total > 0 else 0, "Other%": round(data["stats"]["O_Raw"]/total*100, 1) if total > 0 else 0, **data["jgri"]}
        for lvl in ["N1", "N2", "N3", "N4", "N5", "NA"]:
            row[lvl], row[f"{lvl}%"] = data["jlpt"][lvl], round((data["jlpt"][lvl]/total*100), 1) if total > 0 else 0
        results.append(row)

        p_row = {"File": item['name'], "Tokens": total}
        for lbl, count in data["pos_full"].items():
            p_row[f"{lbl} (Raw)"], p_row[f"{lbl} (%)"] = count, round((count/total*100), 2) if total > 0 else 0
        pos_results.append(p_row)

    df = pd.DataFrame(results)
    for c in ["MMS", "LD", "VPS", "MPN"]:
        df[f"z_{c}"] = zscore(df[c]) if df[c].std() != 0 else 0
    df["JGRI"] = df[[f"z_{c}" for c in ["MMS", "LD", "VPS", "MPN"]]].mean(axis=1).round(3)

    tab_mat, tab_pos = st.tabs(["📊 General Analysis", "📝 POS Distribution"])
    with tab_mat:
        st.dataframe(df, use_container_width=True)
        st.divider()
        st.header("📈 Visualizations")
        
        viz_tasks = [("Tokens", "Tokens per File"), ("TTR", "Type-Token Ratio"), ("MTLD", "Lexical Diversity (MTLD)"), ("Readability", "JReadability Score"), ("JGRI", "Relative Grammatical Complexity (JGRI)")]
        for key, title in viz_tasks:
            fig = px.bar(df, x="File", y=key, title=title)
            st.plotly_chart(fig, use_container_width=True)
            add_download_button(fig, key)

        # Stacked Script Dist
        df_s = df.melt(id_vars=["File"], value_vars=["Kanji%", "Hira%", "Kata%", "Other%"], var_name="Script", value_name="%")
        fig_s = px.bar(df_s, x="File", y="%", color="Script", title="Script Distribution (%)", barmode="stack")
        st.plotly_chart(fig_s, use_container_width=True)
        add_download_button(fig_s, "Script_Distribution")

        # Stacked JLPT Dist
        df_j = df.melt(id_vars=["File"], value_vars=["N1%", "N2%", "N3%", "N4%", "N5%", "NA%"], var_name="Level", value_name="%")
        fig_j = px.bar(df_j, x="File", y="%", color="Level", title="JLPT Distribution (%)", barmode="stack", category_orders={"Level": ["N1%", "N2%", "N3%", "N4%", "N5%", "NA%"]})
        st.plotly_chart(fig_j, use_container_width=True)
        add_download_button(fig_j, "JLPT_Distribution")

    with tab_pos:
        st.header("POS Distribution (English & 日本語)")
        st.dataframe(pd.DataFrame(pos_results), use_container_width=True)

    # N-Gram Pattern Matching
    st.divider()
    st.header("N-Gram Pattern Matching")
    matches = []
    for j in range(len(all_tokens) - n_gram_size + 1):
        window, match = all_tokens[j : j + n_gram_size], True
        for idx in range(n_gram_size):
            w_pat, p_pat = p_words[idx].strip(), p_pos[idx]
            regex_str = "^" + w_pat.replace("*", ".*") + "$"
            if w_pat != "*" and not re.search(regex_str, window[idx]['surface']) and not re.search(regex_str, window[idx]['lemma']): match = False; break
            if p_pat != "Any" and window[idx]['pos'] != p_pat: match = False; break
        if match: matches.append(" ".join([t['surface'] for t in window]))
    
    if matches:
        df_g = pd.DataFrame(Counter(matches).most_common(10), columns=['Sequence', 'Raw Freq'])
        df_g['PMW'] = df_g['Raw Freq'].apply(lambda x: round((x / len(all_tokens)) * 1_000_000, 2))
        st.dataframe(df_g, use_container_width=True)
else: st.info("Awaiting data input...")
