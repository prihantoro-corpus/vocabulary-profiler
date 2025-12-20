import streamlit as st
import pandas as pd
import io
import re
import requests
import numpy as np
import plotly.express as px
import matplotlib.pyplot as plt
from collections import Counter
from fugashi import Tagger
from lexicalrichness import LexicalRichness
from scipy.stats import zscore
from wordcloud import WordCloud
import os

# ===============================================
# --- 1. CONFIGURATION & MAPPINGS ---
# ===============================================

GITHUB_BASE = "https://raw.githubusercontent.com/prihantoro-corpus/vocabulary-profiler/main/"
JLPT_FILES = {"N1": "unknown_source_N1.csv", "N2": "unknown_source_N2.csv", "N3": "unknown_source_N3.csv", "N4": "unknown_source_N4.csv", "N5": "unknown_source_N5.csv"}

POS_FULL_MAP = {
    "Noun (名詞)": "名詞", "Verb (動詞)": "動詞", "Particle (助詞)": "助詞",
    "Adverb (副詞)": "副詞", "Adjective (形容詞)": "形容詞", "Adjectival Noun (形状詞)": "形状詞",
    "Auxiliary Verb (助動詞)": "助動詞", "Conjunction (接続詞)": "接続詞",
    "Pronoun (代名詞)": "代名詞", "Determiner (連体詞)": "連体詞",
    "Interjection (感動詞)": "感動詞", "Prefix (接頭辞)": "接頭辞",
    "Suffix (接尾辞)": "接尾辞", "Symbol/Punc (補助記号)": "補助記号"
}

# ===============================================
# --- 2. UTILITY & LINGUISTIC FUNCTIONS ---
# ===============================================

def add_html_download_button(fig, filename):
    buffer = io.StringIO()
    fig.write_html(buffer, include_plotlyjs='cdn')
    html_bytes = buffer.getvalue().encode()
    st.download_button(label=f"📥 Download {filename} (HTML)", data=html_bytes, file_name=f"{filename}.html", mime="text/html")

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

def analyze_text(text, filename, tagger, jlpt_lists):
    nodes = tagger(text)
    all_nodes = []
    for n in nodes:
        if n.surface:
            lemma = n.feature.orth if hasattr(n.feature, 'orth') and n.feature.orth else n.surface
            all_nodes.append({
                "surface": n.surface,
                "lemma": lemma,
                "pos": n.feature.pos1,
                "file": filename
            })
    
    valid_nodes = [n for n in all_nodes if n['pos'] != "補助記号"]
    sentences = [s for s in re.split(r'[。！？\n]', text.strip()) if s.strip()]
    num_sentences = len(sentences) if sentences else 1
    total_tokens_valid = len(valid_nodes)
    
    scripts = {"K": 0, "H": 0, "T": 0, "NA": 0}
    for n in valid_nodes:
        if re.search(r'[\u4e00-\u9faf]', n['surface']): scripts["K"] += 1
        elif re.search(r'[\u3040-\u309f]', n['surface']): scripts["H"] += 1
        elif re.search(r'[\u30a0-\u30ff]', n['surface']): scripts["T"] += 1
        else: scripts["NA"] += 1

    pos_counts_raw = {k: sum(1 for n in all_nodes if n['pos'] == v) for k, v in POS_FULL_MAP.items()}
    
    jlpt_counts = {lvl: 0 for lvl in ["N1", "N2", "N3", "N4", "N5", "NA"]}
    for n in valid_nodes:
        found = False
        for lvl in ["N1", "N2", "N3", "N4", "N5"]:
            if n['lemma'] in jlpt_lists[lvl]:
                jlpt_counts[lvl] += 1
                found = True
                break
        if not found: jlpt_counts["NA"] += 1

    wps = total_tokens_valid / num_sentences
    pk = (scripts["K"]/total_tokens_valid*100) if total_tokens_valid > 0 else 0
    ph = (scripts["H"]/total_tokens_valid*100) if total_tokens_valid > 0 else 0
    pv = (pos_counts_raw["Verb (動詞)"]/total_tokens_valid*100) if total_tokens_valid > 0 else 0
    pp = (pos_counts_raw["Particle (助詞)"]/total_tokens_valid*100) if total_tokens_valid > 0 else 0
    jread = (11.724 + (wps * -0.056) + (pk * -0.126) + (ph * -0.042) + (pv * -0.145) + (pp * -0.044)) if total_tokens_valid > 0 else 0

    content_words = sum(1 for n in valid_nodes if n['pos'] in ["名詞", "動詞", "形容詞", "副詞", "形状詞"])

    return {
        "all_tokens": all_nodes,
        "stats": {"T_Valid": total_tokens_valid, "T_All": len(all_nodes), "WPS": round(wps, 2), "Read": round(jread, 3), "K%": round(pk, 1), "H%": round(ph, 1), "T%": round((scripts["T"]/total_tokens_valid*100), 1) if total_tokens_valid > 0 else 0, "O%": round((scripts["NA"]/total_tokens_valid*100), 1) if total_tokens_valid > 0 else 0},
        "jlpt": jlpt_counts, 
        "pos_raw": pos_counts_raw,
        "jgri_base": {"MMS": wps, "LD": content_words/total_tokens_valid if total_tokens_valid > 0 else 0, "VPS": pos_counts_raw["Verb (動詞)"]/num_sentences, "MPN": pos_counts_raw["Adverb (副詞)"]/pos_counts_raw["Noun (名詞)"] if pos_counts_raw["Noun (名詞)"] > 0 else 0}
    }

# ===============================================
# --- 3. STREAMLIT APPLICATION ---
# ===============================================

st.set_page_config(layout="wide", page_title="Japanese Lexical Profiler")

if st.sidebar.text_input("Access Password", type="password") != "290683":
    st.info("Enter password to unlock analysis.")
    st.stop()

tagger, jlpt_wordlists = Tagger(), load_jlpt_wordlists()
st.title("📖 Japanese Text Vocabulary Profiler")

# Sidebar
st.sidebar.header("Advanced N-Gram Pattern")
n_size = st.sidebar.number_input("N-Gram Size", 1, 5, 1) # Default to 1 for your request
p_words, p_tags = [], []
for i in range(n_size):
    st.sidebar.write(f"**Position {i+1}**")
    c1, c2 = st.sidebar.columns(2)
    p_words.append(c1.text_input("Word/Regex", value="*", key=f"w_{i}"))
    p_tags.append(c2.selectbox("POS Tag", options=["Any (*)"] + list(POS_FULL_MAP.keys()), key=f"t_{i}"))

st.sidebar.header("Concordance Settings")
left_context_size = st.sidebar.slider("Left Context (Words)", 1, 15, 5)
right_context_size = st.sidebar.slider("Right Context (Words)", 1, 15, 5)

source = st.sidebar.selectbox("Data Source", ["Upload Files", "DICO-JALF 30", "DICO-JALF ALL"])
corpus = []
if source == "Upload Files":
    up = st.sidebar.file_uploader("Upload .txt or .xlsx", accept_multiple_files=True)
    if up:
        for f in up: corpus.append({"name": f.name, "text": f.read().decode('utf-8')})
else:
    url_key = "all" if "ALL" in source else "30%20files%20only"
    url = f"https://raw.githubusercontent.com/prihantoro-corpus/vocabulary-profiler/main/DICO-JALF%20{url_key}.xlsx"
    df_pre = pd.read_excel(io.BytesIO(requests.get(url).content), header=None)
    corpus = [{"name": str(r[0]), "text": str(r[1])} for _, r in df_pre.iterrows()]

if corpus:
    res_gen, res_pos, global_toks_all = [], [], []
    for item in corpus:
        data = analyze_text(item['text'], item['name'], tagger, jlpt_wordlists)
        global_toks_all.extend(data["all_tokens"])
        t_v = data["stats"]["T_Valid"]
        res_gen.append({"File": item['name'], "Tokens": t_v, "TTR": round(len(set([t['lemma'] for t in data["all_tokens"] if t['pos'] != "補助記号"]))/t_v, 3) if t_v > 0 else 0, "Readability": data["stats"]["Read"], **data["jgri_base"]})
        res_pos.append({"File": item['name'], **{f"{k} [%]": round((v/data['stats']['T_All']*100), 2) for k,v in data['pos_raw'].items()}})

    tab_mat, tab_pos = st.tabs(["📊 General Analysis", "📝 Full POS Distribution"])
    
    with tab_mat:
        st.dataframe(pd.DataFrame(res_gen), use_container_width=True)

        # --- UPDATED FLEXIBLE SEARCH LOGIC ---
        st.divider()
        st.header("🔍 Pattern Search & Concordance (KWIC)")
        
        filtered_toks = [t for t in global_toks_all if t['pos'] != "補助記号"]
        t_filtered = len(filtered_toks)
        
        matches_data, concordance_rows = [], []
        for j in range(t_filtered - n_size + 1):
            window, match = filtered_toks[j : j + n_size], True
            for idx in range(n_size):
                w_p_input = p_words[idx].strip()
                t_p_selection = p_tags[idx]
                
                # Logic: If select "Noun (名詞)", get "名詞"
                target_pos_tag = t_p_selection.split(" ")[-1].strip("()") if "(" in t_p_selection else t_p_selection
                
                tok_surf = window[idx].get('surface') or ""
                tok_lem = window[idx].get('lemma') or ""
                tok_pos = window[idx].get('pos') or ""
                
                # Flexible Word Match
                word_match = (w_p_input == "*") or (re.search("^" + w_p_input.replace("*", ".*") + "$", tok_surf) or re.search("^" + w_p_input.replace("*", ".*") + "$", tok_lem))
                
                # Flexible POS Match (Check if selected tag is IN the token POS)
                pos_match = (t_p_selection == "Any (*)") or (target_pos_tag in tok_pos)
                
                if not (word_match and pos_match):
                    match = False
                    break
            
            if match:
                gram_text = " ".join([t['surface'] for t in window])
                gram_pos = " + ".join([t['pos'] for t in window])
                matches_data.append((gram_text, gram_pos))
                l_context = "".join([t['surface'] for t in filtered_toks[max(0, j-left_context_size) : j]])
                r_context = "".join([t['surface'] for t in filtered_toks[j+n_size : min(t_filtered, j+n_size+right_context_size)]])
                concordance_rows.append({"File": window[0]['file'], "Left": l_context, "KWIC": "".join([t['surface'] for t in window]), "Right": r_context})
        
        if matches_data:
            c1, c2 = st.columns([1, 2])
            with c1:
                df_counts = pd.DataFrame([{"Sequence": k[0], "POS": k[1], "Freq": v} for k, v in Counter(matches_data).most_common()])
                st.dataframe(df_counts.head(15), use_container_width=True)
            with c2:
                st.dataframe(pd.DataFrame(concordance_rows), use_container_width=True)
        else:
            st.warning("No sequences matched the specified pattern.")

        # --- VISUALIZATIONS ---
        st.divider()
        st.header("📈 Visualizations")
        cloud_tokens = [t['surface'] for t in filtered_toks if t['pos'] in ["名詞", "動詞", "形容詞", "副詞", "形状詞"]]
        if cloud_tokens and os.path.exists("NotoSansJP[wght].ttf"):
            wc = WordCloud(font_path="NotoSansJP[wght].ttf", background_color="white", width=800, height=350).generate(" ".join(cloud_tokens))
            fig, ax = plt.subplots(figsize=(10, 4)); ax.imshow(wc); ax.axis("off"); st.pyplot(fig)

    with tab_pos:
        st.dataframe(pd.DataFrame(res_pos), use_container_width=True)
else:
    st.info("Upload files to begin.")
