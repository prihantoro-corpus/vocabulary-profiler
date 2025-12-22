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

# URL configuration for external assets
GITHUB_BASE = "https://raw.githubusercontent.com/prihantoro-corpus/vocabulary-profiler/main/"
JLPT_FILES = {
    "N1": "unknown_source_N1.csv",
    "N2": "unknown_source_N2.csv",
    "N3": "unknown_source_N3.csv",
    "N4": "unknown_source_N4.csv",
    "N5": "unknown_source_N5.csv"
}

ROUTLEDGE_URL = "https://raw.githubusercontent.com/prihantoro-corpus/vocabulary-profiler/main/Routledge%205000%20Vocab%20ONLY.xlsx"
LOCAL_ROUTLEDGE_CSV = "Routledge 5000 Vocab ONLY.xlsx - Sheet1.csv"

# Comprehensive Part of Speech Mapping
POS_FULL_MAP = {
    "Noun (名詞)": "名詞",
    "Verb (動詞)": "動詞",
    "Particle (助詞)": "助詞",
    "Adverb (副詞)": "副詞",
    "Adjective (形容詞)": "形容詞",
    "Adjectival Noun (形状詞)": "形状詞",
    "Auxiliary Verb (助動詞)": "助動詞",
    "Conjunction (接続詞)": "接続詞",
    "Pronoun (代名詞)": "代名詞",
    "Determiner (連体詞)": "連体詞",
    "Interjection (感動詞)": "感動詞",
    "Prefix (接頭辞)": "接頭辞",
    "Suffix (接尾辞)": "接尾辞",
    "Symbol/Punc (補助記号)": "補助記号"
}

# Detailed Tooltips for Matrix Scannability
TOOLTIPS = {
    "Tokens": "Total valid linguistic tokens (excluding punctuation).",
    "TTR": "Unique Words / Total Words (Lexical Variety).",
    "MTLD": "Measure of Textual Lexical Diversity (length-independent). Higher = more diverse.",
    "Readability": "JReadability (Lee & Hasebe). Score = 11.724 - 0.056a - 0.126b - 0.042c - 0.145d - 0.044e.",
    "J-Level": "Pedagogical level assigned based on JReadability score.",
    "JGRI": "Japanese Grammatical Relationship Index score (Standardized).",
    "WPS (a)": "Variable (a): Mean words per sentence (Sentence Length).",
    "Kango% (b)": "Variable (b): Percentage of Sino-Japanese words (Chinese origin).",
    "Wago% (c)": "Variable (c): Percentage of native Japanese words.",
    "V% (d)": "Variable (d): Verb density percentage.",
    "P% (e)": "Variable (e): Particle density percentage.",
    "K(raw)": "Count of Kanji script tokens.",
    "K%": "Percentage of tokens containing Kanji.",
    "H(raw)": "Count of Hiragana script tokens.",
    "H%": "Percentage of tokens in Hiragana script.",
    "T(raw)": "Count of Katakana script tokens.",
    "T%": "Percentage of tokens in Katakana script."
}

# Add Dynamic Tooltips for JLPT and Routledge Levels
for l in ["N1","N2","N3","N4","N5","NA"]:
    TOOLTIPS[f"{l}(raw)"] = f"Count of words matching JLPT {l} level."
    TOOLTIPS[f"{l}%"] = f"Percentage of text in JLPT {l} level."
for i in range(1, 6):
    TOOLTIPS[f"TOP-{i}000(raw)"] = f"Count of words in Routledge Top {i}000 list."
    TOOLTIPS[f"TOP-{i}000%"] = f"Percentage of text in Routledge Top {i}000 list."

# ===============================================
# --- 2. UTILITY & LINGUISTIC FUNCTIONS ---
# ===============================================

def add_html_download_button(fig, filename):
    buffer = io.StringIO()
    fig.write_html(buffer, include_plotlyjs='cdn')
    html_bytes = buffer.getvalue().encode()
    st.download_button(label=f"📥 Download Chart (HTML)", data=html_bytes, file_name=f"{filename}.html", mime="text/html")

@st.cache_data
def load_jlpt_wordlists():
    wordlists = {}
    for lvl, f in JLPT_FILES.items():
        try:
            df = pd.read_csv(GITHUB_BASE + f)
            wordlists[lvl] = set(df.iloc[:, 0].astype(str).tolist())
        except: wordlists[lvl] = set()
    return wordlists

@st.cache_data
def load_routledge_wordlist():
    df = None
    if os.path.exists(LOCAL_ROUTLEDGE_CSV):
        try: df = pd.read_csv(LOCAL_ROUTLEDGE_CSV, encoding='utf-8-sig')
        except: pass
    if df is None:
        try:
            resp = requests.get(ROUTLEDGE_URL)
            if resp.status_code == 200: df = pd.read_excel(io.BytesIO(resp.content))
        except: pass
    rout_map = {}
    if df is not None:
        df.columns = [str(c).lower().strip() for c in df.columns]
        for _, row in df.iterrows():
            lvl = str(row.get('level', '')).strip().upper()
            if not lvl: continue
            for col in ['hiragana', 'katakana', 'kanji']:
                val = str(row.get(col, '')).strip()
                if val and val.lower() != 'nan': rout_map[val] = lvl
    return rout_map

def katakana_to_hiragana(text):
    if not text: return ""
    return "".join([chr(ord(c) - 0x60) if "\u30a1" <= c <= "\u30f6" else c for c in text])

def get_jread_level(score):
    if 0.5 <= score < 1.5: return "Upper-advanced"
    elif 1.5 <= score < 2.5: return "Lower-advanced"
    elif 2.5 <= score < 3.5: return "Upper-intermediate"
    elif 3.5 <= score < 4.5: return "Lower-intermediate"
    elif 4.5 <= score < 5.5: return "Upper-elementary"
    elif 5.5 <= score < 6.5: return "Lower-elementary"
    else: return "Other"

def get_jgri_interp(score):
    if score > 0.5: return "Complex"
    elif score < -0.5: return "Simple"
    else: return "Standard"

def analyze_text(text, filename, tagger, jlpt_lists, routledge_list):
    nodes = tagger(text)
    all_nodes = []
    for n in nodes:
        if n.surface:
            f = n.feature
            lemma = f[7] if len(f) >= 8 and f[7] != '*' else n.surface
            reading = f[10] if len(f) >= 11 and f[10] != '*' else ""
            goshu = f[12] if len(f) > 12 else ""
            if not goshu and len(f) > 7:
                for field in f:
                    if field in ["和", "漢", "外", "混"]: goshu = field; break
            all_nodes.append({"surface": n.surface, "lemma": lemma, "reading": reading, "pos": f[0], "goshu": goshu, "file": filename})
    
    valid = [n for n in all_nodes if n['pos'] != "補助記号"]
    num_s = len([s for s in re.split(r'[。！？\n]', text.strip()) if s.strip()]) or 1
    t_v = len(valid)
    
    k_raw, h_raw, t_raw, kango_raw, wago_raw = 0, 0, 0, 0, 0
    for n in valid:
        if re.search(r'[\u4e00-\u9faf]', n['surface']): k_raw += 1
        elif re.search(r'[\u3040-\u309f]', n['surface']): h_raw += 1
        elif re.search(r'[\u30a0-\u30ff]', n['surface']): t_raw += 1
        if "漢" in n['goshu']: kango_raw += 1
        elif "和" in n['goshu']: wago_raw += 1

    pos_c = {k: sum(1 for n in all_nodes if n['pos'] == v) for k, v in POS_FULL_MAP.items()}
    v_raw, p_raw = pos_c.get("Verb (動詞)", 0), pos_c.get("Particle (助詞)", 0)
    wps = t_v / num_s
    pk, pw, pv, pp = (kango_raw/t_v*100), (wago_raw/t_v*100), (v_raw/t_v*100), (p_raw/t_v*100) if t_v > 0 else (0,0,0,0)
    jread = 11.724 - (0.056*wps) - (0.126*pk) - (0.042*pw) - (0.145*pv) - (0.044*pp)

    jlpt_c = {lvl: 0 for lvl in ["N1", "N2", "N3", "N4", "N5", "NA"]}
    rout_c = {f"TOP-{i}000": 0 for i in range(1, 6)}; rout_c["TOP-NA"] = 0
    for n in valid:
        f_j = False
        for lvl in ["N1", "N2", "N3", "N4", "N5"]:
            if n['lemma'] in jlpt_lists[lvl] or n['surface'] in jlpt_lists[lvl]:
                jlpt_c[lvl] += 1; f_j = True; break
        if not f_j: jlpt_c["NA"] += 1
        f_r = False
        r_hira = katakana_to_hiragana(n['reading'])
        for ck in [n['surface'], n['lemma'], r_hira]:
            if ck in routledge_list:
                lv = routledge_list[ck]
                if lv in rout_c: rout_c[lv] += 1; f_r = True; break
        if not f_r: rout_c["TOP-NA"] += 1

    cw = sum(1 for n in valid if n['pos'] in ["名詞", "動詞", "形容詞", "副詞", "形状詞"])
    return {
        "all_tokens": all_nodes,
        "stats": {
            "T_Valid": t_v, "WPS": round(wps, 2), "Read": round(jread, 3), "Kango%": round(pk, 1), "Wago%": round(pw, 1),
            "V%": round(pv, 1), "P%": round(pp, 1), "K%": round(k_raw/t_v*100, 1) if t_v > 0 else 0,
            "H%": round(h_raw/t_v*100, 1) if t_v > 0 else 0, "T%": round(t_raw/t_v*100, 1) if t_v > 0 else 0
        },
        "jlpt": jlpt_c, "rout": rout_c, "pos": pos_c,
        "jgri": {"MMS": wps, "LD": cw/t_v if t_v > 0 else 0, "VPS": v_raw/num_s, "MPN": pos_c.get("Adverb (副詞)", 0)/pos_c.get("Noun (名詞)", 1)}
    }

# ===============================================
# --- 3. STREAMLIT INTERFACE ---
# ===============================================

st.set_page_config(layout="wide", page_title="Japanese Lexical Profiler")
st.sidebar.title("📚 USER'S MANUAL")
st.sidebar.markdown("[Link to Manual](https://docs.google.com/document/d/1wFPY_b90K0NjS6dQEHJsjJDD_ZRbckq6vzY-kqMT9kE/edit?usp=sharing)")

source = st.sidebar.selectbox("📂 Select Data Source", ["Upload Files", "DICO-JALF 30", "DICO-JALF ALL"])
corpus = []
if source == "Upload Files":
    up = st.sidebar.file_uploader("Upload .txt or .xlsx files", accept_multiple_files=True)
    if up:
        for f in up: corpus.append({"name": f.name, "text": f.read().decode('utf-8', errors='ignore')})
else:
    uk = "all" if "ALL" in source else "30%20files%20only"
    url = f"https://raw.githubusercontent.com/prihantoro-corpus/vocabulary-profiler/main/DICO-JALF%20{uk}.xlsx"
    df_p = pd.read_excel(io.BytesIO(requests.get(url).content), header=None)
    corpus = [{"name": str(r[0]), "text": str(r[1])} for _, r in df_p.iterrows()]

if st.sidebar.text_input("Access Password", type="password") != "112233":
    st.info("Please unlock.")
    st.stop()

tagger = Tagger(); jlpt_ls = load_jlpt_wordlists(); rout_ls = load_routledge_wordlist()

st.title("📖 Japanese Text Vocabulary Profiler")
st.sidebar.divider(); st.sidebar.header("🔍 Search Settings")
n_s = st.sidebar.number_input("N-Gram Size", 1, 5, 1)
p_w, p_t = [], []
for i in range(n_s):
    st.sidebar.write(f"**Position {i+1}**")
    c1, c2 = st.sidebar.columns(2)
    p_w.append(c1.text_input("Word", "*", key=f"w_{i}"))
    p_t.append(c2.selectbox("POS", ["Any (*)"] + list(POS_FULL_MAP.keys()), key=f"t_{i}"))
l_ctx, r_ctx = st.sidebar.slider("Left Context", 1, 15, 5), st.sidebar.slider("Right Context", 1, 15, 5)

if corpus:
    res_gen, res_pos, glob_toks = [], [], []
    for item in corpus:
        d = analyze_text(item['text'], item['name'], tagger, jlpt_ls, rout_ls)
        glob_toks.extend(d["all_tokens"])
        tv = d["stats"]["T_Valid"]
        lr = LexicalRichness(" ".join([t['surface'] for t in d["all_tokens"] if t['pos'] != "補助記号"]))
        row = {
            "File": item['name'], "Tokens": tv, "TTR": round(lr.ttr, 3), "MTLD": round(lr.mtld(), 2),
            "Readability": d["stats"]["Read"], "J-Level": get_jread_level(d["stats"]["Read"]),
            "WPS (a)": d["stats"]["WPS"], "Kango% (b)": d["stats"]["Kango%"], "Wago% (c)": d["stats"]["Wago%"],
            "V% (d)": d["stats"]["V%"], "P% (e)": d["stats"]["P%"], "K%": d["stats"]["K%"], "H%": d["stats"]["H%"], "T%": d["stats"]["T%"],
            **d["jgri"]
        }
        for lvl in ["N1", "N2", "N3", "N4", "N5", "NA"]:
            row[f"{lvl}(raw)"] = d["jlpt"][lvl]; row[f"{lvl}%"] = round(d["jlpt"][lvl]/tv*100, 1) if tv > 0 else 0
        for i in range(1, 6):
            lv = f"TOP-{i}000"; row[f"{lv}(raw)"] = d["rout"][lv]; row[f"{lv}%"] = round(d["rout"][lv]/tv*100, 1) if tv > 0 else 0
        res_gen.append(row)
        p_r = {"File": item['name']}
        for k, v in d["pos"].items(): p_r[f"{k} %"] = round(v/len(d["all_tokens"])*100, 2) if d["all_tokens"] else 0
        res_pos.append(p_r)

    df_gen = pd.DataFrame(res_gen)
    jg_cols = ["MMS", "LD", "VPS", "MPN"]
    for c in jg_cols: df_gen[f"z_{c}"] = zscore(df_gen[c]) if df_gen[c].std() != 0 else 0
    df_gen["JGRI"] = df_gen[[f"z_{c}" for c in jg_cols]].mean(axis=1).round(3)
    df_gen["JGRI Interp"] = df_gen["JGRI"].apply(get_jgri_interp)

    t1, t2, t3 = st.tabs(["📊 Matrix", "📈 Charts", "🔍 Search"])
    with t1:
        cols = ["File", "Tokens", "MTLD", "Readability", "J-Level", "JGRI", "JGRI Interp", "WPS (a)", "Kango% (b)", "Wago% (c)", "V% (d)", "P% (e)", "K%", "H%", "T%"]
        st.dataframe(df_gen[cols], column_config={k: st.column_config.NumberColumn(k, help=v) for k, v in TOOLTIPS.items()}, use_container_width=True)

    with t2:
        st.header("📈 Visualizations & Metrics")
        
        # 1. Linguistic Origin Chart (Variable b & c)
        st.subheader("Linguistic Origin Distribution")
        df_etym = df_gen.melt(id_vars=["File"], value_vars=["Kango% (b)", "Wago% (c)"])
        fig_origin = px.bar(df_etym, x="File", y="value", color="variable", barmode="group", 
                            title="Sino-Japanese (Kango) vs. Native (Wago)")
        st.plotly_chart(fig_origin, use_container_width=True)
        add_html_download_button(fig_origin, "Kango_Wago_Dist")
        
        # 2. Main Metrics Bar Charts (Tokens, TTR, MTLD, Readability, JGRI)
        metric_list = [
            ("Tokens", "Total Tokens Count"),
            ("TTR", "Type-Token Ratio (Lexical Variety)"),
            ("MTLD", "Measure of Textual Lexical Diversity (MTLD)"),
            ("Readability", "jReadability Score (Higher = Easier)"),
            ("JGRI", "Syntactic Complexity (JGRI)")
        ]
        
        for col_name, title_name in metric_list:
            fig = px.bar(df_gen, x="File", y=col_name, title=title_name, template="plotly_white")
            st.plotly_chart(fig, use_container_width=True)
            add_html_download_button(fig, col_name)

        # 3. Stacked Proficiency Charts (JLPT & Routledge)
        st.divider()
        st.subheader("Proficiency & Frequency Profiling")
        
        # JLPT Stacked Chart
        df_jlpt = df_gen.melt(id_vars=["File"], value_vars=["N1%", "N2%", "N3%", "N4%", "N5%", "NA%"])
        fig_jlpt = px.bar(df_jlpt, x="File", y="value", color="variable", title="JLPT Distribution (%)", barmode="stack")
        st.plotly_chart(fig_jlpt, use_container_width=True)
        add_html_download_button(fig_jlpt, "JLPT_Distribution")
        
        # Routledge Stacked Chart
        df_rout = df_gen.melt(id_vars=["File"], value_vars=[f"TOP-{i}000%" for i in range(1, 6)] + ["TOP-NA%"])
        fig_rout = px.bar(df_rout, x="File", y="value", color="variable", title="Routledge Frequency Profiling (%)", barmode="stack")
        st.plotly_chart(fig_rout, use_container_width=True)
        add_html_download_button(fig_rout, "Routledge_Profiling")

        # 4. Orthographic Script Distribution
        st.divider()
        st.subheader("Orthographic Distribution")
        df_script = df_gen.melt(id_vars=["File"], value_vars=["K%", "H%", "T%"])
        fig_script = px.bar(df_script, x="File", y="value", color="variable", title="Script Distribution (%) (Kanji/Hira/Kata)", barmode="stack")
        st.plotly_chart(fig_script, use_container_width=True)
        add_html_download_button(fig_script, "Script_Distribution")

        # 5. Word Cloud with PNG Download
        st.divider()
        st.subheader("Content Word Cloud")
        cloud_toks = [t['surface'] for t in glob_toks if t['pos'] in ["名詞", "動詞", "形容詞"]]
        if cloud_toks and os.path.exists("NotoSansJP[wght].ttf"):
            wc = WordCloud(font_path="NotoSansJP[wght].ttf", background_color="white", width=800, height=400).generate(" ".join(cloud_toks))
            fig_wc, ax = plt.subplots(figsize=(10, 5))
            ax.imshow(wc, interpolation='bilinear')
            ax.axis("off")
            st.pyplot(fig_wc)
            
            # Image download logic
            img_buf = io.BytesIO()
            plt.savefig(img_buf, format='png', bbox_inches='tight')
            st.download_button(label="📥 Download Wordcloud (PNG)", data=img_buf.getvalue(), file_name="wordcloud.png", mime="image/png")

    with t3:
        filtered = [t for t in glob_toks if t['pos'] != "補助記号"]
        matches, concordance = [], []
        for j in range(len(filtered) - n_s + 1):
            win, match = filtered[j : j + n_s], True
            for idx in range(n_s):
                wp, tp = p_w[idx].strip(), p_t[idx]
                target = tp.split(" ")[-1].strip("()") if "(" in tp else tp
                wm = (wp == "*") or (re.search("^"+wp.replace("*", ".*")+"$", win[idx]['surface']) or re.search("^"+wp.replace("*", ".*")+"$", win[idx]['lemma']))
                pm = (tp == "Any (*)") or (target in win[idx]['pos'])
                if not (wm and pm): match = False; break
            if match:
                matches.append((" ".join([t['surface'] for t in win]), " + ".join([t['pos'] for t in win])))
                concordance.append({"File": win[0]['file'], "Left": "".join([t['surface'] for t in filtered[max(0, j-l_ctx):j]]), "KWIC": "".join([t['surface'] for t in win]), "Right": "".join([t['surface'] for t in filtered[j+n_s:j+n_s+r_ctx]])})
        if matches:
            c1, c2 = st.columns(2)
            df_m = pd.DataFrame([{"Seq": k[0], "POS": k[1], "Freq": v} for k, v in Counter(matches).most_common()])
            df_m['PMW'] = round(df_m['Freq'] / len(filtered) * 1000000, 2)
            c1.dataframe(df_m, use_container_width=True)
            c2.dataframe(pd.DataFrame(concordance), use_container_width=True)
else: st.info("Please upload files.")
