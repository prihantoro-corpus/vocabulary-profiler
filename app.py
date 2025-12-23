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

# URL configuration
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

TOOLTIPS = {
    "Tokens": "Total valid linguistic tokens (excluding punctuation).",
    "TTR": "Unique Words / Total Words (Lexical Variety).",
    "MTLD": "Measure of Textual Lexical Diversity (length-independent). Higher = more diverse.",
    "Readability": "JReadability (Lee & Hasebe). Lower score means higher difficulty.",
    "J-Level": "Pedagogical level assigned based on JReadability score.",
    "JGRI": "Japanese Grammatical Relationship Index. Standardized complexity score.",
    "JGRI Interp": "Interpretation: < -0.5: Simple; -0.5 to 0.5: Standard; > 0.5: Complex.",
    "WPS": "Mean words per sentence. Syntactic complexity indicator.",
    "K(raw)": "Count of Kanji script tokens.",
    "K%": "Percentage of Kanji characters.",
    "H(raw)": "Count of Hiragana script tokens.",
    "H%": "Percentage of Hiragana characters.",
    "T(raw)": "Count of Katakana script tokens.",
    "T%": "Percentage of Katakana characters.",
    "O(raw)": "Count of Other script tokens.",
    "O%": "Percentage of Other characters.",
    "V(raw)": "Raw count of Verbs.",
    "V%": "Verb density percentage.",
    "P(raw)": "Raw count of Particles.",
    "P%": "Particle density percentage.",
    "Kango(raw)": "Count of Sino-Japanese words (漢語).",
    "Kango%": "Percentage of Sino-Japanese words.",
    "Wago(raw)": "Count of Native Japanese words (和語).",
    "Wago%": "Percentage of Native Japanese words.",
    "Gairai(raw)": "Count of Foreign Loanwords (外来語).",
    "Gairai%": "Percentage of Foreign Loanwords.",
    "Konshu(raw)": "Count of words with mixed origins (混種語).",
    "Konshu%": "Percentage of mixed origin words."
}

# Add JLPT and Routledge Tooltips
for l in ["N1","N2","N3","N4","N5","NA"]:
    TOOLTIPS[f"{l}(raw)"] = f"Count of words matching JLPT {l} level."
    TOOLTIPS[f"{l}%"] = f"Percentage of text in JLPT {l} level."
for i in range(1, 6):
    TOOLTIPS[f"TOP-{i}000(raw)"] = f"Count of words in Routledge Top {i}000 list."
    TOOLTIPS[f"TOP-{i}000%"] = f"Percentage of text in Routledge Top {i}000 list."
TOOLTIPS["TOP-NA(raw)"] = "Words not found in Routledge Top 5000 list."
TOOLTIPS["TOP-NA%"] = "Percentage of words outside Top 5000."

# ===============================================
# --- 2. UTILITY & LINGUISTIC FUNCTIONS ---
# ===============================================

def add_html_download_button(fig, filename):
    buffer = io.StringIO()
    fig.write_html(buffer, include_plotlyjs='cdn')
    html_bytes = buffer.getvalue().encode()
    st.download_button(
        label=f"📥 Download Chart (HTML)",
        data=html_bytes,
        file_name=f"{filename}.html",
        mime="text/html"
    )

@st.cache_data
def load_jlpt_wordlists():
    wordlists = {}
    for lvl, f in JLPT_FILES.items():
        try:
            df = pd.read_csv(GITHUB_BASE + f)
            wordlists[lvl] = set(df.iloc[:, 0].astype(str).tolist())
        except:
            wordlists[lvl] = set()
    return wordlists

@st.cache_data
def load_routledge_wordlist():
    df = None
    if os.path.exists(LOCAL_ROUTLEDGE_CSV):
        try:
            df = pd.read_csv(LOCAL_ROUTLEDGE_CSV, encoding='utf-8-sig')
        except:
            pass
    if df is None:
        try:
            resp = requests.get(ROUTLEDGE_URL)
            if resp.status_code == 200:
                df = pd.read_excel(io.BytesIO(resp.content))
        except:
            pass
    if df is None:
        try:
            csv_url = GITHUB_BASE + "Routledge%205000%20Vocab%20ONLY.xlsx%20-%20Sheet1.csv"
            df = pd.read_csv(csv_url)
        except:
            return {}

    rout_map = {}
    if df is not None:
        df.columns = [str(c).lower().strip() for c in df.columns]
        if 'rank' in df.columns:
            df['rank'] = pd.to_numeric(df['rank'], errors='coerce')
            df = df.sort_values(by='rank', ascending=True)
        
        for _, row in df.iterrows():
            level = str(row.get('level', '')).strip().upper()
            if not level: continue
            
            forms = []
            for col in ['hiragana', 'katakana', 'kanji']:
                val = str(row.get(col, '')).strip()
                if val and val.lower() != 'nan' and val != '':
                    forms.append(val)
            
            if not forms:
                try:
                    for i in [2, 3, 4]:
                        val = str(row.iloc[i]).strip()
                        if val and val.lower() != 'nan' and val != '':
                            forms.append(val)
                except: pass
            
            for f in forms:
                if f not in rout_map:
                    rout_map[f] = level
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
    
    # 1. MORPHOLOGICAL ANALYSIS & FEATURE EXTRACTION
    for n in nodes:
        if n.surface:
            f = n.feature
            # Safe extraction of Lemma and Reading
            lemma_orth = f[7] if len(f) >= 8 and f[7] != '*' else n.surface
            reading_kata = f[10] if len(f) >= 11 and f[10] != '*' else ""
            
            # BULLETPROOF GOSHU (ORIGIN) EXTRACTION
            goshu = ""
            if len(f) > 12 and f[12] in ["和", "漢", "外", "混"]:
                goshu = f[12]
            elif len(f) > 7:
                for field in f:
                    if field in ["和", "漢", "外", "混"]:
                        goshu = field
                        break

            all_nodes.append({
                "surface": n.surface,
                "lemma": lemma_orth,
                "reading": reading_kata,
                "pos": f[0],
                "goshu": goshu,
                "file": filename
            })
    
    # PREPARING SUBSETS
    valid_nodes = [n for n in all_nodes if n['pos'] != "補助記号"]
    sentences = [s for s in re.split(r'[。！？\n]', text.strip()) if s.strip()]
    num_sentences = len(sentences) if sentences else 1
    total_tokens_valid = len(valid_nodes)
    
    # 2. SCRIPT & ORIGIN COUNTS
    k_raw, h_raw, t_raw, o_raw = 0, 0, 0, 0
    kango_raw, wago_raw, gairaigo_raw, konshugo_raw = 0, 0, 0, 0
    
    for n in valid_nodes:
        # Script Analysis
        if re.search(r'[\u4e00-\u9faf]', n['surface']): k_raw += 1
        elif re.search(r'[\u3040-\u309f]', n['surface']): h_raw += 1
        elif re.search(r'[\u30a0-\u30ff]', n['surface']): t_raw += 1
        else: o_raw += 1
            
        # Origin Analysis
        current_goshu = n.get('goshu', "")
        if current_goshu == "漢": 
            kango_raw += 1
        elif current_goshu == "和": 
            wago_raw += 1
        elif current_goshu == "外":  # Foreign/Loanwords
            gairaigo_raw += 1
        elif current_goshu == "混":  # Mixed origins
            konshugo_raw += 1

    # 3. POS COUNTS (For JGRI and Matrix)
    pos_counts_raw = {k: sum(1 for n in all_nodes if n['pos'] == v) for k, v in POS_FULL_MAP.items()}
    v_raw = pos_counts_raw.get("Verb (動詞)", 0)
    p_raw = pos_counts_raw.get("Particle (助詞)", 0)
    
    # 4. PROFILING (JLPT & ROUTLEDGE)
    jlpt_counts = {lvl: 0 for lvl in ["N1", "N2", "N3", "N4", "N5", "NA"]}
    rout_counts = {f"TOP-{i}000": 0 for i in range(1, 6)}
    rout_counts["TOP-NA"] = 0

    for n in valid_nodes:
        # JLPT Match
        found_jlpt = False
        for lvl in ["N1", "N2", "N3", "N4", "N5"]:
            if n['lemma'] in jlpt_lists[lvl] or n['surface'] in jlpt_lists[lvl]:
                jlpt_counts[lvl] += 1
                found_jlpt = True
                break
        if not found_jlpt: jlpt_counts["NA"] += 1

        # Routledge Match
        found_rout = False
        r_hira = katakana_to_hiragana(n['reading'])
        for check in [n['surface'], n['lemma'], r_hira, n['reading']]:
            if check and check in routledge_list:
                lvl = routledge_list[check]
                if lvl in rout_counts:
                    rout_counts[lvl] += 1
                    found_rout = True
                    break
        if not found_rout: rout_counts["TOP-NA"] += 1

    # 5. FINAL CALCULATIONS
    wps = total_tokens_valid / num_sentences
    pk = (k_raw / total_tokens_valid * 100) if total_tokens_valid > 0 else 0
    ph = (h_raw / total_tokens_valid * 100) if total_tokens_valid > 0 else 0
    pt = (t_raw / total_tokens_valid * 100) if total_tokens_valid > 0 else 0
    po = (o_raw / total_tokens_valid * 100) if total_tokens_valid > 0 else 0

    pkango = (kango_raw / total_tokens_valid * 100) if total_tokens_valid > 0 else 0
    pwago = (wago_raw / total_tokens_valid * 100) if total_tokens_valid > 0 else 0
    pv = (v_raw / total_tokens_valid * 100) if total_tokens_valid > 0 else 0
    pp = (p_raw / total_tokens_valid * 100) if total_tokens_valid > 0 else 0
    
    # JReadability Formula (Stays academically accurate)
    jread = (11.724 + (wps * -0.056) + (pkango * -0.126) + (pwago * -0.042) + (pv * -0.145) + (pp * -0.044)) if total_tokens_valid > 0 else 0
    
    pgairai = (gairaigo_raw / total_tokens_valid * 100) if total_tokens_valid > 0 else 0
    pkonshu = (konshugo_raw / total_tokens_valid * 100) if total_tokens_valid > 0 else 0
    
    # NEW: Calculate 'Other' to fill the remaining percentage gap
    p_other_origin = 100 - (pkango + pwago + pgairai + pkonshu)

    content_words = sum(1 for n in valid_nodes if n['pos'] in ["名詞", "動詞", "形容詞", "副詞", "形状詞"])

    return {
        "all_tokens": all_nodes,
        "stats": {
            "T_Valid": total_tokens_valid, "T_All": len(all_nodes), "WPS": round(wps, 2),
            "Read": round(jread, 3), "K_raw": k_raw, "K%": round(pk, 1),
            "H_raw": h_raw, "H%": round(ph, 1), "T_raw": t_raw, "T%": round(pt, 1),
            "O_raw": o_raw, "O%": round(po, 1), "V_raw": v_raw, "V%": round(pv, 1),
            "P_raw": p_raw, "P%": round(pp, 1),
            "Kango_raw": kango_raw, "Kango%": round(pkango, 1),
            "Wago_raw": wago_raw, "Wago%": round(pwago, 1),
            "Gairai_raw": gairaigo_raw, "Gairai%": round(pgairai, 1),
            "Konshu_raw": konshugo_raw, "Konshu%": round(pkonshu, 1),
            "OriginOther%": round(max(0, p_other_origin), 1)
            
        },
        "jlpt": jlpt_counts, 
        "rout": rout_counts, 
        "pos_raw": pos_counts_raw,
        "jgri_base": {
            "MMS": wps, 
            "LD": content_words / total_tokens_valid if total_tokens_valid > 0 else 0,
            "VPS": v_raw / num_sentences,
            "MPN": pos_counts_raw.get("Adverb (副詞)", 0) / max(pos_counts_raw.get("Noun (名詞)", 1), 1)
        }
    }

# ===============================================
# --- 3. STREAMLIT APPLICATION ---
# ===============================================

st.set_page_config(layout="wide", page_title="Japanese Lexical Profiler")

st.sidebar.title("📚 USER'S MANUAL")
st.sidebar.markdown("[Click here to view the Manual](https://docs.google.com/document/d/1wFPY_b90K0NjS6dQEHJsjJDD_ZRbckq6vzY-kqMT9kE/edit?usp=sharing)")
st.sidebar.divider()

source = st.sidebar.selectbox("📂 Select Data Source", ["Upload Files", "DICO-JALF 30", "DICO-JALF ALL"])
corpus = []
if source == "Upload Files":
    up = st.sidebar.file_uploader("Upload .txt or .xlsx files", accept_multiple_files=True)
    if up:
        for f in up:
            corpus.append({"name": f.name, "text": f.read().decode('utf-8', errors='ignore')})
else:
    url_key = "all" if "ALL" in source else "30%20files%20only"
    url = f"https://raw.githubusercontent.com/prihantoro-corpus/vocabulary-profiler/main/DICO-JALF%20{url_key}.xlsx"
    df_pre = pd.read_excel(io.BytesIO(requests.get(url).content), header=None)
    corpus = [{"name": str(r[0]), "text": str(r[1])} for _, r in df_pre.iterrows()]

if st.sidebar.text_input("Access Password", type="password") != "112233":
    st.info("Please enter the password to unlock analysis.")
    st.stop()

tagger = Tagger()
jlpt_wordlists = load_jlpt_wordlists()
rout_list = load_routledge_wordlist()

if not rout_list: st.sidebar.error("⚠️ Routledge 5000 list not loaded.")
else: st.sidebar.success(f"✅ Routledge list loaded: {len(rout_list)} forms.")

st.title("📖 Japanese Text Vocabulary Profiler")

st.sidebar.divider()
st.sidebar.header("🔍 Advanced Search Settings")
n_size = st.sidebar.number_input("N-Gram Size", 1, 5, 1)
p_words, p_tags = [], []
for i in range(n_size):
    st.sidebar.write(f"**Position {i+1}**")
    c1, c2 = st.sidebar.columns(2)
    p_words.append(c1.text_input("Word/Regex", value="*", key=f"w_{i}"))
    p_tags.append(c2.selectbox("POS Tag", options=["Any (*)"] + list(POS_FULL_MAP.keys()), key=f"t_{i}"))

l_ctx_size = st.sidebar.slider("Left Context", 1, 15, 5)
r_ctx_size = st.sidebar.slider("Right Context", 1, 15, 5)

if corpus:
    res_gen, res_pos, global_toks_all = [], [], []
    for item in corpus:
        data = analyze_text(item['text'], item['name'], tagger, jlpt_wordlists, rout_list)
        global_toks_all.extend(data["all_tokens"])
        t_v = data["stats"]["T_Valid"]
        t_a = data["stats"]["T_All"]
        
        text_for_lex = " ".join([t['surface'] for t in data["all_tokens"] if t['pos'] != "補助記号"])
        lr = LexicalRichness(text_for_lex) if len(text_for_lex.split()) > 10 else None
        
# Extract variables from data['stats'] for cleaner mapping
        s = data["stats"]
        
        row = {
            "File": item['name'], 
            "Tokens": t_v,
            "TTR": round(len(set([t['lemma'] for t in data["all_tokens"] if t['pos'] != "補助記号"]))/t_v, 3) if t_v > 0 else 0,
            "MTLD": round(lr.mtld(), 2) if lr else 0,
            "Readability": s["Read"], 
            "J-Level": get_jread_level(s["Read"]),
            "WPS": s["WPS"], 
            "K(raw)": s["K_raw"], "K%": s["K%"],
            "H(raw)": s["H_raw"], "H%": s["H%"], 
            "T(raw)": s["T_raw"], "T%": s["T%"], 
            "O(raw)": s["O_raw"], "O%": s["O%"],
            "V(raw)": s["V_raw"], "V%": s["V%"], 
            "P(raw)": s["P_raw"], "P%": s["P%"],
            "Kango(raw)": s["Kango_raw"], "Kango%": s["Kango%"],
            "Wago(raw)": s["Wago_raw"], "Wago%": s["Wago%"],
            "Gairai(raw)": s["Gairai_raw"], "Gairai%": s["Gairai%"],
            "Konshu(raw)": s["Konshu_raw"], "Konshu%": s["Konshu%"],
            "OriginOther%": s["OriginOther%"],
            **data["jgri_base"]
        }
        
        for lvl in ["N1", "N2", "N3", "N4", "N5", "NA"]:
            row[f"{lvl}(raw)"] = data["jlpt"][lvl]
            row[f"{lvl}%"] = round((data["jlpt"][lvl]/t_v*100), 1) if t_v > 0 else 0
            
        for i in range(1, 6):
            l = f"TOP-{i}000"
            row[f"{l}(raw)"] = data["rout"][l]
            row[f"{l}%"] = round((data["rout"][l]/t_v*100), 1) if t_v > 0 else 0
        row["TOP-NA(raw)"] = data["rout"]["TOP-NA"]
        row["TOP-NA%"] = round((data["rout"]["TOP-NA"]/t_v*100), 1) if t_v > 0 else 0
        
        res_gen.append(row)
        p_row = {"File": item['name']}
        for label, count in data["pos_raw"].items():
            p_row[f"{label} [%]"] = round((count/t_a*100), 2) if t_a > 0 else 0
        res_pos.append(p_row)

    df_gen = pd.DataFrame(res_gen)
    jgri_cols = ["MMS", "LD", "VPS", "MPN"]
    for c in jgri_cols:
        if df_gen[c].std() != 0: df_gen[f"z_{c}"] = zscore(df_gen[c])
        else: df_gen[f"z_{c}"] = 0
            
    df_gen["JGRI"] = df_gen[[f"z_{c}" for c in jgri_cols]].mean(axis=1).round(3)
    df_gen["JGRI Interp"] = df_gen["JGRI"].apply(get_jgri_interp)

    tab_mat, tab_pos = st.tabs(["📊 General Analysis", "📝 Full POS Distribution"])
    
    with tab_mat:
        st.header("Analysis Matrix")
        cols_to_show = ["File", "Tokens", "TTR", "MTLD", "Readability", "J-Level", "JGRI", "JGRI Interp", "WPS",
                        "K(raw)", "K%", "H(raw)", "H%", "T(raw)", "T%", "O(raw)", "O%",
                        "Kango(raw)", "Kango%", "Wago(raw)", "Wago%", # Added these 4
                        "Gairai(raw)", "Gairai%", "Konshu(raw)", "Konshu%",
                        "V(raw)", "V%", "P(raw)", "P%"] + \
                       [f"{l}{s}" for l in ["N1","N2","N3","N4","N5","NA"] for s in ["(raw)", "%"]] + \
                       [f"TOP-{i}000{s}" for i in range(1, 6) for s in ["(raw)", "%"]] + ["TOP-NA(raw)", "TOP-NA%"]
        
        st.dataframe(df_gen[cols_to_show], column_config={k: st.column_config.NumberColumn(k, help=v) for k, v in TOOLTIPS.items()}, use_container_width=True)

        st.divider()
        st.header("🔍 Pattern Search & Concordance (KWIC)")
        filtered_toks = [t for t in global_toks_all if t['pos'] != "補助記号"]
        t_filtered = len(filtered_toks)
        matches_data, concordance_rows = [], []
        
        for j in range(t_filtered - n_size + 1):
            window, match = filtered_toks[j : j + n_size], True
            for idx in range(n_size):
                w_p_in = p_words[idx].strip()
                t_p_in = p_tags[idx]
                target_tag = t_p_in.split(" ")[-1].strip("()") if "(" in t_p_in else t_p_in
                tok_surf, tok_lem, tok_pos = window[idx].get('surface', ""), window[idx].get('lemma', ""), window[idx].get('pos', "")
                w_match = (w_p_in == "*") or (re.search("^"+w_p_in.replace("*", ".*")+"$", tok_surf) or re.search("^"+w_p_in.replace("*", ".*")+"$", tok_lem))
                p_match = (t_p_in == "Any (*)") or (target_tag in tok_pos)
                if not (w_match and p_match):
                    match = False
                    break
            if match:
                matches_data.append((" ".join([t['surface'] for t in window]), " + ".join([t['pos'] for t in window])))
                l_c = "".join([t['surface'] for t in filtered_toks[max(0, j-l_ctx_size) : j]])
                r_c = "".join([t['surface'] for t in filtered_toks[j+n_size : min(t_filtered, j+n_size+r_ctx_size)]])
                concordance_rows.append({"File": window[0]['file'], "Left Context": l_c, "KWIC": "".join([t['surface'] for t in window]), "Right Context": r_c})
        
        if matches_data:
            c1, c2 = st.columns([1, 2])
            with c1:
                st.subheader("N-Gram Frequencies")
                df_counts = pd.DataFrame([{"Sequence": k[0], "POS": k[1], "Raw Freq": v} for k, v in Counter(matches_data).most_common()])
                df_counts['PMW'] = df_counts['Raw Freq'].apply(lambda x: round((x / t_filtered) * 1_000_000, 2))
                st.dataframe(df_counts.head(10), use_container_width=True)
                st.download_button("📥 Download N-Grams", df_counts.to_csv(index=False).encode('utf-8-sig'), "ngrams.csv")
            with c2:
                st.subheader("Concordance (KWIC)")
                df_conc = pd.DataFrame(concordance_rows)
                st.dataframe(df_conc.head(10), use_container_width=True)
                st.download_button("📥 Download Concordance", df_conc.to_csv(index=False).encode('utf-8-sig'), "concordance.csv")

        st.divider()
        st.header("📈 Visualizations")
        cloud_toks = [t['surface'] for t in filtered_toks if t['pos'] in ["名詞", "動詞", "形容詞", "副詞", "形状詞"]]
        if cloud_toks and os.path.exists("NotoSansJP[wght].ttf"):
            wc = WordCloud(font_path="NotoSansJP[wght].ttf", background_color="white", width=800, height=350).generate(" ".join(cloud_toks))
            fig_cloud, ax = plt.subplots(figsize=(10, 4))
            ax.imshow(wc); ax.axis("off"); st.pyplot(fig_cloud)

        v_list = [("Tokens", "Tokens per File"), ("TTR", "Type-Token Ratio"), ("MTLD", "Lexical Diversity (MTLD)"), ("Readability", "JReadability Score"), ("JGRI", "Complexity Score")]
        for col_name, title_name in v_list:
            fig = px.bar(df_gen, x="File", y=col_name, title=title_name, template="plotly_white")
            st.plotly_chart(fig, use_container_width=True); add_html_download_button(fig, col_name)

        # Script Distribution Chart (with Download Button)
        df_s = df_gen.melt(id_vars=["File"], value_vars=["K%", "H%", "T%", "O%"], var_name="Script", value_name="%")
        fig_script = px.bar(df_s, x="File", y="%", color="Script", title="Script Distribution", barmode="stack", template="plotly_white")
        st.plotly_chart(fig_script, use_container_width=True)
        add_html_download_button(fig_script, "script_distribution")
        
# Word Origin Distribution Stacked Chart (Modified for 100% Distribution)
        df_o = df_gen.melt(
            id_vars=["File"], 
            # Include the Other category to reach 100%
            value_vars=["Kango%", "Wago%", "Gairai%", "Konshu%", "OriginOther%"],
            var_name="Origin", 
            value_name="%"
        )
        fig_origin = px.bar(
            df_o, 
            x="File", 
            y="%", 
            color="Origin", 
            title="Word Origin Distribution (100% Stacked)", 
            barmode="stack",
            template="plotly_white",
            color_discrete_map={
                "Kango%": "#EF553B", 
                "Wago%": "#636EFA",
                "Gairai%": "#00CC96",
                "Konshu%": "#AB63FA",
                "OriginOther%": "#E5ECF6" # Light grey for unclassified
            }
        )
        fig_origin = px.bar(
            df_o, 
            x="File", 
            y="%", 
            color="Origin", 
            title="Word Origin Distribution (Kango/Wago/Gairai/Konshu)", 
            barmode="stack",
            template="plotly_white",
            # Explicit colors for all 4 categories
            color_discrete_map={
                "Kango%": "#EF553B", 
                "Wago%": "#636EFA",
                "Gairai%": "#00CC96",
                "Konshu%": "#AB63FA"
            }
        )
        st.plotly_chart(fig_origin, use_container_width=True)
        add_html_download_button(fig_origin, "origin_distribution")
# JLPT Distribution Chart (with Download Button)
        df_j = df_gen.melt(id_vars=["File"], value_vars=["N1%", "N2%", "N3%", "N4%", "N5%", "NA%"], var_name="Level", value_name="%")
        fig_jlpt = px.bar(df_j, x="File", y="%", color="Level", title="JLPT Distribution", barmode="stack", category_orders={"Level": ["N1%", "N2%", "N3%", "N4%", "N5%", "NA%"]}, template="plotly_white")
        st.plotly_chart(fig_jlpt, use_container_width=True)
        add_html_download_button(fig_jlpt, "jlpt_distribution")  
      

    with tab_pos:
        st.header("14-Tier POS Distribution (%)")
        df_pos_viz = pd.DataFrame(res_pos)
        st.dataframe(df_pos_viz, use_container_width=True)

        st.divider()
        st.subheader("POS Distribution Visualization")

        # Melt the dataframe for Plotly: Files as rows, POS tags as color segments
        # We exclude 'File' from the value_vars to plot all POS percentages
        pos_cols = [c for c in df_pos_viz.columns if c != "File"]
        df_pos_melted = df_pos_viz.melt(id_vars=["File"], value_vars=pos_cols, var_name="POS Tag", value_name="%")

        # Create Horizontal Stacked Bar Chart
        # x="%" makes the bars horizontal, y="File" puts filenames on the vertical axis
        fig_pos = px.bar(
            df_pos_melted, 
            y="File", 
            x="%", 
            color="POS Tag", 
            title="14-Tier POS Distribution per File (100% Stacked)",
            orientation='h',
            barmode="stack",
            template="plotly_white",
            height=max(400, len(df_pos_viz) * 50) # Adjust height based on number of files
        )

        # Update layout to ensure the X-axis stays at 100%
        fig_pos.update_layout(xaxis_range=[0, 100], xaxis_title="Percentage (%)", yaxis_title="File Name")

        st.plotly_chart(fig_pos, use_container_width=True)
        
        # Add the HTML Download Button
        add_html_download_button(fig_pos, "pos_distribution_chart")
else:
    st.info("Upload files to begin.")
