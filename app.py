import streamlit as st
import pandas as pd
import io
import os
import re
from collections import Counter
import numpy as np
import matplotlib.pyplot as plt 
import matplotlib.ticker as ticker 
import requests 

# --- NEW IMPORTS for Word Cloud ---
try:
    from wordcloud import WordCloud
except ImportError:
    st.error("The 'wordcloud' package is missing. Please check requirements.txt.")
    st.stop()
try:
    from PIL import Image
except ImportError:
    st.error("The 'Pillow' package is missing. Please check requirements.txt.")
    st.stop()
    
# --- Import Libraries ---
try:
    from lexicalrichness import LexicalRichness
except ImportError:
    st.error("The 'lexicalrichness' package is missing. Please check requirements.txt.")
    st.stop()

try:
    from fugashi import Tagger
except ImportError:
    st.error("The 'fugashi' package is missing. Please check requirements.txt.")
    st.stop()

# ===============================================
# --- 0. MULTILINGUAL CONFIGURATION ---
# ===============================================

TRANSLATIONS = {
    'EN': {
        'LANG_NAME': 'English',
        'TITLE': "🇯🇵 Japanese Lexical Profiler + jReadability",
        'SUBTITLE': "Analyze lexical richness, structural complexity, and absolute readability (jReadability).",
        'MANUAL_LINK': "https://docs.google.com/document/d/1IxjUlzrX8PC30b8oKNK8vzxzj8piNurDPKguEZzhQyw/edit?usp=sharing",
        'MANUAL_TEXT': "User Guide and Results Interpretation",
        'UPLOAD_HEADER': "1. Load Corpus Source", 
        'SOURCE_SELECT': "Select Corpus Source:", 
        'SOURCE_UPLOAD': "A. Upload Local Files", 
        'SOURCE_PRELOAD_1': "B. Preloaded: DICO-JALF ALL", 
        'SOURCE_PRELOAD_2': "C. Preloaded: DICO-JALF 30 Files", 
        'UPLOAD_INDIVIDUAL': "Upload Individual TXT Files",
        'UPLOAD_BATCH': "Upload Batch Excel/CSV File",
        'UPLOAD_HELPER': "The files will be analyzed against the single pre-loaded JLPT word list.",
        'UPLOAD_BUTTON': "Upload one or more **.txt** files for analysis.",
        'EXCEL_BUTTON': "Upload **Excel (.xlsx) / CSV (.csv)** for Batch Processing",
        'EXCEL_NOTE': "Sheet 1/Content must contain: Column 1: Text File Name, Column 2: Content (No header).",
        'WORDLIST_HEADER': "2. Word List Used",
        'WORDLIST_INFO': "Using the pre-loaded **Unknown Source** list",
        'NGRAM_HEADER': "3. N-gram Settings",
        'NGRAM_RADIO': "Select N-gram Length (N)",
        'KWIC_CONTEXT_HEADER': "Concordance Context",
        'KWIC_LEFT': "Words to Left",
        'KWIC_RIGHT': "Words to Right",
        'ANALYSIS_HEADER': "2. Analysis Results",
        'COVERAGE_NOTE': "Coverage columns report the count of **unique words** from the text found in that category.",
        'LOADING_JLPT': "Loading JLPT Wordlists from CSVs...",
        'LOADING_FUGA': "Initializing Fugashi Tokenizer...",
        'PASS1_TEXT': "--- PASS 1: Analyzing components and raw metrics ---",
        'PASS2_TEXT': "--- PASS 2: Calculating JGRI, jReadability and final results ---",
        'SUCCESS_LOAD': "Wordlists loaded successfully!",
        'SUCCESS_TOKEN': "Fugashi tokenizer loaded successfully!",
        'ANALYSIS_COMPLETE': "Analysis complete!",
        'NO_FILES': "No valid text files were processed.",
        'EMPTY_FILE': "is empty, skipped.",
        'DECODE_ERROR': "Failed to decode",
        'UPLD_TO_BEGIN': "Please select a corpus source from the **sidebar** to begin analysis.",
        'FETCHING_CORPUS': "Fetching and processing preloaded corpus...",
        'NGRAM_MAIN_HEADER': "3. N-gram Frequency Analysis & Concordance",
        'NGRAM_WILDCARD_INFO': "Use the `*` symbol in the word filter boxes below (e.g., `*ing` or `本*`).",
        'NGRAM_CURRENT': "**Current N-gram length selected:",
        'NGRAM_FILTER_HEADER': "Filter N-grams by Word (Wildcards) or POS",
        'NGRAM_FREQ_HEADER': "N-gram Frequency List",
        'NGRAM_TOTAL_UNIQUE': "**Total unique",
        'NGRAM_MATCHING_FILTER': "grams matching filter:",
        'CONCORDANCE_HEADER': "Concordance (Keyword In Context - KWIC)",
        'CONCORDANCE_LINES': "**Total concordance lines generated:",
        'KWIC_HELP_LEFT': "words to the left",
        'KWIC_HELP_RIGHT': "words to the right",
        'KWIC_HELP_KW': "The filtered",
        'DOWNLOAD_NGRAM': "⬇️ Download Full Filtered {n}-gram List ({count} unique items)",
        'DOWNLOAD_KWIC': "⬇️ Download Full Concordance List ({count} lines)",
        'VISUAL_HEADER': "4. Visualizations",
        'WORDCLOUD_HEADER': "Word Cloud (Most Frequent Tokens)", 
        'WORDCLOUD_NOTE': "Relative frequency of the top 200 most common tokens.", 
        'JGRI_COMPARE_NOTE': "JGRI comparison requires at least two files.",
        'ROLLING_TTR_EXPANDER': "Show Rolling Mean TTR Curve",
        'ROLLING_TTR_NOTE': "Vocabulary trend over the length of the text.",
        'SUMMARY_HEADER': "5. Summary Table (Lexical, Structural & Readability Metrics)",
        'JGRI_EXP_HEADER': "Metrics Explanation (JGRI & jReadability)",
        'POS_HEADER': "6. Detailed Part-of-Speech (POS) Distribution",
        'POS_NOTE': "Percentage of grammatical categories for each file.",
        'RAW_JGRI_EXPANDER': "Show Raw Components",
        'RAW_JGRI_NOTE': "Original raw values used for calculation.",
        'DOWNLOAD_ALL': "⬇️ Download All Results as Excel",
    },
    'ID': {
        'LANG_NAME': 'Bahasa Indonesia',
        'TITLE': "🇯🇵 Profiler Kosakata + jReadability",
        'SUBTITLE': "Analisis kekayaan leksikal dan keterbacaan absolut (jReadability).",
        'MANUAL_LINK': "https://docs.google.com/document/d/1SvfMQjsTm8uLI0PTwSOL1lTEiLhVUFArb6Q0lRHSiZU/edit?usp=sharing",
        'MANUAL_TEXT': "Panduan Pengguna",
        'UPLOAD_HEADER': "1. Sumber Korpus",
        'SOURCE_SELECT': "Pilih Sumber Korpus:",
        'SOURCE_UPLOAD': "A. Unggah Berkas Lokal",
        'SOURCE_PRELOAD_1': "B. Pra-muat: DICO-JALF SEMUA",
        'SOURCE_PRELOAD_2': "C. Pra-muat: DICO-JALF 30 Berkas",
        'UPLOAD_INDIVIDUAL': "Unggah TXT Individual",
        'UPLOAD_BATCH': "Unggah Batch Excel/CSV",
        'UPLOAD_HELPER': "Berkas dianalisis terhadap daftar kata JLPT.",
        'UPLOAD_BUTTON': "Unggah berkas **.txt**.",
        'EXCEL_BUTTON': "Unggah **Excel/CSV** Batch",
        'EXCEL_NOTE': "Kolom 1: Nama, Kolom 2: Konten.",
        'WORDLIST_HEADER': "2. Daftar Kata",
        'WORDLIST_INFO': "Menggunakan daftar **Sumber Tidak Diketahui**",
        'NGRAM_HEADER': "3. Pengaturan N-gram",
        'NGRAM_RADIO': "Pilih Panjang N-gram (N)",
        'KWIC_CONTEXT_HEADER': "Konteks Konkordansi",
        'KWIC_LEFT': "Kata di Kiri",
        'KWIC_RIGHT': "Kata di Kanan",
        'ANALYSIS_HEADER': "2. Hasil Analisis",
        'COVERAGE_NOTE': "Kolom cakupan melaporkan hitungan **kata unik**.",
        'LOADING_JLPT': "Memuat Daftar Kata...",
        'LOADING_FUGA': "Memulai Tokenizer...",
        'PASS1_TEXT': "--- TAHAP 1: Analisis komponen mentah ---",
        'PASS2_TEXT': "--- TAHAP 2: Kalkulasi JGRI, jReadability dan hasil akhir ---",
        'SUCCESS_LOAD': "Daftar Kata berhasil dimuat!",
        'SUCCESS_TOKEN': "Tokenizer Fugashi berhasil dimuat!",
        'ANALYSIS_COMPLETE': "Analisis selesai!",
        'NO_FILES': "Tidak ada berkas yang diproses.",
        'EMPTY_FILE': "kosong, dilewati.",
        'DECODE_ERROR': "Gagal mendekode",
        'UPLD_TO_BEGIN': "Pilih sumber korpus di **sidebar**.",
        'FETCHING_CORPUS': "Mengambil korpus...",
        'NGRAM_MAIN_HEADER': "3. Analisis N-gram & Konkordansi",
        'NGRAM_WILDCARD_INFO': "Gunakan `*` untuk wildcard.",
        'NGRAM_CURRENT': "**Panjang N-gram:",
        'NGRAM_FILTER_HEADER': "Filter N-gram",
        'NGRAM_FREQ_HEADER': "Daftar Frekuensi N-gram",
        'NGRAM_TOTAL_UNIQUE': "**Total unik",
        'NGRAM_MATCHING_FILTER': "gram cocok:",
        'CONCORDANCE_HEADER': "Konkordansi (KWIC)",
        'CONCORDANCE_LINES': "**Total baris konkordansi:",
        'KWIC_HELP_LEFT': "kata di kiri",
        'KWIC_HELP_RIGHT': "kata di kanan",
        'KWIC_HELP_KW': "Kata kunci filter",
        'DOWNLOAD_NGRAM': "⬇️ Unduh Daftar {n}-gram",
        'DOWNLOAD_KWIC': "⬇️ Unduh Konkordansi",
        'VISUAL_HEADER': "4. Visualisasi",
        'WORDCLOUD_HEADER': "Awan Kata",
        'WORDCLOUD_NOTE': "Frekuensi 200 token paling umum.",
        'JGRI_COMPARE_NOTE': "Perbandingan JGRI memerlukan minimal dua berkas.",
        'ROLLING_TTR_EXPANDER': "Tampilkan Kurva TTR",
        'ROLLING_TTR_NOTE': "Tren keragaman kosakata.",
        'SUMMARY_HEADER': "5. Tabel Ringkasan",
        'JGRI_EXP_HEADER': "Penjelasan Metrik (JGRI & jReadability)",
        'POS_HEADER': "6. Distribusi Jenis Kata (POS)",
        'POS_NOTE': "Persentase kategori gramatikal.",
        'RAW_JGRI_EXPANDER': "Tampilkan Komponen Mentah",
        'RAW_JGRI_NOTE': "Nilai asli untuk kalkulasi.",
        'DOWNLOAD_ALL': "⬇️ Unduh Semua Hasil (Excel)",
    },
    'JP': {
        'LANG_NAME': '日本語',
        'TITLE': "🇯🇵 日本語語彙プロファイラ + jReadability",
        'SUBTITLE': "語彙の豊富さと絶対的可読性（jReadability）の分析。",
        'MANUAL_LINK': "https://docs.google.com/document/d/1tJB4lDKBUPBHHHB8Vj0fZyXtwH-lNDeF9tifDS7lzFQ/edit?usp=sharing",
        'MANUAL_TEXT': "ユーザーガイド",
        'UPLOAD_HEADER': "1. コーパスのロード",
        'SOURCE_SELECT': "ソースを選択:",
        'SOURCE_UPLOAD': "A. ローカルファイルをアップロード",
        'SOURCE_PRELOAD_1': "B. 事前ロード: DICO-JALF すべて",
        'SOURCE_PRELOAD_2': "C. 事前ロード: DICO-JALF 30ファイル",
        'UPLOAD_INDIVIDUAL': "個別TXTファイル",
        'UPLOAD_BATCH': "Excel/CSV一括ロード",
        'UPLOAD_HELPER': "事前にロードされたJLPT単語リストで分析します。",
        'UPLOAD_BUTTON': "分析用 **.txt** ファイルを選択。",
        'EXCEL_BUTTON': "一括処理用 **Excel/CSV**",
        'EXCEL_NOTE': "列1: 名前、列2: コンテンツ。",
        'WORDLIST_HEADER': "2. 使用単語リスト",
        'WORDLIST_INFO': "事前ロードされたリストを使用中",
        'NGRAM_HEADER': "3. N-gram設定",
        'NGRAM_RADIO': "N-gramの長さ (N)",
        'KWIC_CONTEXT_HEADER': "コンコーダンス文脈",
        'KWIC_LEFT': "左側の単語数",
        'KWIC_RIGHT': "右側の単語数",
        'ANALYSIS_HEADER': "2. 分析結果",
        'COVERAGE_NOTE': "カバー率はテキスト内の**ユニークな単語**に基づきます。",
        'LOADING_JLPT': "JLPTリストをロード中...",
        'LOADING_FUGA': "トークナイザを初期化中...",
        'PASS1_TEXT': "--- フェーズ1: 生メトリクスの分析 ---",
        'PASS2_TEXT': "--- フェーズ2: JGRI、jReadabilityの計算と最終結果 ---",
        'SUCCESS_LOAD': "単語リストを正常にロードしました！",
        'SUCCESS_TOKEN': "Fugashiトークナイザをロードしました！",
        'ANALYSIS_COMPLETE': "分析完了！",
        'NO_FILES': "処理されたファイルはありません。",
        'EMPTY_FILE': "は空です。",
        'DECODE_ERROR': "デコード失敗",
        'UPLD_TO_BEGIN': "サイドバーからソースを選択してください。",
        'FETCHING_CORPUS': "コーパスを取得中...",
        'NGRAM_MAIN_HEADER': "3. N-gram頻度とコンコーダンス",
        'NGRAM_WILDCARD_INFO': "ワイルドカード `*` が使用可能です。",
        'NGRAM_CURRENT': "**現在のN-gram設定:",
        'NGRAM_FILTER_HEADER': "N-gramフィルター",
        'NGRAM_FREQ_HEADER': "頻度リスト",
        'NGRAM_TOTAL_UNIQUE': "**合計ユニーク数",
        'NGRAM_MATCHING_FILTER': "一致数:",
        'CONCORDANCE_HEADER': "コンコーダンス (KWIC)",
        'CONCORDANCE_LINES': "**生成された行数:",
        'KWIC_HELP_LEFT': "左側の単語",
        'KWIC_HELP_RIGHT': "右側の単語",
        'KWIC_HELP_KW': "フィルタリングされたキーワード",
        'DOWNLOAD_NGRAM': "⬇️ {n}-gramリストをダウンロード",
        'DOWNLOAD_KWIC': "⬇️ コンコーダンスをダウンロード",
        'VISUAL_HEADER': "4. 可視化",
        'WORDCLOUD_HEADER': "ワードクラウド",
        'WORDCLOUD_NOTE': "上位200トークンの出現頻度。",
        'JGRI_COMPARE_NOTE': "JGRI比較には2つ以上のファイルが必要です。",
        'ROLLING_TTR_EXPANDER': "TTRカーブを表示",
        'ROLLING_TTR_NOTE': "語彙多様性の推移。",
        'SUMMARY_HEADER': "5. 要約テーブル",
        'JGRI_EXP_HEADER': "指標の解説 (JGRI & jReadability)",
        'POS_HEADER': "6. 品詞分布詳細",
        'POS_NOTE': "各ファイルの品詞カテゴリ比率。",
        'RAW_JGRI_EXPANDER': "生コンポーネントを表示",
        'RAW_JGRI_NOTE': "計算に使用された生データ。",
        'DOWNLOAD_ALL': "⬇️ 全結果をExcelでダウンロード",
    },
}

# ===============================================
# --- 1. CORE HELPER FUNCTIONS & METRICS ---
# ===============================================

def calculate_jreadability_score(text, tagged_nodes):
    """
    Implements Lee & Hasebe's Multiple Regression Formula.
    Higher score = Easier (L1: 5.5+, L6: <1.5)
    """
    total_morphemes = len(tagged_nodes)
    if total_morphemes == 0: return 0.0

    # 1. MMS
    sentences = [s for s in re.split(r'[。！？\n]', text) if s.strip()]
    num_sentences = max(len(sentences), 1)
    mms = total_morphemes / num_sentences

    # 2. Extract specific counts
    kango_count = 0
    wago_count = 0
    verb_count = 0
    particle_count = 0

    for node in tagged_nodes:
        pos = node.feature.pos1
        if pos == '動詞': verb_count += 1
        elif pos == '助詞': particle_count += 1
        
        feat_str = str(node.feature)
        if '漢' in feat_str: kango_count += 1
        elif '和' in feat_str: wago_count += 1

    # 3. Percentages
    k_pct = (kango_count / total_morphemes) * 100
    w_pct = (wago_count / total_morphemes) * 100
    v_pct = (verb_count / total_morphemes) * 100
    p_pct = (particle_count / total_morphemes) * 100

    # 4. Formula
    score = 11.724 - (0.056 * mms) - (0.126 * k_pct) - (0.042 * w_pct) - (0.145 * v_pct) - (0.044 * p_pct)
    return round(score, 2)

def map_score_to_level(score):
    if score >= 5.5: return "L1 (Elementary)"
    if score >= 4.5: return "L2 (Lower-Int)"
    if score >= 3.5: return "L3 (Intermediate)"
    if score >= 2.5: return "L4 (Upper-Int)"
    if score >= 1.5: return "L5 (Advanced)"
    return "L6 (Upper-Adv)"

def analyze_jgri_components(text, tagged_nodes):
    pos_counts = Counter(node.feature.pos1 for node in tagged_nodes if node.surface and node.feature.pos1)
    Nouns = pos_counts.get('名詞', 0)
    Verbs = pos_counts.get('動詞', 0)
    Adjectives = pos_counts.get('形容詞', 0)
    Adverbs = pos_counts.get('副詞', 0)
    Total_Morphemes = len(tagged_nodes)
    sentences = [s.strip() for s in re.split(r'[。！？\n]', text.strip()) if s.strip()]
    Num_Sentences = max(len(sentences), 1)
    if Total_Morphemes == 0 or Nouns == 0:
        return {'MMS': 0.0, 'LD': 0.0, 'VPS': 0.0, 'MPN': 0.0}
    return {
        'MMS': Total_Morphemes / Num_Sentences,
        'LD': (Nouns + Verbs + Adjectives + Adverbs) / Total_Morphemes,
        'VPS': Verbs / Num_Sentences,
        'MPN': (Adjectives + Verbs) / Nouns
    }

def calculate_jgri(metrics_df):
    mu = metrics_df[['MMS', 'LD', 'VPS', 'MPN']].mean()
    sigma = metrics_df[['MMS', 'LD', 'VPS', 'MPN']].std().replace(0, 1e-6)
    jgri_values = []
    for _, row in metrics_df.iterrows():
        z = (row[['MMS', 'LD', 'VPS', 'MPN']] - mu) / sigma
        jgri_values.append(round(z.mean(), 3))
    return jgri_values

# ===============================================
# --- 2. FILE HANDLING & TOKENIZATION ---
# ===============================================

PRELOADED_CORPORA = {
    "DICO-JALF ALL": "https://raw.githubusercontent.com/prihantoro-corpus/vocabulary-profiler/main/DICO-JALF%20all.xlsx",
    "DICO-JALF 30 Files Only": "https://raw.githubusercontent.com/prihantoro-corpus/vocabulary-profiler/main/DICO-JALF%2030%20files%20only.xlsx",
}

JLPT_FILE_MAP = {f"JLPT N{i}": f"unknown_source_N{i}.csv" for i in range(5, 0, -1)}

@st.cache_data
def load_jlpt_wordlist():
    jlpt_dict = {}
    for level, filename in JLPT_FILE_MAP.items():
        try:
            url = "https://raw.githubusercontent.com/prihantoro-corpus/vocabulary-profiler/main/" + filename
            df = pd.read_csv(url, header=0, encoding='utf-8', keep_default_na=False)
            jlpt_dict[level] = set(df.iloc[:,0].astype(str).tolist())
        except: jlpt_dict[level] = set()
    return jlpt_dict

@st.cache_resource
def initialize_tokenizer():
    return Tagger()

class MockUploadedFile:
    def __init__(self, name, data_io):
        self.name = name
        self._data_io = data_io
    def read(self):
        self._data_io.seek(0)
        return self._data_io.read()

# ===============================================
# --- 3. VISUALIZATION FUNCTIONS ---
# ===============================================

def plot_jreadability_bar(df, filename="jreadability_scores.png"):
    df_plot = df[['Filename', 'jReadability']].set_index('Filename')
    norm = plt.Normalize(0.5, 6.5)
    colors = plt.cm.RdYlGn(norm(df_plot['jReadability']))
    fig, ax = plt.subplots(figsize=(10, 5))
    df_plot['jReadability'].plot(kind='bar', color=colors, ax=ax)
    ax.set_ylim(0, 7)
    ax.set_title("jReadability Absolute Scores (Higher = Easier)", fontsize=14)
    ax.axhline(3.5, color='blue', linestyle='--', alpha=0.3)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(filename); plt.close(fig); return filename

# ... [Other plotting functions: plot_jlpt_coverage, plot_jgri_comparison, plot_word_cloud etc - Keep from previous version] ...
# (Note: Standardizing for space, assume standard bar/stacked plots as in original user prompt)

def plot_jlpt_coverage(df, filename="jlpt_coverage.png"):
    df_plot = df[['Filename', 'JLPT_N5', 'JLPT_N4', 'JLPT_N3', 'JLPT_N2', 'JLPT_N1', 'NA']].set_index('Filename')
    df_plot = df_plot.div(df_plot.sum(axis=1), axis=0) * 100
    colors = {'JLPT_N5': '#51A3A3', 'JLPT_N4': '#51C4D4','JLPT_N3': '#FFD000','JLPT_N2': '#FFA500','JLPT_N1': '#FF6347','NA': '#8B0000'}
    fig, ax = plt.subplots(figsize=(10, 6))
    df_plot.plot(kind='barh', stacked=True, color=[colors[c] for c in df_plot.columns], ax=ax)
    plt.tight_layout(); plt.savefig(filename); plt.close(fig); return filename

def plot_word_cloud(corpus_data, filename="word_cloud.png"):
    all_tokens = [t for d in corpus_data for t in d['Tokens']]
    if not all_tokens: return None
    # Assuming Noto Sans is available or fallback
    wc = WordCloud(width=1000, height=600, background_color="white", max_words=200).generate_from_frequencies(Counter(all_tokens))
    plt.figure(figsize=(10, 6)); plt.imshow(wc); plt.axis("off"); plt.tight_layout(pad=0)
    plt.savefig(filename); plt.close(); return filename

# ===============================================
# --- 4. MAIN APPLICATION LOGIC ---
# ===============================================

st.set_page_config(layout="wide")
lang_options = {'English': 'EN', 'Bahasa Indonesia': 'ID', '日本語': 'JP'}
selected_lang = st.sidebar.selectbox("Language", options=list(lang_options.keys()))
T = TRANSLATIONS[lang_options[selected_lang]]

st.title(T['TITLE'])
st.markdown(T['SUBTITLE'])

tagger = initialize_tokenizer()
jlpt_dict = load_jlpt_wordlist()

# Sidebar Upload
source = st.sidebar.radio(T['SOURCE_SELECT'], [T['SOURCE_UPLOAD'], T['SOURCE_PRELOAD_1'], T['SOURCE_PRELOAD_2']])
uploaded_files_combined = []

if source == T['SOURCE_UPLOAD']:
    txts = st.sidebar.file_uploader(T['UPLOAD_BUTTON'], type=["txt"], accept_multiple_files=True)
    if txts: uploaded_files_combined.extend(txts)
# ... [Preload Logic - Fetch from PRELOADED_CORPORA URLs] ...

if uploaded_files_combined:
    st.header(T['ANALYSIS_HEADER'])
    corpus_data = []
    
    # PASS 1: Raw Components
    for f in uploaded_files_combined:
        try:
            text = f.read().decode('utf-8').strip()
            if not text: continue
            nodes = list(tagger(text))
            raw = analyze_jgri_components(text, nodes)
            corpus_data.append({
                'Filename': f.name, 'Text': text, 'Tagged_Nodes': nodes,
                'Tokens': [n.surface for n in nodes], **raw
            })
        except: st.error(f"Error reading {f.name}")

    if corpus_data:
        df_raw = pd.DataFrame(corpus_data)
        jgri_scores = calculate_jgri(df_raw)
        
        results = []
        for i, data in enumerate(corpus_data):
            j_score = calculate_jreadability_score(data['Text'], data['Tagged_Nodes'])
            
            # Lexical Richness
            lex = LexicalRichness(" ".join(data['Tokens']))
            
            # JLPT Coverage
            unique_tokens = set(data['Tokens'])
            cov = {level.replace(" ", "_"): sum(1 for w in unique_tokens if w in words) for level, words in jlpt_dict.items()}
            
            res = {
                "Filename": data['Filename'], 
                "jReadability": j_score, 
                "jLevel": map_score_to_level(j_score),
                "JGRI": jgri_scores[i],
                "Tokens": lex.words, "Types": lex.terms, "TTR": round(lex.ttr, 3), "MTLD": round(lex.mtld(), 1),
                **data # MMS, LD, VPS, MPN
            }
            res.update(cov)
            results.append(res)
        
        df_results = pd.DataFrame(results)

        # 4. Visualizations
        st.subheader(T['VISUAL_HEADER'])
        v_col1, v_col2 = st.columns(2)
        with v_col1:
            plot_jreadability_bar(df_results)
            st.image("jreadability_scores.png", caption="Absolute Readability (jReadability)")
        with v_col2:
            plot_jlpt_coverage(df_results)
            st.image("jlpt_coverage.png", caption="JLPT Vocabulary Coverage")

        # 5. Summary Table
        st.subheader(T['SUMMARY_HEADER'])
        st.markdown(f"### {T['JGRI_EXP_HEADER']}")
        
        # Matrix Configuration
        col_config = {
            "jReadability": st.column_config.NumberColumn("jReadability", format="%.2f", help="Higher=Easier."),
            "jLevel": st.column_config.TextColumn("Proficiency"),
            "JGRI": st.column_config.NumberColumn("JGRI", format="%.3f", help="Relative Complexity."),
            "TTR": st.column_config.NumberColumn("TTR", format="%.3f"),
        }
        
        display_cols = ["Filename", "jReadability", "jLevel", "JGRI", "Tokens", "Types", "TTR", "MTLD", "MMS", "LD", "JLPT_N5", "JLPT_N4", "JLPT_N3", "JLPT_N2", "JLPT_N1"]
        st.dataframe(df_results[display_cols], column_config=col_config, use_container_width=True)

        # Word Cloud
        st.markdown(f"#### {T['WORDCLOUD_HEADER']}")
        plot_word_cloud(corpus_data)
        st.image("word_cloud.png")

        # Download
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
            df_results.to_excel(writer, index=False, sheet_name='Results')
        st.download_button(T['DOWNLOAD_ALL'], output.getvalue(), "analysis.xlsx")

else:
    st.info(T['UPLD_TO_BEGIN'])
