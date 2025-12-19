import streamlit as st
import pandas as pd
import io
import os
import re
from collections import Counter
import numpy as np
import matplotlib.pyplot as plt 
import matplotlib.ticker as ticker 
import requests # Need requests to fetch external files

# NEW IMPORTS for Word Cloud
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
    
# --- Import Libraries (Assuming they are in requirements.txt) ---
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
# --- JREADABILITY (Japanese Readability Formula)
# ===============================================

def analyze_jreadability(text, tagged_nodes):
    """Computes JReadability score and components."""
    sentences = re.split(r'[。！？\n]', text.strip())
    sentences = [s for s in sentences if s.strip()]
    if not sentences:
        return {"a": 0, "b": 0, "c": 0, "d": 0, "e": 0, "JREAD": None}

    total_chars = sum(len(s) for s in sentences)
    a = total_chars / len(sentences)

    valid_nodes = [n for n in tagged_nodes if n.surface]
    total_tokens = len(valid_nodes)
    if total_tokens == 0:
        return {"a": round(a,2), "b": 0, "c": 0, "d": 0, "e": 0, "JREAD": None}

    verbs = sum(1 for n in valid_nodes if n.feature.pos1 == "動詞")
    particles = sum(1 for n in valid_nodes if n.feature.pos1 == "助詞")

    kango = sum(1 for n in valid_nodes if re.fullmatch(r"[\u4E00-\u9FFF]+", n.surface))
    wago = sum(1 for n in valid_nodes if re.fullmatch(r"[\u3040-\u309F]+", n.surface))

    b = (kango / total_tokens) * 100
    c = (wago / total_tokens) * 100
    d = (verbs / total_tokens) * 100
    e = (particles / total_tokens) * 100

    X = 11.724 - (0.056 * a) - (0.126 * b) - (0.042 * c) - (0.145 * d) - (0.044 * e)

    return {
        "a": round(a, 2),
        "b": round(b, 2),
        "c": round(c, 2),
        "d": round(d, 2),
        "e": round(e, 2),
        "JREAD": round(X, 3)
    }


# ===============================================
# --- 0. MULTILINGUAL CONFIGURATION (UPDATED) ---
# ===============================================

TRANSLATIONS = {
    'EN': {
        'LANG_NAME': 'English',
        'TITLE': "🇯🇵 Japanese Lexical Profiler",
        'SUBTITLE': "Analyze lexical richness, structural complexity, and JLPT word coverage.",
        'MANUAL_LINK': "https://docs.google.com/document/d/1IxjUlzrX8PC30b8oKNK8vzxzj8piNurDPKguEZzhQyw/edit?usp=sharing",
        'MANUAL_TEXT': "User Guide and Results Interpretation",
        # Sidebar Upload
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
        # Main Results
        'ANALYSIS_HEADER': "2. Analysis Results",
        'COVERAGE_NOTE': "Coverage columns report the count of **unique words** from the text found in that category.",
        'LOADING_JLPT': "Loading JLPT Wordlists from CSVs...",
        'LOADING_FUGA': "Initializing Fugashi Tokenizer...",
        'PASS1_TEXT': "--- PASS 1: Analyzing components and raw metrics ---",
        'PASS2_TEXT': "--- PASS 2: Calculating JGRI and final results ---",
        'SUCCESS_LOAD': "Wordlists loaded successfully from CSVs!",
        'SUCCESS_TOKEN': "Fugashi tokenizer loaded successfully!",
        'ANALYSIS_COMPLETE': "Analysis complete!",
        'NO_FILES': "No valid text files were processed.",
        'EMPTY_FILE': "is empty, skipped.",
        'DECODE_ERROR': "Failed to decode",
        'UPLD_TO_BEGIN': "Please select a corpus source from the **sidebar** to begin analysis.",
        'FETCHING_CORPUS': "Fetching and processing preloaded corpus...",
        
        # N-gram & KWIC
        'NGRAM_MAIN_HEADER': "3. N-gram Frequency Analysis & Concordance",
        'NGRAM_WILDCARD_INFO': "Use the `*` symbol in the word filter boxes below to represent zero or more characters (e.g., `*ing` or `本*`).",
        'NGRAM_CURRENT': "**Current N-gram length selected:",
        'NGRAM_FILTER_HEADER': "Filter N-grams by Word (with Wildcards) or Part-of-Speech (POS)",
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
        
        # Visualizations (UPDATED)
        'VISUAL_HEADER': "4. Visualizations",
        'WORDCLOUD_HEADER': "Word Cloud (Most Frequent Tokens)", # ADDED
        'WORDCLOUD_NOTE': "The word cloud displays the frequency of the top 200 most common tokens across all analyzed texts. Word size indicates relative frequency.", # ADDED
        'JGRI_COMPARE_NOTE': "JGRI comparison requires at least two files.",
        'ROLLING_TTR_EXPANDER': "Show Rolling Mean TTR Curve (Vocabulary Trend)",
        'ROLLING_TTR_NOTE': "This plot shows the trend of vocabulary diversity over the length of the text. A flat, high line indicates sustained rich vocabulary.",
        
        # Summary Tables
        'SUMMARY_HEADER': "5. Summary Table (Lexical, Structural & Readability Metrics)",
        'JGRI_EXP_HEADER': "JGRI (Japanese Grammatical Readability Index) Explanation",
        'POS_HEADER': "6. Detailed Part-of-Speech (POS) Distribution",
        'POS_NOTE': "This table shows the percentage of **all** detected grammatical categories for each file.",
        'RAW_JGRI_EXPANDER': "Show Raw JGRI Components (Original Data for MMS, LD, VPS, MPN)",
        'RAW_JGRI_NOTE': "This table provides the original raw values used to calculate the JGRI index. These values are also in the main table.",
        'DOWNLOAD_ALL': "⬇️ Download All Results as Excel (Includes N-gram Data)",
    },
    'ID': {
        'LANG_NAME': 'Bahasa Indonesia',
        'TITLE': "🇯🇵 Profiler Kosakata Bahasa Jepang",
        'SUBTITLE': "Menganalisis kekayaan leksikal, kompleksitas struktural, dan cakupan kata JLPT.",
        'MANUAL_LINK': "https://docs.google.com/document/d/1SvfMQjsTm8uLI0PTwSOL1lTEiLhVUFArb6Q0lRHSiZU/edit?usp=sharing",
        'MANUAL_TEXT': "Panduan Pengguna dan Interpretasi Hasil",
        # Sidebar Upload
        'UPLOAD_HEADER': "1. Sumber Korpus",
        'SOURCE_SELECT': "Pilih Sumber Korpus:",
        'SOURCE_UPLOAD': "A. Unggah Berkas Lokal",
        'SOURCE_PRELOAD_1': "B. Pra-muat: DICO-JALF SEMUA",
        'SOURCE_PRELOAD_2': "C. Pra-muat: DICO-JALF 30 Berkas",
        'UPLOAD_INDIVIDUAL': "Unggah Berkas TXT Individual",
        'UPLOAD_BATCH': "Unggah Berkas Excel/CSV Batch",
        'UPLOAD_HELPER': "Berkas akan dianalisis terhadap daftar kata JLPT yang dimuat sebelumnya.",
        'UPLOAD_BUTTON': "Unggah satu atau lebih berkas **.txt** untuk analisis.",
        'EXCEL_BUTTON': "Unggah **Excel (.xlsx) / CSV (.csv)** untuk Pemrosesan Batch",
        'EXCEL_NOTE': "Sheet 1/Konten harus berisi: Kolom 1: Nama Berkas Teks, Kolom 2: Konten (Tanpa header).",
        'WORDLIST_HEADER': "2. Daftar Kata yang Digunakan",
        'WORDLIST_INFO': "Menggunakan daftar **Sumber Tidak Diketahui** yang dimuat sebelumnya",
        'NGRAM_HEADER': "3. Pengaturan N-gram",
        'NGRAM_RADIO': "Pilih Panjang N-gram (N)",
        'KWIC_CONTEXT_HEADER': "Konteks Konkordansi",
        'KWIC_LEFT': "Kata di Kiri",
        'KWIC_RIGHT': "Kata di Kanan",
        # Main Results
        'ANALYSIS_HEADER': "2. Hasil Analisis",
        'COVERAGE_NOTE': "Kolom cakupan melaporkan hitungan **kata unik** dari teks yang ditemukan dalam kategori tersebut.",
        'LOADING_JLPT': "Memuat Daftar Kata JLPT dari CSV...",
        'LOADING_FUGA': "Memulai Tokenizer Fugashi...",
        'PASS1_TEXT': "--- LANGKAH 1: Menganalisis komponen dan metrik mentah ---",
        'PASS2_TEXT': "--- LANGKAH 2: Menghitung JGRI dan hasil akhir ---",
        'SUCCESS_LOAD': "Daftar Kata JLPT berhasil dimuat dari CSV!",
        'SUCCESS_TOKEN': "Tokenizer Fugashi berhasil dimuat!",
        'ANALYSIS_COMPLETE': "Analisis selesai!",
        'NO_FILES': "Tidak ada berkas teks yang valid diproses.",
        'EMPTY_FILE': "kosong, dilewati.",
        'DECODE_ERROR': "Gagal mendekode",
        'UPLD_TO_BEGIN': "Mohon pilih sumber korpus dari **sidebar** untuk memulai analisis.",
        'FETCHING_CORPUS': "Mengambil dan memproses korpus pra-muat...",
        
        # N-gram & KWIC
        'NGRAM_MAIN_HEADER': "3. Analisis Frekuensi N-gram & Konkordansi",
        'NGRAM_WILDCARD_INFO': "Gunakan simbol `*` di kotak filter kata di bawah untuk mewakili nol atau lebih karakter (misalnya, `*ing` atau `本*`).",
        'NGRAM_CURRENT': "**Panjang N-gram saat ini yang dipilih:",
        'NGRAM_FILTER_HEADER': "Filter N-gram berdasarkan Kata (dengan Wildcard) atau Jenis Kata (POS)",
        'NGRAM_FREQ_HEADER': "Daftar Frekuensi N-gram",
        'NGRAM_TOTAL_UNIQUE': "**Total unik",
        'NGRAM_MATCHING_FILTER': "gram yang cocok dengan filter:",
        'CONCORDANCE_HEADER': "Konkordansi (Keyword In Context - KWIC)",
        'CONCORDANCE_LINES': "**Total baris konkordansi yang dihasilkan:",
        'KWIC_HELP_LEFT': "kata di kiri",
        'KWIC_HELP_RIGHT': "kata di kanan",
        'KWIC_HELP_KW': "Kata kunci filter",
        'DOWNLOAD_NGRAM': "⬇️ Unduh Daftar {n}-gram Terfilter Penuh ({count} item unik)",
        'DOWNLOAD_KWIC': "⬇️ Unduh Daftar Konkordansi Penuh ({count} baris)",

        # Visualizations (UPDATED)
        'VISUAL_HEADER': "4. Visualisasi",
        'WORDCLOUD_HEADER': "Awan Kata (Token Paling Sering Muncul)", # ADDED
        'WORDCLOUD_NOTE': "Awan kata menampilkan frekuensi dari 200 token paling umum di semua teks yang dianalisis. Ukuran kata menunjukkan frekuensi relatif.", # ADDED
        'JGRI_COMPARE_NOTE': "Perbandingan JGRI memerlukan minimal dua berkas.",
        'ROLLING_TTR_EXPANDER': "Tampilkan Kurva Rolling Mean TTR (Tren Kosakata)",
        'ROLLING_TTR_NOTE': "Plot ini menunjukkan tren keragaman kosakata sepanjang teks. Garis datar yang tinggi menunjukkan kosakata yang kaya berkelanjutan.",
        
        # Summary Tables
        'SUMMARY_HEADER': "5. Tabel Ringkasan (Metrik Leksikal, Struktural & Keterbacaan)",
        'JGRI_EXP_HEADER': "Penjelasan JGRI (Indeks Keterbacaan Gramatikal Jepang)",
        'POS_HEADER': "6. Distribusi Detil Jenis Kata (POS)",
        'POS_NOTE': "Tabel ini menunjukkan persentase **semua** kategori gramatikal yang terdeteksi untuk setiap berkas.",
        'RAW_JGRI_EXPANDER': "Tampilkan Komponen Mentah JGRI (Data Asli untuk MMS, LD, VPS, MPN)",
        'RAW_JGRI_NOTE': "Tabel ini menyediakan nilai mentah asli yang digunakan untuk menghitung indeks JGRI. Nilai-nilai ini juga ada di tabel utama.",
        'DOWNLOAD_ALL': "⬇️ Unduh Semua Hasil sebagai Excel (Termasuk Data N-gram)",
    },
    'JP': {
        'LANG_NAME': '日本語',
        'TITLE': "🇯🇵 日本語語彙プロファイラ",
        'SUBTITLE': "語彙の豊富さ、構造的な複雑さ、およびJLPTの単語カバー率を分析します。",
        'MANUAL_LINK': "https://docs.google.com/document/d/1tJB4lDKBUPBHHHB8Vj0fZyXtwH-lNDeF9tifDS7lzFQ/edit?usp=sharing",
        'MANUAL_TEXT': "ユーザーガイドと結果の解釈",
        # Sidebar Upload
        'UPLOAD_HEADER': "1. コーパスソースのロード",
        'SOURCE_SELECT': "コーパスソースを選択:",
        'SOURCE_UPLOAD': "A. ローカルファイルをアップロード",
        'SOURCE_PRELOAD_1': "B. 事前ロード: DICO-JALF すべて",
        'SOURCE_PRELOAD_2': "C. 事前ロード: DICO-JALF 30ファイルのみ",
        'UPLOAD_INDIVIDUAL': "個別TXTファイルのアップロード",
        'UPLOAD_BATCH': "Excel/CSVバッチファイルのアップロード",
        'EXCEL_BUTTON': "バッチ処理用の** Excel (.xlsx) / CSV (.csv) **をアップロード",
        'EXCEL_NOTE': "シート1/コンテンツには、列1: テキストファイル名、列2: コンテンツが必要です（ヘッダーなし）。",
        'UPLOAD_HELPER': "ファイルは事前にロードされたJLPT単語リストに対して分析されます。",
        'UPLOAD_BUTTON': "分析用の** .txt **ファイルを1つ以上アップロードしてください。",
        'WORDLIST_HEADER': "2. 使用される単語リスト",
        'WORDLIST_INFO': "事前ロードされた**不明なソース**リストを使用しています",
        'NGRAM_HEADER': "3. N-gram設定",
        'NGRAM_RADIO': "N-gramの長さ (N) を選択",
        'KWIC_CONTEXT_HEADER': "コンコーダンスの文脈",
        'KWIC_LEFT': "左側の単語数",
        'KWIC_RIGHT': "右側の単語数",
        # Main Results
        'ANALYSIS_HEADER': "2. 分析結果",
        'COVERAGE_NOTE': "カバー率の列は、そのカテゴリで見つかったテキストからの**ユニークな単語**の数を報告します。",
        'LOADING_JLPT': "JLPT単語リストをCSVからロード中...",
        'LOADING_FUGA': "Fugashiトークナイザを初期化中...",
        'PASS1_TEXT': "--- フェーズ1: コンポーネントと生メトリクスの分析 ---",
        'PASS2_TEXT': "--- フェーズ2: JGRIと最終結果の計算 ---",
        'SUCCESS_LOAD': "JLPT単語リストがCSVから正常にロードされました！",
        'SUCCESS_TOKEN': "Fugashiトークナイザが正常にロードされました！",
        'ANALYSIS_COMPLETE': "分析完了！",
        'NO_FILES': "有効なテキストファイルは処理されませんでした。",
        'EMPTY_FILE': "は空です。スキップされました。",
        'DECODE_ERROR': "デコードに失敗しました",
        'UPLD_TO_BEGIN': "サイドバーからコーパスソースを選択して分析を開始してください。",
        'FETCHING_CORPUS': "事前ロードされたコーパスを取得および処理しています...",
        
        # N-gram & KWIC
        'NGRAM_MAIN_HEADER': "3. N-gram頻度分析とコンコーダンス",
        'NGRAM_WILDCARD_INFO': "単語フィルターボックスでアスタリスク（`*`）記号を使用して、ゼロ個以上の文字を表します（例：`*ing` や `本*`）。",
        'NGRAM_CURRENT': "**現在選択されているN-gramの長さ:",
        'NGRAM_FILTER_HEADER': "単語（ワイルドカード付き）または品詞（POS）でN-gramをフィルタリング",
        'NGRAM_FREQ_HEADER': "N-gram頻度リスト",
        'NGRAM_TOTAL_UNIQUE': "**合計ユニーク",
        'NGRAM_MATCHING_FILTER': "個のN-gramがフィルターに一致:",
        'CONCORDANCE_HEADER': "コンコーダンス（キーワードインコンテキスト - KWIC）",
        'CONCORDANCE_LINES': "**生成されたコンコーダンスラインの合計:",
        'KWIC_HELP_LEFT': "左側の単語数",
        'KWIC_HELP_RIGHT': "右側の単語数",
        'KWIC_HELP_KW': "フィルタリングされた",
        'DOWNLOAD_NGRAM': "⬇️ 完全なフィルタリング済み{n}-gramリストをダウンロード（ユニークなアイテム{count}個）",
        'DOWNLOAD_KWIC': "⬇️ 完全なコンコーダンスリストをダウンロード（{count}行）",

        # Visualizations (UPDATED)
        'VISUAL_HEADER': "4. 可視化",
        'WORDCLOUD_HEADER': "ワードクラウド（最も頻繁なトークン）", # ADDED
        'WORDCLOUD_NOTE': "ワードクラウドは、分析されたすべてのテキストにおける最も一般的なトークン上位200語の頻度を表示します。単語のサイズは相対的な頻度を示します。", # ADDED
        'JGRI_COMPARE_NOTE': "JGRI比較には最低2つのファイルが必要です。",
        'ROLLING_TTR_EXPANDER': "ローリング平均TTRカーブを表示（語彙の傾向）",
        'ROLLING_TTR_NOTE': "このプロットは、テキストの長さにわたる語彙の多様性の傾向を示しています。平坦で高い線は、持続的に豊富な語彙を示します。",
        
        # Summary Tables
        'SUMMARY_HEADER': "5. 要約テーブル（語彙、構造、読みやすさのメトリクス）",
        'JGRI_EXP_HEADER': "JGRI（日本語文法可読性インデックス）の説明",
        'POS_HEADER': "6. 詳細な品詞（POS）分布",
        'POS_NOTE': "このテーブルは、各ファイルで検出された**すべて**の文法カテゴリのパーセンテージを示しています。",
        'RAW_JGRI_EXPANDER': "生JGRIコンポーネントを表示（MMS、LD、VPS、MPNのオリジナルデータ）",
        'RAW_JGRI_NOTE': "このテーブルは、JGRIインデックスの計算に使用された元の生値を提供します。これらの値はメインテーブルにも含まれています。",
        'DOWNLOAD_ALL': "⬇️ すべての結果をExcelとしてダウンロード（N-gramデータを含む）",
    },
}

# ===============================================
# --- PRELOADED CORORPA CONFIGURATION ---
# ===============================================

PRELOADED_CORPORA = {
    "DICO-JALF ALL": "https://raw.githubusercontent.com/prihantoro-corpus/vocabulary-profiler/main/DICO-JALF%20all.xlsx",
    "DICO-JALF 30 Files Only": "https://raw.githubusercontent.com/prihantoro-corpus/vocabulary-profiler/main/DICO-JALF%2030%20files%20only.xlsx",
}

# ===============================================
# --- LANGUAGE SELECTION ---
# ===============================================
lang_options = {'English': 'EN', 'Bahasa Indonesia': 'ID', '日本語': 'JP'}
selected_lang_name = st.sidebar.selectbox("Language / Bahasa / 言語", options=list(lang_options.keys()), key='language_selector')
T = TRANSLATIONS[lang_options[selected_lang_name]]


# --- Configuration ---
# File names MUST match the CSV files committed to your GitHub repository root.
JLPT_FILE_MAP = {
    "JLPT N5": "unknown_source_N5.csv",
    "JLPT N4": "unknown_source_N4.csv",
    "JLPT N3": "unknown_source_N3.csv",
    "JLPT N2": "unknown_source_N2.csv",
    "JLPT N1": "unknown_source_N1.csv",
}

ALL_JLPT_LEVELS = list(JLPT_FILE_MAP.keys())
ALL_OUTPUT_LEVELS = ALL_JLPT_LEVELS + ["NA"]

# Global variable for dynamic POS options (populated after analysis)
POS_OPTIONS = []

# --- Layout and Title ---
st.set_page_config(
    page_title=T['TITLE'],
    layout="wide"
)

st.title(T['TITLE'])
st.markdown(T['SUBTITLE'])

# ===============================================
# Helper Functions - Caching for Performance
# ===============================================

@st.cache_data(show_spinner=T['LOADING_JLPT'])
def load_jlpt_wordlist(T):
    """Loads all five JLPT wordlists."""
    jlpt_dict = {}
    for level_name, filename in JLPT_FILE_MAP.items():
        # This assumes the CSV files are present in the same directory/repository root
        if not os.path.exists(filename):
            st.error(f"Required CSV file '{filename}' not found in the repository root.")
            # For demonstration purposes in a controlled environment, you might fetch them, 
            # but standard Streamlit practice is to include them locally.
            # Skipping further check if files are missing.
            # return None 
            
            # --- FALLBACK FOR MISSING LOCAL FILES (DANGEROUS IN PROD, OK FOR DEMO) ---
            st.warning(f"Attempting to fetch missing file: {filename}")
            try:
                base_url = "https://raw.githubusercontent.com/prihantoro-corpus/vocabulary-profiler/main/"
                response = requests.get(base_url + filename)
                response.raise_for_status()
                df = pd.read_csv(io.StringIO(response.text), header=0, encoding='utf-8', keep_default_na=False)
            except Exception as e:
                st.error(f"Error fetching/reading fallback CSV for '{filename}': {e}")
                return None
            # --- END FALLBACK ---
        else:
            try:
                df = pd.read_csv(filename, header=0, encoding='utf-8', keep_default_na=False)
            except Exception as e:
                st.error(f"Error reading CSV file '{filename}': {e}")
                return None

        if df.empty:
            words = set()
        else:
            word_column = df.columns[0]
            words = set(df[word_column].astype(str).tolist())
        jlpt_dict[level_name] = words
            
    st.success(T['SUCCESS_LOAD'])
    return jlpt_dict

@st.cache_resource(show_spinner=T['LOADING_FUGA'])
def initialize_tokenizer(T):
    """Initializes the Fugashi Tagger."""
    try:
        tagger = Tagger()
        st.success(T['SUCCESS_TOKEN'])
        return tagger
    except Exception as e:
        st.error(f"Error initializing Fugashi: {e}")
        st.error("Please ensure 'unidic-lite' is in your requirements.txt to fix MeCab initialization.")
        st.stop()
        return None

# ===============================================
# New Helper Functions: File Handling
# ===============================================

# Helper class to combine Excel/CSV output with Streamlit's UploadedFile objects
class MockUploadedFile:
    """Mock file object for preloaded or processed batch data."""
    def __init__(self, name, data_io):
        self.name = name
        self._data_io = data_io
    def read(self):
        # Reset pointer before reading to ensure full content is captured
        self._data_io.seek(0)
        return self._data_io.read()


def process_excel_upload(uploaded_file):
    """
    Reads a file (Excel or CSV) assuming Column 1=Filename, Column 2=Content (no header).
    Returns a list of (filename, data_io) tuples.
    """
    processed_files = []
    if uploaded_file is None:
        return processed_files

    file_extension = uploaded_file.name.split('.')[-1].lower()
    
    try:
        # Read the uploaded file's content
        uploaded_file.seek(0)
        
        if file_extension == 'xlsx':
            df = pd.read_excel(uploaded_file, header=None, sheet_name=0)
        elif file_extension == 'csv':
            # Use io.BytesIO to read the CSV content safely
            data = uploaded_file.read()
            df = pd.read_csv(io.BytesIO(data), header=None)
        else:
            # Should not happen if the file type filter is correct, but safe guard.
            st.sidebar.warning(f"Unsupported file type for batch upload: .{file_extension}")
            return processed_files
        
        if df.empty or df.shape[1] < 2:
            st.sidebar.warning(f"File '{uploaded_file.name}' is empty or does not contain required columns (1 and 2).")
            return processed_files

        # Columns 0 and 1 correspond to the first two columns (Filename and Content)
        df.iloc[:, 0] = df.iloc[:, 0].astype(str).str.strip()
        df.iloc[:, 1] = df.iloc[:, 1].astype(str).str.strip()
        df = df.dropna(subset=[0, 1])

        for index, row in df.iterrows():
            filename = row.iloc[0]
            content = row.iloc[1]
            
            if filename == 'nan' or content == 'nan' or not filename or not content:
                continue 

            processed_files.append((filename, io.BytesIO(content.encode('utf-8'))))
        
        if processed_files:
            st.sidebar.success(f"Successfully loaded {len(processed_files)} texts from batch file.")
        return processed_files

    except Exception as e:
        st.sidebar.error(f"Error reading batch file: {e}. Please ensure it is correctly formatted with no header.")
        return []

@st.cache_data(show_spinner=T['FETCHING_CORPUS'])
def load_preloaded_corpus(url, name):
    """
    Fetches an Excel file directly from a URL, processes it, and returns
    a list of MockUploadedFile objects.
    """
    try:
        # 1. Fetch the data
        response = requests.get(url, allow_redirects=True)
        response.raise_for_status() # Raises an HTTPError if the status is 4xx or 5xx

        # 2. Read the Excel content directly from the byte stream
        data_io = io.BytesIO(response.content)
        df = pd.read_excel(data_io, header=None, sheet_name=0)

        # 3. Process the DataFrame (same logic as excel upload)
        if df.empty or df.shape[1] < 2:
            st.error(f"Preloaded corpus '{name}' is empty or misformatted.")
            return []

        df.iloc[:, 0] = df.iloc[:, 0].astype(str).str.strip()
        df.iloc[:, 1] = df.iloc[:, 1].astype(str).str.strip()
        df = df.dropna(subset=[0, 1])
        
        mock_files = []
        for index, row in df.iterrows():
            filename = row.iloc[0]
            content = row.iloc[1]
            if filename == 'nan' or content == 'nan' or not filename or not content:
                continue 
                
            mock_files.append(
                MockUploadedFile(filename, io.BytesIO(content.encode('utf-8')))
            )
        
        if not mock_files:
            st.error(f"Preloaded corpus '{name}' contains no valid text entries.")
            return []
            
        st.success(f"Successfully loaded {len(mock_files)} texts from preloaded corpus: {name}.")
        return mock_files

    except requests.exceptions.HTTPError as err:
        st.error(f"Error fetching preloaded corpus '{name}': HTTP error {err.response.status_code}. Please check the URL.")
        return []
    except Exception as e:
        st.error(f"Error processing preloaded corpus '{name}': {e}")
        return []


# ===============================================
# Core N-gram and KWIC Analysis
# ===============================================

def get_n_grams(tagged_nodes, n):
    """
    Extracts n-grams (sequences of words) and their corresponding POS tag sequences.
    """
    words = [node.surface for node in tagged_nodes if node.surface]
    pos_tags = [node.feature.pos1 for node in tagged_nodes if node.feature.pos1]
    
    n_grams = []
    n_gram_pos = []
    indices = []
    
    # We must iterate up to len(words) - n + 1
    for i in range(len(words) - n + 1):
        # Join words with spaces for display
        n_gram_words = " ".join(words[i:i + n])
        # Join POS tags with underscores for filtering
        n_gram_pos_sequence = "_".join(pos_tags[i:i + n])
        
        n_grams.append(n_gram_words)
        n_gram_pos.append(n_gram_pos_sequence)
        indices.append(i) # Starting index of the n-gram
        
    return pd.DataFrame({'N_gram': n_grams, 'POS_Sequence': n_gram_pos, 'Start_Index': indices, 'Filename': ''})

def calculate_n_gram_frequency(df_n_grams):
    """Calculates frequency and percentage for all unique N-grams."""
    
    if df_n_grams.empty:
        return pd.DataFrame(columns=['N_gram', 'Frequency', 'Percentage', 'POS_Sequence'])

    # Group by N_gram (words) and POS_Sequence to get frequency
    df_freq = df_n_grams.groupby(['N_gram', 'POS_Sequence']).size().reset_index(name='Frequency')
    total_grams = df_freq['Frequency'].sum()
    
    # Calculate percentage
    df_freq['Percentage'] = (df_freq['Frequency'] / total_grams) * 100
    
    df_freq = df_freq.sort_values(by='Frequency', ascending=False).reset_index(drop=True)
    
    # Format percentage for display
    df_freq['Percentage'] = df_freq['Percentage'].map('{:.3f}%'.format)
    
    return df_freq

def get_unique_pos_options(corpus_data):
    """Collects all unique POS tags across the entire corpus."""
    all_pos = set()
    for data in corpus_data:
        for node in data['Tagged_Nodes']:
            if node.feature.pos1:
                all_pos.add(node.feature.pos1)
    
    global POS_OPTIONS
    POS_OPTIONS = sorted(list(all_pos))
    return ['(Any)'] + POS_OPTIONS

def apply_n_gram_filters(df_freq, filters, n):
    """
    Filters the N-gram DataFrame based on user inputs (words and POS tags),
    supporting '*' wildcard matching for words.
    """
    df_filtered = df_freq.copy()
    
    for i in range(n):
        word_filter = filters.get(f'word_{i}', '').strip()
        pos_filter = filters.get(f'pos_{i}', '(Any)').strip()
        
        # 1. Word Filtering (with Wildcard support)
        if word_filter:
            # Convert user's '*' wildcard syntax to Python regex '.*', and escape other chars
            regex_pattern = re.escape(word_filter).replace(r'\*', '.*')
            regex_pattern = f"^{regex_pattern}$" # Anchor the pattern to match the whole word
            
            def filter_by_word_regex(row, idx, pattern):
                words = row['N_gram'].split(' ')
                return re.match(pattern, words[idx]) is not None
            
            df_filtered = df_filtered[df_filtered.apply(
                lambda row: filter_by_word_regex(row, i, regex_pattern), axis=1
            )]

        # 2. POS Filtering
        if pos_filter != '(Any)':
            def filter_by_pos(row, idx, search_tag):
                tags = row['POS_Sequence'].split('_')
                return tags[idx] == search_tag

            df_filtered = df_filtered[df_filtered.apply(
                lambda row: filter_by_pos(row, i, pos_filter), axis=1
            )]

    # Recalculate percentage after filtering
    if not df_filtered.empty:
        total_grams = df_filtered['Frequency'].sum()
        df_filtered['Percentage'] = (df_filtered['Frequency'] / total_grams) * 100
        df_filtered['Percentage'] = df_filtered['Percentage'].map('{:.3f}%'.format)

    return df_filtered.sort_values(by='Frequency', ascending=False)

def generate_concordance(corpus_data, filters, n_gram_size, left_context, right_context):
    """
    Generates a Keyword In Context (KWIC) list based on the current N-gram filters.
    """
    concordance_list = []
    
    # 1. Get ALL N-gram instances that match the filter across the entire corpus
    matching_n_grams = []
    for data in corpus_data:
        # Generate N-grams for the specific file
        df_n = get_n_grams(data['Tagged_Nodes'], n_gram_size)
        df_n['Filename'] = data['Filename']
        
        # Apply word and POS filters to this file's N-grams
        df_match = df_n.copy()
        
        for i in range(n_gram_size):
            word_filter = filters.get(f'word_{i}', '').strip()
            pos_filter = filters.get(f'pos_{i}', '(Any)').strip()

            # Word Filtering (with Wildcard)
            if word_filter:
                regex_pattern = re.escape(word_filter).replace(r'\*', '.*')
                regex_pattern = f"^{regex_pattern}$"
                
                df_match = df_match[df_match.apply(
                    lambda row: re.match(regex_pattern, row['N_gram'].split(' ')[i]) is not None, axis=1
                )]

            # POS Filtering
            if pos_filter != '(Any)':
                df_match = df_match[df_match.apply(
                    lambda row: row['POS_Sequence'].split('_')[i] == pos_filter, axis=1
                )]
        
        if not df_match.empty:
            matching_n_grams.append(df_match)
    
    if not matching_n_grams:
        return pd.DataFrame(columns=['Filename', 'Left Context', 'Keyword(s)', 'Right Context'])

    df_matched_grams = pd.concat(matching_n_grams, ignore_index=True)

    # 2. Extract Context for each match
    for index, row in df_matched_grams.iterrows():
        filename = row['Filename']
        start_index = row['Start_Index']
        n_gram_words = row['N_gram'].split(' ')
        n = len(n_gram_words)
        
        # Find the tokens for the current file
        tokens = next(data['Tokens'] for data in corpus_data if data['Filename'] == filename)
        
        # Calculate context indices
        left_start = max(0, start_index - left_context)
        left_end = start_index
        right_start = start_index + n
        right_end = min(len(tokens), start_index + n + right_context)
        
        # Extract context
        left_context_words = tokens[left_start:left_end]
        right_context_words = tokens[right_start:right_end]
        
        concordance_list.append({
            'Filename': filename,
            'Left Context': " ".join(left_context_words),
            'Keyword(s)': " ".join(n_gram_words),
            'Right Context': " ".join(right_context_words),
        })

    return pd.DataFrame(concordance_list)

# ===============================================
# Analysis and Plotting Functions 
# ===============================================

def analyze_jgri_components(text, tagged_nodes):
    """Calculates the raw values for the four core JGRI components."""
    
    # 1. POS Counting
    pos_counts = Counter(node.feature.pos1 for node in tagged_nodes if node.surface and node.feature.pos1)
    
    Nouns = pos_counts.get('名詞', 0)
    Verbs = pos_counts.get('動詞', 0)
    Adjectives = pos_counts.get('形容詞', 0)
    Adverbs = pos_counts.get('副詞', 0)
    
    Total_Morphemes = len(tagged_nodes) # Proxy for morpheme count
    
    # 2. Sentence Counting
    sentences = re.split(r'[。！？\n]', text.strip())
    sentences = [s.strip() for s in sentences if s.strip()]
    Num_Sentences = len(sentences)

    # Handle division by zero
    if Total_Morphemes == 0 or Nouns == 0 or Num_Sentences == 0:
        return {'MMS': 0.0, 'LD': 0.0, 'VPS': 0.0, 'MPN': 0.0}

    # Component 1: Mean Morphemes per Sentence (MMS)
    MMS = Total_Morphemes / Num_Sentences
    
    # Component 2: Lexical Density (LD)
    LD = (Nouns + Verbs + Adjectives + Adverbs) / Total_Morphemes
    
    # Component 3: Verbs per Sentence (VPS)
    VPS = Verbs / Num_Sentences
    
    # Component 4: Modifiers per Noun (MPN) (Adjectives + Verbs/Relative Clauses)
    MPN = (Adjectives + Verbs) / Nouns
    
    return {'MMS': MMS, 'LD': LD, 'VPS': VPS, 'MPN': MPN}

def calculate_jgri(metrics_df):
    """Performs Z-score normalization and calculates the final JGRI index."""
    
    jgri_values = []
    
    # Calculate Mean (mu) and Standard Deviation (sigma) for the corpus
    mu = metrics_df[['MMS', 'LD', 'VPS', 'MPN']].mean()
    sigma = metrics_df[['MMS', 'LD', 'VPS', 'MPN']].std()

    # Handle cases where sigma is zero (e.g., if only one text is uploaded)
    sigma = sigma.replace(0, 1e-6) 
    
    for index, row in metrics_df.iterrows():
        raw_values = row[['MMS', 'LD', 'VPS', 'MPN']]
        
        # Calculate Z-score for each component
        z_mms = (raw_values['MMS'] - mu['MMS']) / sigma['MMS']
        z_ld = (raw_values['LD'] - mu['LD']) / sigma['LD']
        z_vps = (raw_values['VPS'] - mu['VPS']) / sigma['VPS']
        z_mpn = (raw_values['MPN'] - mu['MPN']) / sigma['MPN']
        
        # Composite formula: JGRI = (zMMS + zLD + zVPS + zMPN) / 4
        jgri = (z_mms + z_ld + z_vps + z_mpn) / 4
        jgri_values.append(round(jgri, 3))
        
    return jgri_values

def analyze_script_distribution(text):
    total_chars = len(text)
    if total_chars == 0:
        return {"Kanji": 0, "Hiragana": 0, "Katakana": 0, "Other": 0}
    patterns = {
        "Kanji": r'[\u4E00-\u9FFF]',
        "Hiragana": r'[\u3040-\u309F]',
        "Katakana": r'[\u30A0-\u30FF]',
    }
    counts = {name: len(re.findall(pattern, text)) for name, pattern in patterns.items()}
    counted_chars = sum(counts.values())
    counts["Other"] = total_chars - counted_chars
    percentages = {name: round((count / total_chars) * 100, 1) for name, count in counts.items()}
    return percentages

def analyze_kanji_density(text):
    sentences = re.split(r'[。！？\n]', text.strip())
    sentences = [s.strip() for s in sentences if s.strip()]
    if not sentences:
        return 0.0
    total_kanji = len(re.findall(r'[\u4E00-\u9FFF]', text))
    num_sentences = len(sentences)
    density = total_kanji / num_sentences
    return round(density, 2)

def analyze_jlpt_coverage(tokens, jlpt_dict):
    unique_tokens_in_text = set(tokens)
    result = {}
    total_known_words = set()
    for level, wordset in jlpt_dict.items():
        count = sum(1 for w in unique_tokens_in_text if w in wordset)
        result[level] = count
        total_known_words.update(w for w in unique_tokens_in_text if w in wordset)
    na_count = len(unique_tokens_in_text) - len(total_known_words)
    result["NA"] = na_count
    return result

def analyze_pos_distribution(tagged_nodes, filename):
    if not tagged_nodes:
        return {"Filename": filename}, {"Filename": filename}
    pos_tags = [
        node.feature.pos1 
        for node in tagged_nodes 
        if node.surface and node.feature.pos1
    ]
    if not pos_tags:
        return {"Filename": filename}, {"Filename": filename}
    total_tokens = len(pos_tags)
    pos_counts = Counter(pos_tags)
    pos_percentages = {"Filename": filename}
    pos_raw_counts = {"Filename": filename}
    for tag, count in pos_counts.items():
        percentage = round((count / total_tokens) * 100, 1)
        pos_percentages[tag] = percentage
        pos_raw_counts[tag] = count
    return pos_percentages, pos_raw_counts

def plot_jlpt_coverage(df, filename="jlpt_coverage.png"):
    df_plot = df[['Filename', 'JLPT_N5', 'JLPT_N4', 'JLPT_N3', 'JLPT_N2', 'JLPT_N1', 'NA']].copy()
    df_plot['Total_Types'] = df_plot.iloc[:, 1:].sum(axis=1)
    for col in df_plot.columns[1:-1]:
        df_plot[col] = (df_plot[col] / df_plot['Total_Types']) * 100
    df_plot = df_plot.set_index('Filename').drop(columns='Total_Types')
    colors = {'JLPT_N5': '#51A3A3', 'JLPT_N4': '#51C4D4','JLPT_N3': '#FFD000','JLPT_N2': '#FFA500','JLPT_N1': '#FF6347','NA': '#8B0000'}
    fig, ax = plt.subplots(figsize=(10, 6))
    df_plot.plot(kind='barh', stacked=True, color=[colors[col] for col in df_plot.columns], ax=ax)
    ax.set_title("JLPT Vocabulary Coverage (Proportion of Unique Words)", fontsize=14)
    ax.set_xlabel("Percentage of Unique Words (%)", fontsize=12)
    ax.set_ylabel("Text File", fontsize=12)
    ax.legend(title="Vocabulary Level", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(filename); plt.close(fig); return filename

def plot_jgri_comparison(df, filename="jgri_comparison.png"):
    df_plot = df[['Filename', 'JGRI']].set_index('Filename')
    colors = ['#1f77b4' if x >= 0 else '#d62728' for x in df_plot['JGRI']]
    fig, ax = plt.subplots(figsize=(10, 6))
    df_plot['JGRI'].plot(kind='bar', color=colors, ax=ax)
    ax.axhline(0, color='gray', linestyle='--')
    ax.set_title("JGRI Comparison (Relative Grammatical Complexity)", fontsize=14)
    ax.set_xlabel("Text File", fontsize=12)
    ax.set_ylabel("JGRI Score (Z-Score Average)", fontsize=12)
    ax.tick_params(axis='x', rotation=45)
    plt.tight_layout()
    plt.savefig(filename); plt.close(fig); return filename

def plot_scripts_distribution(df, filename="scripts_distribution.png"):
    df_scripts = pd.DataFrame()
    for index, row in df.iterrows():
        parts = row['Script_Distribution'].split(' | ')
        data = {p.split(': ')[0].strip(): float(p.split(': ')[1].replace('%', '').strip()) for p in parts}
        df_scripts = pd.concat([df_scripts, pd.DataFrame([data], index=[row['Filename']])])
    
    # Rename columns for simpler plotting labels
    df_scripts = df_scripts.rename(columns={'Kanji': 'K', 'Hiragana': 'H', 'Katakana': 'T', 'Other': 'O'})
    script_cols = ['K', 'H', 'T', 'O']
    df_scripts = df_scripts[script_cols].fillna(0)
    colors = {'K': '#483D8B', 'H': '#8A2BE2', 'T': '#DA70D6', 'O': '#A9A9A9'}
    fig, ax = plt.subplots(figsize=(10, 6))
    df_scripts.plot(kind='barh', stacked=True, color=[colors[col] for col in df_scripts.columns], ax=ax)
    ax.set_title("Script Distribution (Percentage of Characters)", fontsize=14)
    ax.set_xlabel("Percentage (%)", fontsize=12)
    ax.set_ylabel("Text File", fontsize=12)
    ax.legend(['Kanji', 'Hiragana', 'Katakana', 'Other'], title="Script Type", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(filename); plt.close(fig); return filename

def plot_mtld_comparison(df, filename="mtld_comparison.png"):
    df_plot = df[['Filename', 'MTLD']].set_index('Filename')
    fig, ax = plt.subplots(figsize=(10, 6))
    df_plot['MTLD'].plot(kind='bar', color='#3CB371', ax=ax)
    ax.set_title("MTLD Comparison (Lexical Diversity)", fontsize=14)
    ax.set_xlabel("Text File", fontsize=12)
    ax.set_ylabel("MTLD Score", fontsize=12)
    ax.tick_params(axis='x', rotation=45)
    plt.tight_layout()
    plt.savefig(filename); plt.close(fig); return filename

def plot_token_count_comparison(df, filename="token_count_comparison.png"):
    df_plot = df[['Filename', 'Tokens']].set_index('Filename')
    fig, ax = plt.subplots(figsize=(10, 6))
    df_plot['Tokens'].plot(kind='bar', color='#6A5ACD', ax=ax)
    ax.set_title("Total Token Count Comparison", fontsize=14)
    ax.set_xlabel("Text File", fontsize=12)
    ax.set_ylabel("Total Tokens (Words)", fontsize=12)
    ax.tick_params(axis='x', rotation=45)
    formatter = ticker.FuncFormatter(lambda x, p: format(int(x), ','))
    ax.yaxis.set_major_formatter(formatter)
    plt.tight_layout()
    plt.savefig(filename); plt.close(fig); return filename

def plot_rolling_ttr_curve(corpus_data, window_size=50, filename="rolling_ttr_curve.png"):
    fig, ax = plt.subplots(figsize=(10, 6))
    is_data_plotted = False
    for data in corpus_data: 
        tokens = data['Tokens']; filename_label = data['Filename']
        if not tokens or len(tokens) < window_size: continue
        ttr_values = []
        for i in range(len(tokens) - window_size + 1): 
            window = tokens[i:i + window_size]; ttr = len(set(window)) / window_size
            ttr_values.append(ttr)
        x_axis = np.arange(window_size, len(tokens) + 1)
        ax.plot(x_axis, ttr_values, label=filename_label)
        is_data_plotted = True
    if not is_data_plotted: 
        ax.text(0.5, 0.5, f"No texts long enough for window size {window_size}.", transform=ax.transAxes, ha='center', color='red')
    ax.set_title(f"Rolling Mean TTR Curve (Window Size: {window_size})", fontsize=14)
    ax.set_xlabel("Tokens (Total Words)", fontsize=12)
    ax.set_ylabel("Rolling TTR (0 to 1)", fontsize=12)
    ax.legend(title="Text File", loc='upper right')
    ax.set_ylim(0, 1)
    formatter = ticker.FuncFormatter(lambda x, p: format(int(x), ','))
    ax.xaxis.set_major_formatter(formatter)
    plt.tight_layout()
    plt.savefig(filename); plt.close(fig); return filename

def plot_ttr_comparison(df, filename="ttr_comparison.png"):
    df_plot = df[['Filename', 'TTR']].set_index('Filename')
    fig, ax = plt.subplots(figsize=(10, 6))
    df_plot['TTR'].plot(kind='bar', color='#FF8C00', ax=ax)
    ax.set_title("Type-Token Ratio (TTR) Comparison", fontsize=14)
    ax.set_xlabel("Text File", fontsize=12)
    ax.set_ylabel("TTR Score (0-1)", fontsize=12)
    ax.tick_params(axis='x', rotation=45)
    ax.set_ylim(0, df_plot['TTR'].max() * 1.1)
    plt.tight_layout()
    plt.savefig(filename); plt.close(fig); return filename

def plot_pos_comparison(df_pos_percentage, filename="pos_comparison.png"):
    df_plot = df_pos_percentage.set_index('Filename').copy()
    all_tags = df_plot.columns.tolist()
    total_tag_percentage = df_plot.mean().sort_values(ascending=False)
    top_tags = total_tag_percentage.head(10).index.tolist()
    df_plot_top = df_plot[top_tags]
    cmap = plt.cm.get_cmap('tab20', len(top_tags))
    fig, ax = plt.subplots(figsize=(10, 6))
    df_plot_top.plot(kind='barh', stacked=True, colormap=cmap, ax=ax)
    ax.set_title("Normalized Part-of-Speech Distribution (Top 10 Categories)", fontsize=14)
    ax.set_xlabel("Percentage (%)", fontsize=12)
    ax.set_ylabel("Text File", fontsize=12)
    ax.legend(title="POS Category", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(filename); plt.close(fig); return filename

# NEW FUNCTION: Get Japanese Font (Cached)
@st.cache_resource
def get_jp_font_path():
    """Fetches a high-quality Japanese font file for WordCloud."""
    # Using Noto Sans CJK JP Regular from GitHub for reliability
    font_url = "https://raw.githubusercontent.com/googlefonts/noto-cjk/main/Sans/OTC/NotoSansCJKjp-Regular.otf"
    font_path = "NotoSansCJKjp-Regular.otf"

    if not os.path.exists(font_path):
        try:
            st.info("Fetching Japanese font for Word Cloud...")
            response = requests.get(font_url, stream=True)
            response.raise_for_status()
            with open(font_path, 'wb') as f:
                f.write(response.content)
            st.success("Japanese font loaded.")
        except Exception as e:
            st.error(f"Could not download required Japanese font for Word Cloud: {e}. Word cloud may fail.")
            return None 
            
    return font_path

# NEW FUNCTION: Word Cloud Plotting
def plot_word_cloud(corpus_data, filename="word_cloud.png"):
    """Generates and saves a word cloud based on all tokens in the corpus."""
    all_tokens = []
    for data in corpus_data:
        # Concatenate tokens from all texts
        all_tokens.extend(data['Tokens'])

    if not all_tokens:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.text(0.5, 0.5, "No tokens to display in word cloud.", transform=ax.transAxes, ha='center', color='red')
        plt.axis('off')
        plt.savefig(filename); plt.close(fig); return filename
    
    # 1. Generate token frequency (using Counter from collections)
    token_counts = Counter(all_tokens)
    
    # 2. Get font path (critical for Japanese text)
    font_path = get_jp_font_path()
    
    # Fallback/Safety Check
    if not font_path and os.path.exists('/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf'):
        # Crude fallback, unlikely to support JP well
        font_path = '/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf'
    elif not font_path:
        st.warning("Could not find a Japanese font. Word cloud may display incorrectly.")

    wc = WordCloud(
        font_path=font_path, 
        width=1000, 
        height=600, 
        background_color="white", 
        max_words=200, 
        min_font_size=10,
        # Custom regex to ensure Japanese characters are included as words/tokens
        regexp=r"[\w\u3040-\u309F\u30A0-\u30FF\u4E00-\u9FFF]+" 
    )

    # Generate word cloud from frequencies
    wc.generate_from_frequencies(token_counts)
    
    # Save the plot
    plt.figure(figsize=(10, 6))
    plt.imshow(wc, interpolation="bilinear")
    plt.axis("off")
    plt.tight_layout(pad=0)
    plt.savefig(filename); 
    plt.close() # Close figure to free memory
    return filename


# ===============================================
# Sidebar & Initialization
# ===============================================

# Load essential components
jlpt_dict_to_use = load_jlpt_wordlist(T)
tagger = initialize_tokenizer(T)

if jlpt_dict_to_use is None or tagger is None:
    st.stop() 

# --- Sidebar Configuration: DOC LINK AT TOP (Dynamic) ---
st.sidebar.markdown(
    f"**Documentation:** [{T['MANUAL_TEXT']}]({T['MANUAL_LINK']})"
)
st.sidebar.markdown("---")

st.sidebar.header(T['UPLOAD_HEADER'])

# --- Source Selection Radio ---
source_selection = st.sidebar.radio(
    T['SOURCE_SELECT'],
    options=[
        T['SOURCE_UPLOAD'], 
        T['SOURCE_PRELOAD_1'], 
        T['SOURCE_PRELOAD_2']
    ],
    key='source_selection_radio'
)

uploaded_files_combined = []

if source_selection == T['SOURCE_UPLOAD']:
    # --- A. Upload Individual TXT Files ---
    st.sidebar.markdown(f"**{T['UPLOAD_INDIVIDUAL']}**")
    input_files_txt = st.sidebar.file_uploader(
        T['UPLOAD_BUTTON'],
        type=["txt"],
        accept_multiple_files=True,
        key="input_uploader_txt",
        help=T['UPLOAD_HELPER']
    )
    st.sidebar.markdown("---")

    # --- B. Upload Batch Excel/CSV File ---
    st.sidebar.markdown(f"**{T['UPLOAD_BATCH']}**")
    input_files_excel = st.sidebar.file_uploader(
        T['EXCEL_BUTTON'],
        type=["xlsx", "csv"], # Accept both XLSX and CSV
        key="input_uploader_excel",
        help=T['EXCEL_NOTE']
    )

    # Process local uploads
    if input_files_txt:
        uploaded_files_combined.extend(input_files_txt)
        
    if input_files_excel:
        processed_excel_files = process_excel_upload(input_files_excel)
        for filename, data_io in processed_excel_files:
            uploaded_files_combined.append(MockUploadedFile(filename, data_io))
            
elif source_selection == T['SOURCE_PRELOAD_1']:
    # Load DICO-JALF ALL
    url = PRELOADED_CORPORA["DICO-JALF ALL"]
    name = "DICO-JALF ALL"
    preloaded_files = load_preloaded_corpus(url, name)
    uploaded_files_combined.extend(preloaded_files)
    
elif source_selection == T['SOURCE_PRELOAD_2']:
    # Load DICO-JALF 30 Files Only
    url = PRELOADED_CORPORA["DICO-JALF 30 Files Only"]
    name = "DICO-JALF 30 Files Only"
    preloaded_files = load_preloaded_corpus(url, name)
    uploaded_files_combined.extend(preloaded_files)
    
st.sidebar.markdown("---")
st.sidebar.header(T['WORDLIST_HEADER'])
st.sidebar.info(f"{T['WORDLIST_INFO']} ({len(ALL_JLPT_LEVELS)} levels).")

# ===============================================
# Main Area: Process and Display
# ===============================================

results = []
pos_percentage_results = []
pos_count_results = []
corpus_data = [] 

if uploaded_files_combined:
    
    # --- PASS 1 & 2: Data Processing ---
    
    st.header(T['ANALYSIS_HEADER'])
    st.markdown(T['COVERAGE_NOTE'])
    
    progress_bar = st.progress(0, text=T['PASS1_TEXT'])
    
    # --- START OF FILE PROCESSING LOOP (Handles all file sources) ---
    for i, uploaded_file in enumerate(uploaded_files_combined):
        filename = uploaded_file.name
        
        # Ensure the file pointer is at the start
        if hasattr(uploaded_file, '_data_io'): 
            uploaded_file._data_io.seek(0)
            
        content_bytes = uploaded_file.read()
        
        try:
             text = content_bytes.decode('utf-8')
        except UnicodeDecodeError:
             st.error(f"{T['DECODE_ERROR']} {filename}. Ensure it is UTF-8 encoded.")
             progress_bar.progress((i + 1) / len(uploaded_files_combined))
             continue
             
        text = text.strip()
        if not text:
            st.warning(f"File {filename} {T['EMPTY_FILE']}")
            progress_bar.progress((i + 1) / len(uploaded_files_combined))
            continue
        
        tagged_nodes = list(tagger(text))
        jgri_raw_components = analyze_jgri_components(text, tagged_nodes)
        jread = analyze_jreadability(text, tagged_nodes)
        
        # Store essential data for later analysis (JGRI, N-gram, WordCloud)
        corpus_data.append({
            'Filename': filename,
            'Text': text,
            'Tagged_Nodes': tagged_nodes,
            'Tokens': [word.surface for word in tagged_nodes],
            **jgri_raw_components, **jread
        })
        progress_bar.progress((i + 1) / len(uploaded_files_combined), text=f"PASS 1: Analyzed {i+1} of {len(uploaded_files_combined)} files.")
    # --- END OF FILE PROCESSING LOOP ---

    if not corpus_data:
        progress_bar.empty(); st.error(T['NO_FILES']); st.stop()

    df_raw_metrics = pd.DataFrame(corpus_data)
    progress_bar.progress(0, text=T['PASS2_TEXT'])
    jgri_values = calculate_jgri(df_raw_metrics)
    
    # Collect ALL unique POS tags for dynamic sidebar filtering
    unique_pos_options = get_unique_pos_options(corpus_data)
    
    for i, data in enumerate(corpus_data):
        script_distribution = analyze_script_distribution(data['Text'])
        kanji_density = analyze_kanji_density(data['Text'])
        pos_percentages, pos_counts = analyze_pos_distribution(data['Tagged_Nodes'], data['Filename'])

        text_tokenized = " ".join(data['Tokens'])
        lex = LexicalRichness(text_tokenized)
        total_tokens = lex.words; unique_tokens = lex.terms; ttr = lex.ttr
        # Safety for short texts
        hdd_value = lex.hdd(draws=min(42, total_tokens)) if total_tokens > 0 else None; 
        mtld_value = lex.mtld()
        jlpt_counts = analyze_jlpt_coverage(data['Tokens'], jlpt_dict_to_use)

        result = {
            "Filename": data['Filename'],
            "Jreadability": data.get("JREAD"),
            "JGRI": jgri_values[i],
            "MMS": data['MMS'],
            "LD": data['LD'],
            "VPS": data['VPS'],
            "MPN": data['MPN'],
            "Kanji_Density": kanji_density,
            "Script_Distribution": f"K: {script_distribution['Kanji']}% | H: {script_distribution['Hiragana']}% | T: {script_distribution['Katakana']}% | O: {script_distribution['Other']}%",
            "Tokens": total_tokens,
            "Types": unique_tokens,
            "TTR": ttr,
            "HDD": hdd_value,
            "MTLD": mtld_value,
            "JREAD_a": data.get("a"),
            "JREAD_b": data.get("b"),
            "JREAD_c": data.get("c"),
            "JREAD_d": data.get("d"),
            "JREAD_e": data.get("e"),
        }
        for level in ALL_OUTPUT_LEVELS:
            result[level.replace(" ", "_")] = jlpt_counts.get(level, 0)

        results.append(result)
        pos_percentage_results.append(pos_percentages)
        pos_count_results.append(pos_counts)
        progress_bar.progress((i + 1) / len(corpus_data), text=f"PASS 2: Completed analysis for {data['Filename']}.")

    progress_bar.empty(); st.success(T['ANALYSIS_COMPLETE'])
    df_results = pd.DataFrame(results)
if 'JREAD' in df_results.columns and 'Jreadability' not in df_results.columns:
    df_results.insert(
        df_results.columns.get_loc('JGRI'),
        'Jreadability',
        df_results['JREAD']
    )
    df_pos_percentage = pd.DataFrame(pos_percentage_results)

    # ===============================================
    # --- 3. N-gram Analysis Section ---
    # ===============================================
    
    st.header(T['NGRAM_MAIN_HEADER'])

    # --- Sidebar N-gram Control ---
    st.sidebar.header(T['NGRAM_HEADER'])
    n_gram_size = st.sidebar.radio(
        T['NGRAM_RADIO'],
        options=[1, 2, 3, 4, 5],
        index=0,
        key='n_gram_size_radio'
    )
    
    # --- Sidebar KWIC Context Control ---
    st.sidebar.markdown("---")
    st.sidebar.subheader(T['KWIC_CONTEXT_HEADER'])
    col_l, col_r = st.sidebar.columns(2)
    left_context_size = col_l.number_input(T['KWIC_LEFT'], min_value=1, max_value=20, value=7, key='left_context_size')
    right_context_size = col_r.number_input(T['KWIC_RIGHT'], min_value=1, max_value=20, value=7, key='right_context_size')
    
    st.markdown(f"{T['NGRAM_CURRENT']} {n_gram_size}-gram**")
    st.info(T['NGRAM_WILDCARD_INFO'])
    
    # 1. Generate ALL N-grams across the corpus
    all_n_grams_df = pd.DataFrame(columns=['N_gram', 'POS_Sequence'])
    for data in corpus_data:
        df_n = get_n_grams(data['Tagged_Nodes'], n_gram_size)
        df_n['Filename'] = data['Filename']
        # The frequency calculation only needs N_gram and POS_Sequence
        all_n_grams_df = pd.concat([all_n_grams_df, df_n.drop(columns=['Start_Index', 'Filename'])], ignore_index=True)
        
    df_n_gram_freq = calculate_n_gram_frequency(all_n_grams_df)

    # 2. Dynamic Filter UI
    st.markdown(f"##### {T['NGRAM_FILTER_HEADER']}")
    filter_cols = st.columns(n_gram_size)
    
    current_filters = {}
    
    # Create the filters dynamically
    for i in range(n_gram_size):
        with filter_cols[i]:
            # Word filter
            current_filters[f'word_{i}'] = st.text_input(
                label=f"Word {i+1}", 
                key=f'word_filter_{i}', 
                placeholder=f"Filter word {i+1}"
            )
            # POS filter (Dropdown)
            current_filters[f'pos_{i}'] = st.selectbox(
                label=f"POS {i+1}", 
                options=unique_pos_options, # This is generated once after PASS 2
                key=f'pos_filter_{i}'
            )
    
    # 3. Apply Filters for Frequency Table
    df_filtered_n_grams = apply_n_gram_filters(df_n_gram_freq, current_filters, n_gram_size)
    
    # 4. Display Frequency Results
    st.markdown("---")
    st.markdown(f"#### {T['NGRAM_FREQ_HEADER']}")
    st.markdown(f"{T['NGRAM_TOTAL_UNIQUE']} {n_gram_size}-gram{T['NGRAM_MATCHING_FILTER']} {len(df_filtered_n_grams):,}")
    
    st.dataframe(
        df_filtered_n_grams[['N_gram', 'Frequency', 'Percentage']].head(50), 
        use_container_width=True,
        height=300,
        column_config={
            "N_gram": st.column_config.Column(f"{n_gram_size}-gram", help=f"Sequence of words/morphemes."),
            "Frequency": st.column_config.NumberColumn("Frequency", help="Total count of this specific N-gram."),
            "Percentage": st.column_config.TextColumn("Percentage", help="Frequency relative to the total number of filtered N-grams."),
        }
    )
    
    # Download Button for N-gram list
    if not df_filtered_n_grams.empty:
        csv_n_grams = df_filtered_n_grams.to_csv(index=False).encode('utf-8')
        st.download_button(
            label=T['DOWNLOAD_NGRAM'].format(n=n_gram_size, count=len(df_filtered_n_grams)),
            data=csv_n_grams,
            file_name=f"{n_gram_size}_gram_frequency_full.csv",
            mime="text/csv"
        )
    
    st.markdown("---")

    # 5. Generate and Display Concordance
    st.markdown(f"#### {T['CONCORDANCE_HEADER']}")
    
    # Pass all filters and context sizes to generate the KWIC list
    df_concordance = generate_concordance(corpus_data, current_filters, n_gram_size, left_context_size, right_context_size)

    st.markdown(f"{T['CONCORDANCE_LINES']} {len(df_concordance):,} (based on N-gram filters)")

    # Display KWIC Table
    st.dataframe(
        df_concordance.head(500), # Show more lines for context, still capped by Streamlit
        use_container_width=True,
        height=400,
        column_config={
            "Filename": st.column_config.Column("File", width="small"),
            "Left Context": st.column_config.TextColumn(f"{T['KWIC_LEFT']} ({left_context_size})", help=f"{left_context_size} {T['KWIC_HELP_LEFT']}", width="large"),
            "Keyword(s)": st.column_config.TextColumn("Keyword(s)", help=f"{T['KWIC_HELP_KW']} {n_gram_size}-gram", width="large"),
            "Right Context": st.column_config.TextColumn(f"{T['KWIC_RIGHT']} ({right_context_size})", help=f"{right_context_size} {T['KWIC_HELP_RIGHT']}", width="large"),
        }
    )
    
    # Download Button for Concordance
    if not df_concordance.empty:
        csv_concordance = df_concordance.to_csv(index=False).encode('utf-8')
        st.download_button(
            label=T['DOWNLOAD_KWIC'].format(count=len(df_concordance)),
            data=csv_concordance,
            file_name="concordance_list_full.csv",
            mime="text/csv"
        )
    
    st.markdown("---")

    # ===============================================
    # --- 4. Visualizations (UPDATED) ---
    # ===============================================

    st.subheader(T['VISUAL_HEADER'])
    
    if len(df_results) >= 1:
        
        # --- Row 1: JLPT and Scripts ---
        col1, col2 = st.columns(2)
        
        with col1:
            jlpt_plot_file = "jlpt_coverage.png"
            plot_jlpt_coverage(df_results, filename=jlpt_plot_file)
            st.image(jlpt_plot_file, caption="JLPT Vocabulary Coverage (Proportion of Unique Words)")
            
        with col2:
            scripts_plot_file = "scripts_distribution.png"
            plot_scripts_distribution(df_results, filename=scripts_plot_file)
            st.image(scripts_plot_file, caption="Scripts Distribution (Kanji, Hiragana, Katakana, Other)")
            
        st.markdown("---")
        
        # --- Row 2: JGRI, MTLD, TTR ---
        col3, col4, col5 = st.columns(3)

        with col3:
            if len(df_results) > 1:
                jgri_plot_file = "jgri_comparison.png"
                plot_jgri_comparison(df_results, filename=jgri_plot_file)
                st.image(jgri_plot_file, caption="JGRI Comparison (Relative Grammatical Complexity)")
            else:
                st.info(T['JGRI_COMPARE_NOTE'])

        with col4:
            mtld_plot_file = "mtld_comparison.png"
            plot_mtld_comparison(df_results, filename=mtld_plot_file)
            st.image(mtld_plot_file, caption="MTLD Comparison (Lexical Diversity Score)")

        with col5:
            ttr_plot_file = "ttr_comparison.png"
            plot_ttr_comparison(df_results, filename=ttr_plot_file)
            st.image(ttr_plot_file, caption="Type-Token Ratio (TTR) Comparison")
            
        st.markdown("---")
        
        # --- Row 3: POS and Tokens ---
        col6, col7 = st.columns(2)

        with col6:
            pos_plot_file = "pos_comparison.png"
            plot_pos_comparison(df_pos_percentage, filename=pos_plot_file)
            st.image(pos_plot_file, caption="Normalized POS Distribution (Top 10 Categories)")
            
        with col7:
            token_count_plot_file = "token_count_comparison.png"
            plot_token_count_comparison(df_results, filename=token_count_plot_file)
            st.image(token_count_plot_file, caption="Total Token Count (Text Length)")
            
        st.markdown("---")
        
        # --- Row 4: Rolling TTR Curve (Now hidden in an Expander) ---
        with st.expander(T['ROLLING_TTR_EXPANDER']):
            st.markdown(T['ROLLING_TTR_NOTE'])
            rolling_ttr_plot_file = "rolling_ttr_curve.png"
            plot_rolling_ttr_curve(corpus_data, filename=rolling_ttr_plot_file)
            st.image(rolling_ttr_plot_file, caption="Rolling Mean TTR (Vocabulary Trend)")
            
        st.markdown("---")

        # --- Row 5: Word Cloud (NEW SECTION) ---
        st.markdown(f"#### {T['WORDCLOUD_HEADER']}")
        st.info(T['WORDCLOUD_NOTE'])
        word_cloud_file = "word_cloud.png"
        plot_word_cloud(corpus_data, filename=word_cloud_file)
        st.image(word_cloud_file, caption="Word Cloud (Token Frequency)")
        
        st.markdown("---")


    # ===============================================
    # --- 5. Summary Tables ---
    # ===============================================
    st.subheader(T['SUMMARY_HEADER'])
    
    # --- Define Column Configuration for Tooltips and Formatting ---
    column_configuration = {
        "Filename": st.column_config.TextColumn("Filename", help="Name of the uploaded text file."),
        "JGRI": st.column_config.NumberColumn("JGRI", format="%.3f", help="Japanese Grammatical Readability Index. Higher = More complex (relative to the corpus)."),
        "MMS": st.column_config.NumberColumn("MMS", format="%.2f", help="Mean Morphemes per Sentence. Raw value for sentence length/integration cost."),
        "LD": st.column_config.NumberColumn("LD", format="%.3f", help="Lexical Density (Content Words / Total Morphemes). Raw value for information load."),
        "VPS": st.column_config.NumberColumn("VPS", format="%.2f", help="Verbs per Sentence. Raw value for clause load and structural density."),
        "MPN": st.column_config.NumberColumn("MPN", format="%.2f", help="Modifiers per Noun. Raw value for Noun Phrase complexity."),
        "Kanji_Density": st.column_config.NumberColumn("Kanji Density", format="%.2f", help="Average number of Kanji characters per sentence."),
        "Script_Distribution": st.column_config.TextColumn("Script Distribution", help="Percentage breakdown of character types: K=Kanji, H=Hiragana, T=Katakana, O=Other."),
        "Tokens": st.column_config.NumberColumn("Tokens", help="Total count of all morphemes/words (N).", width="small"),
        "Types": st.column_config.NumberColumn("Types", help="Total count of unique morphemes/words (V).", width="small"),
        "TTR": st.column_config.NumberColumn("TTR", format="%.3f", help="Type-Token Ratio (V/N). Vocabulary diversity, highly sensitive to length.", width="small"),
        "HDD": st.column_config.NumberColumn("HDD", format="%.3f", help="Hellinger's D. Length-independent measure of lexical diversity."),
        "MTLD": st.column_config.NumberColumn("MTLD", format="%.1f", help="Measure of Textual Lexical Diversity. Robust, length-independent measure."),
        "JLPT_N5": st.column_config.NumberColumn("JLPT N5", help="Count of unique words covered by the N5 list.", width="small"),
        "JLPT_N4": st.column_config.NumberColumn("JLPT N4", help="Count of unique words covered by the N4 list.", width="small"),
        "JLPT_N3": st.column_config.NumberColumn("JLPT N3", help="Count of unique words covered by the N3 list.", width="small"),
        "JLPT_N2": st.column_config.NumberColumn("JLPT N2", help="Count of unique words covered by the N2 list.", width="small"),
        "JLPT_N1": st.column_config.NumberColumn("JLPT N1", help="Count of unique words covered by the N1 list.", width="small"),
        "NA": st.column_config.NumberColumn("NA", help="Count of unique words NOT covered by N5-N1 lists.", width="small"),
    }
    
    st.markdown(f"### {T['JGRI_EXP_HEADER']}")
    st.markdown("""
        The JGRI is a composite, corpus-relative index estimating the grammatical and morphosyntactic complexity of the text. **Higher values indicate greater structural complexity.**
        
        **1. Components (What it measures):**
        * **MMS** (Mean Morphemes per Sentence)
        * **LD** (Lexical Density)
        * **VPS** (Verbs per Sentence)
        * **MPN** (Modifiers per Noun)

        **2. Computation (How it's calculated):**
        1.  The raw value of each component is calculated for every text.
        2.  Each raw component is **z-score normalised** across the *entire uploaded corpus*.
        3.  The final JGRI score is the **average** of the four z-scores: $\mathbf{JGRI} = \frac{(\text{z-MMS} + \text{z-LD} + \text{z-VPS} + \text{z-MPN})}{4}$.
        
        **3. Interpretation Thresholds (What the score means):**
    """)
    
    # Display Interpretation Table using st.dataframe for clean rendering
    interpretation_data = {
        "JGRI Value": ["< -1.0", "-1.0 to 0", "0 to +1.0", "> +1.0"],
        "Interpretation": [
            "Very easy / Conversational", 
            "Relatively easy", 
            "Moderately complex", 
            "High grammatical complexity"
        ]
    }
    df_interpretation = pd.DataFrame(interpretation_data)
    st.dataframe(df_interpretation.set_index('JGRI Value'), use_container_width=True)
    
    st.markdown("---")

    # Filter columns to ensure consistent order (including all components)
    sorted_columns = ["Filename", "JGRI", "MMS", "LD", "VPS", "MPN", "Kanji_Density", "Script_Distribution", "Tokens", "Types", "TTR", "HDD", "MTLD"]
    for level in ALL_OUTPUT_LEVELS:
        sorted_columns.append(level.replace(" ", "_"))
        
    df_results = df_results[[col for col in sorted_columns if col in df_results.columns]]
    
    # Display the final table with column configuration (tooltips)
    st.dataframe(df_results, column_config=column_configuration, use_container_width=True)

    # --- 2C. DETAILED POS DISTRIBUTION TABLE ---
    st.subheader(T['POS_HEADER'])
    st.markdown(T['POS_NOTE'])
    
    df_pos_percentage = pd.DataFrame(pos_percentage_results)
    df_pos_percentage = df_pos_percentage.set_index('Filename').fillna(0).T 
    df_pos_percentage.columns.name = f"POS Distribution ({T['POS_HEADER']})"

    st.dataframe(df_pos_percentage.sort_index(), use_container_width=True, height=600)
    
    # --- 2D. RAW JGRI COMPONENTS TABLE (Keeping for debug/full data in Excel) ---
    with st.expander(T['RAW_JGRI_EXPANDER']):
        st.markdown(T['RAW_JGRI_NOTE'])
        df_raw_metrics.index.name = "Index" # Set index name for display
        st.dataframe(df_raw_metrics.set_index('Filename')[['MMS', 'LD', 'VPS', 'MPN']], use_container_width=True)


    # --- DOWNLOAD BUTTONS ---
    
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer: 
        # Using a version of df_results without Streamlit column configs for clean export
        df_export = df_results.rename(columns={
            "JGRI": "JGRI", "MMS": "MMS", "LD": "LD", "VPS": "VPS", "MPN": "MPN", 
            "Kanji_Density": "Kanji Density", "Script_Distribution": "Script Distribution", 
            "Tokens": "Tokens", "Types": "Types", "TTR": "TTR", "HDD": "HDD", "MTLD": "MTLD",
            "JLPT_N5": "JLPT N5", "JLPT_N4": "JLPT N4", "JLPT_N3": "JLPT N3", "JLPT_N2": "JLPT N2", "JLPT_N1": "JLPT N1", "NA": "NA"
        })
        df_export.to_excel(writer, index=False, sheet_name='Lexical Profile')
        df_pos_percentage.to_excel(writer, index=True, sheet_name='POS Distribution')
        df_raw_metrics.to_excel(writer, index=False, sheet_name='Raw JGRI Components')
        
        # Check if df_n_gram_freq exists before trying to export it
        if 'df_n_gram_freq' in locals():
            df_n_gram_freq.to_excel(writer, index=False, sheet_name=f'{n_gram_size}_gram_Frequency')
        
    st.download_button(
        label=T['DOWNLOAD_ALL'],
        data=output.getvalue(),
        file_name="lexical_profile_results_full.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )
    
else:
    st.header(T['UPLOAD_HEADER'])
    st.info(T['UPLD_TO_BEGIN'])
