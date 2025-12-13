import streamlit as st
import pandas as pd
import io
import os
import re
from collections import Counter
import numpy as np
import matplotlib.pyplot as plt 
import matplotlib.ticker as ticker 

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

# --- Layout and Title ---
st.set_page_config(
    page_title="🇯🇵 Japanese Lexical Profiler",
    layout="wide"
)

st.title("🇯🇵 Japanese Lexical Profiler")
st.markdown("Analyze lexical richness, **structural complexity**, and JLPT word coverage.")

# ===============================================
# Helper Functions - Caching for Performance
# ===============================================

@st.cache_data(show_spinner="Loading JLPT Wordlists from CSVs...")
def load_jlpt_wordlist():
    """Loads all five JLPT wordlists."""
    jlpt_dict = {}
    for level_name, filename in JLPT_FILE_MAP.items():
        if not os.path.exists(filename):
            st.error(f"Required CSV file '{filename}' not found in the repository root.")
            return None
        try:
            df = pd.read_csv(filename, header=0, encoding='utf-8', keep_default_na=False)
            if df.empty:
                 words = set()
            else:
                 word_column = df.columns[0]
                 words = set(df[word_column].astype(str).tolist())
            jlpt_dict[level_name] = words
        except Exception as e:
            st.error(f"Error reading CSV file '{filename}': {e}")
            return None
    st.success("JLPT Wordlists loaded successfully from CSVs!")
    return jlpt_dict

@st.cache_resource(show_spinner="Initializing Fugashi Tokenizer...")
def initialize_tokenizer():
    """Initializes the Fugashi Tagger."""
    try:
        tagger = Tagger()
        st.success("Fugashi tokenizer loaded successfully!")
        return tagger
    except Exception as e:
        st.error(f"Error initializing Fugashi: {e}")
        st.error("Please ensure 'unidic-lite' is in your requirements.txt to fix MeCab initialization.")
        st.stop()
        return None

# ===============================================
# Core N-gram Analysis (NEW)
# ===============================================

def get_n_grams(tagged_nodes, n):
    """
    Extracts n-grams (sequences of words) and their corresponding POS tag sequences.
    """
    words = [node.surface for node in tagged_nodes if node.surface]
    pos_tags = [node.feature.pos1 for node in tagged_nodes if node.feature.pos1]
    
    n_grams = []
    n_gram_pos = []
    
    # We must iterate up to len(words) - n + 1
    for i in range(len(words) - n + 1):
        # Join words with spaces for display
        n_gram_words = " ".join(words[i:i + n])
        # Join POS tags with underscores for filtering
        n_gram_pos_sequence = "_".join(pos_tags[i:i + n])
        
        n_grams.append(n_gram_words)
        n_gram_pos.append(n_gram_pos_sequence)
        
    return pd.DataFrame({'N_gram': n_grams, 'POS_Sequence': n_gram_pos})

def calculate_n_gram_frequency(df_n_grams):
    """Calculates frequency and percentage for all unique N-grams."""
    
    if df_n_grams.empty:
        return pd.DataFrame(columns=['N_gram', 'Frequency', 'Percentage', 'POS_Sequence'])

    # Group by both N_gram (words) and POS_Sequence
    df_freq = df_n_grams.groupby(['N_gram', 'POS_Sequence']).size().reset_index(name='Frequency')
    total_grams = df_freq['Frequency'].sum()
    
    # Calculate percentage
    df_freq['Percentage'] = (df_freq['Frequency'] / total_grams) * 100
    
    df_freq = df_freq.sort_values(by='Frequency', ascending=False).reset_index(drop=True)
    
    # Format columns for display
    df_freq['Percentage'] = df_freq['Percentage'].map('{:.3f}%'.format)
    
    return df_freq

def get_unique_pos_options(corpus_data):
    """Collects all unique POS tags across the entire corpus."""
    all_pos = set()
    for data in corpus_data:
        # data['Tagged_Nodes'] is a list of node objects
        for node in data['Tagged_Nodes']:
            if node.feature.pos1:
                all_pos.add(node.feature.pos1)
    
    global POS_OPTIONS
    POS_OPTIONS = sorted(list(all_pos))
    return ['(Any)'] + POS_OPTIONS

# ===============================================
# Filtering Logic (NEW)
# ===============================================

def apply_n_gram_filters(df_freq, filters, n):
    """
    Filters the N-gram DataFrame based on user inputs (words and POS tags).
    """
    df_filtered = df_freq.copy()
    
    for i in range(n):
        word_filter = filters.get(f'word_{i}', '').strip()
        pos_filter = filters.get(f'pos_{i}', '(Any)').strip()
        
        # 1. Word Filtering
        if word_filter:
            # We must filter the N_gram column (space-separated words)
            def filter_by_word(row, idx, search_term):
                words = row['N_gram'].split(' ')
                return words[idx] == search_term
            
            df_filtered = df_filtered[df_filtered.apply(
                lambda row: filter_by_word(row, i, word_filter), axis=1
            )]

        # 2. POS Filtering
        if pos_filter != '(Any)':
            # We must filter the POS_Sequence column (underscore-separated tags)
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

# ===============================================
# Other Helper Functions (Plotting, etc.)
# ===============================================
# (JGRI, JLPT, Script, TTR, etc. functions remain the same)
def analyze_jgri_components(text, tagged_nodes):
    pos_counts = Counter(node.feature.pos1 for node in tagged_nodes if node.surface and node.feature.pos1)
    Nouns = pos_counts.get('名詞', 0); Verbs = pos_counts.get('動詞', 0); Adjectives = pos_counts.get('形容詞', 0); Adverbs = pos_counts.get('副詞', 0)
    Total_Morphemes = len(tagged_nodes)
    sentences = re.split(r'[。！？\n]', text.strip()); sentences = [s.strip() for s in sentences if s.strip()]; Num_Sentences = len(sentences)
    if Total_Morphemes == 0 or Nouns == 0 or Num_Sentences == 0:
        return {'MMS': 0.0, 'LD': 0.0, 'VPS': 0.0, 'MPN': 0.0}
    MMS = Total_Morphemes / Num_Sentences
    LD = (Nouns + Verbs + Adjectives + Adverbs) / Total_Morphemes
    VPS = Verbs / Num_Sentences
    MPN = (Adjectives + Verbs) / Nouns
    return {'MMS': MMS, 'LD': LD, 'VPS': VPS, 'MPN': MPN}

def calculate_jgri(metrics_df):
    jgri_values = []; mu = metrics_df[['MMS', 'LD', 'VPS', 'MPN']].mean(); sigma = metrics_df[['MMS', 'LD', 'VPS', 'MPN']].std()
    sigma = sigma.replace(0, 1e-6) 
    for index, row in metrics_df.iterrows():
        raw_values = row[['MMS', 'LD', 'VPS', 'MPN']]
        z_mms = (raw_values['MMS'] - mu['MMS']) / sigma['MMS']; z_ld = (raw_values['LD'] - mu['LD']) / sigma['LD']
        z_vps = (raw_values['VPS'] - mu['VPS']) / sigma['VPS']; z_mpn = (raw_values['MPN'] - mu['MPN']) / sigma['MPN']
        jgri = (z_mms + z_ld + z_vps + z_mpn) / 4; jgri_values.append(round(jgri, 3))
    return jgri_values

def analyze_script_distribution(text):
    total_chars = len(text)
    if total_chars == 0:
        return {"Kanji": 0, "Hiragana": 0, "Katakana": 0, "Other": 0}
    patterns = {"Kanji": r'[\u4E00-\u9FFF]', "Hiragana": r'[\u3040-\u309F]', "Katakana": r'[\u30A0-\u30FF]',}
    counts = {name: len(re.findall(pattern, text)) for name, pattern in patterns.items()}
    counted_chars = sum(counts.values()); counts["Other"] = total_chars - counted_chars
    percentages = {name: round((count / total_chars) * 100, 1) for name, count in counts.items()}
    return percentages

def analyze_kanji_density(text):
    sentences = re.split(r'[。！？\n]', text.strip()); sentences = [s.strip() for s in sentences if s.strip()]
    if not sentences: return 0.0
    total_kanji = len(re.findall(r'[\u4E00-\u9FFF]', text)); num_sentences = len(sentences)
    density = total_kanji / num_sentences; return round(density, 2)

def analyze_jlpt_coverage(tokens, jlpt_dict):
    unique_tokens_in_text = set(tokens); result = {}; total_known_words = set()
    for level, wordset in jlpt_dict.items():
        count = sum(1 for w in unique_tokens_in_text if w in wordset); result[level] = count
        total_known_words.update(w for w in unique_tokens_in_text if w in wordset)
    na_count = len(unique_tokens_in_text) - len(total_known_words); result["NA"] = na_count
    return result

def analyze_pos_distribution(tagged_nodes, filename):
    if not tagged_nodes: return {"Filename": filename}, {"Filename": filename}
    pos_tags = [node.feature.pos1 for node in tagged_nodes if node.surface and node.feature.pos1]
    if not pos_tags: return {"Filename": filename}, {"Filename": filename}
    total_tokens = len(pos_tags); pos_counts = Counter(pos_tags)
    pos_percentages = {"Filename": filename}; pos_raw_counts = {"Filename": filename}
    for tag, count in pos_counts.items():
        percentage = round((count / total_tokens) * 100, 1); pos_percentages[tag] = percentage; pos_raw_counts[tag] = count
    return pos_percentages, pos_raw_counts

def plot_jlpt_coverage(df, filename="jlpt_coverage.png"):
    df_plot = df[['Filename', 'JLPT_N5', 'JLPT_N4', 'JLPT_N3', 'JLPT_N2', 'JLPT_N1', 'NA']].copy(); df_plot['Total_Types'] = df_plot.iloc[:, 1:].sum(axis=1)
    for col in df_plot.columns[1:-1]: df_plot[col] = (df_plot[col] / df_plot['Total_Types']) * 100
    df_plot = df_plot.set_index('Filename').drop(columns='Total_Types')
    colors = {'JLPT_N5': '#51A3A3', 'JLPT_N4': '#51C4D4','JLPT_N3': '#FFD000','JLPT_N2': '#FFA500','JLPT_N1': '#FF6347','NA': '#8B0000'}
    fig, ax = plt.subplots(figsize=(10, 6))
    df_plot.plot(kind='barh', stacked=True, color=[colors[col] for col in df_plot.columns], ax=ax)
    ax.set_title("JLPT Vocabulary Coverage (Proportion of Unique Words)", fontsize=14); ax.set_xlabel("Percentage of Unique Words (%)", fontsize=12)
    ax.set_ylabel("Text File", fontsize=12); ax.legend(title="Vocabulary Level", bbox_to_anchor=(1.05, 1), loc='upper left'); plt.tight_layout()
    plt.savefig(filename); plt.close(fig); return filename

def plot_jgri_comparison(df, filename="jgri_comparison.png"):
    df_plot = df[['Filename', 'JGRI']].set_index('Filename')
    colors = ['#1f77b4' if x >= 0 else '#d62728' for x in df_plot['JGRI']]
    fig, ax = plt.subplots(figsize=(10, 6))
    df_plot['JGRI'].plot(kind='bar', color=colors, ax=ax); ax.axhline(0, color='gray', linestyle='--')
    ax.set_title("JGRI Comparison (Relative Grammatical Complexity)", fontsize=14); ax.set_xlabel("Text File", fontsize=12)
    ax.set_ylabel("JGRI Score (Z-Score Average)", fontsize=12); ax.tick_params(axis='x', rotation=45)
    plt.tight_layout(); plt.savefig(filename); plt.close(fig); return filename

def plot_scripts_distribution(df, filename="scripts_distribution.png"):
    df_scripts = pd.DataFrame()
    for index, row in df.iterrows():
        parts = row['Script_Distribution'].split(' | ')
        data = {p.split(': ')[0].strip(): float(p.split(': ')[1].replace('%', '').strip()) for p in parts}
        df_scripts = pd.concat([df_scripts, pd.DataFrame([data], index=[row['Filename']])])
    script_cols = ['K', 'H', 'T', 'O']; df_scripts = df_scripts[script_cols].fillna(0)
    colors = {'K': '#483D8B', 'H': '#8A2BE2', 'T': '#DA70D6', 'O': '#A9A9A9'}
    fig, ax = plt.subplots(figsize=(10, 6))
    df_scripts.plot(kind='barh', stacked=True, color=[colors[col] for col in df_scripts.columns], ax=ax)
    ax.set_title("Script Distribution (Percentage of Characters)", fontsize=14); ax.set_xlabel("Percentage (%)", fontsize=12)
    ax.set_ylabel("Text File", fontsize=12); ax.legend(['Kanji', 'Hiragana', 'Katakana', 'Other'], title="Script Type", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout(); plt.savefig(filename); plt.close(fig); return filename

def plot_mtld_comparison(df, filename="mtld_comparison.png"):
    df_plot = df[['Filename', 'MTLD']].set_index('Filename'); fig, ax = plt.subplots(figsize=(10, 6))
    df_plot['MTLD'].plot(kind='bar', color='#3CB371', ax=ax)
    ax.set_title("MTLD Comparison (Lexical Diversity)", fontsize=14); ax.set_xlabel("Text File", fontsize=12)
    ax.set_ylabel("MTLD Score", fontsize=12); ax.tick_params(axis='x', rotation=45)
    plt.tight_layout(); plt.savefig(filename); plt.close(fig); return filename

def plot_token_count_comparison(df, filename="token_count_comparison.png"):
    df_plot = df[['Filename', 'Tokens']].set_index('Filename'); fig, ax = plt.subplots(figsize=(10, 6))
    df_plot['Tokens'].plot(kind='bar', color='#6A5ACD', ax=ax)
    ax.set_title("Total Token Count Comparison", fontsize=14); ax.set_xlabel("Text File", fontsize=12); ax.set_ylabel("Total Tokens (Words)", fontsize=12)
    ax.tick_params(axis='x', rotation=45); formatter = ticker.FuncFormatter(lambda x, p: format(int(x), ','))
    ax.yaxis.set_major_formatter(formatter); plt.tight_layout(); plt.savefig(filename); plt.close(fig); return filename

def plot_rolling_ttr_curve(corpus_data, window_size=50, filename="rolling_ttr_curve.png"):
    fig, ax = plt.subplots(figsize=(10, 6)); is_data_plotted = False
    for data in corpus_data:
        tokens = data['Tokens']; filename_label = data['Filename']
        if not tokens or len(tokens) < window_size: continue
        ttr_values = []
        for i in range(len(tokens) - window_size + 1):
            window = tokens[i:i + window_size]; ttr = len(set(window)) / window_size
            ttr_values.append(ttr)
        x_axis = np.arange(window_size, len(tokens) + 1)
        ax.plot(x_axis, ttr_values, label=filename_label); is_data_plotted = True
    if not is_data_plotted:
        ax.text(0.5, 0.5, f"No texts long enough for window size {window_size}.", 
                transform=ax.transAxes, ha='center', color='red')
    ax.set_title(f"Rolling Mean TTR Curve (Window Size: {window_size})", fontsize=14); ax.set_xlabel("Tokens (Total Words)", fontsize=12)
    ax.set_ylabel("Rolling TTR (0 to 1)", fontsize=12); ax.legend(title="Text File", loc='upper right'); ax.set_ylim(0, 1)
    formatter = ticker.FuncFormatter(lambda x, p: format(int(x), ','))
    ax.xaxis.set_major_formatter(formatter); plt.tight_layout(); plt.savefig(filename); plt.close(fig); return filename

def plot_ttr_comparison(df, filename="ttr_comparison.png"):
    df_plot = df[['Filename', 'TTR']].set_index('Filename'); fig, ax = plt.subplots(figsize=(10, 6))
    df_plot['TTR'].plot(kind='bar', color='#FF8C00', ax=ax); ax.set_title("Type-Token Ratio (TTR) Comparison", fontsize=14)
    ax.set_xlabel("Text File", fontsize=12); ax.set_ylabel("TTR Score (0-1)", fontsize=12); ax.tick_params(axis='x', rotation=45)
    ax.set_ylim(0, df_plot['TTR'].max() * 1.1); plt.tight_layout(); plt.savefig(filename); plt.close(fig); return filename

def plot_pos_comparison(df_pos_percentage, filename="pos_comparison.png"):
    df_plot = df_pos_percentage.set_index('Filename').copy()
    all_tags = df_plot.columns.tolist(); total_tag_percentage = df_plot.mean().sort_values(ascending=False)
    top_tags = total_tag_percentage.head(10).index.tolist()
    df_plot_top = df_plot[top_tags]
    cmap = plt.cm.get_cmap('tab20', len(top_tags))
    fig, ax = plt.subplots(figsize=(10, 6))
    df_plot_top.plot(kind='barh', stacked=True, colormap=cmap, ax=ax)
    ax.set_title("Normalized Part-of-Speech Distribution (Top 10 Categories)", fontsize=14); ax.set_xlabel("Percentage (%)", fontsize=12)
    ax.set_ylabel("Text File", fontsize=12); ax.legend(title="POS Category", bbox_to_anchor=(1.05, 1), loc='upper left'); plt.tight_layout()
    plt.savefig(filename); plt.close(fig); return filename


# ===============================================
# Sidebar & Initialization
# ===============================================

# Load essential components
jlpt_dict_to_use = load_jlpt_wordlist()
tagger = initialize_tokenizer()

if jlpt_dict_to_use is None or tagger is None:
    st.stop() 

# --- Sidebar Configuration: DOC LINK AT TOP ---
st.sidebar.markdown(
    "**Documentation:** [User Guide and Results Interpretation](https://docs.google.com/document/d/1wFPY_b90K0NjS6dQEHJsjJDD_ZRbckq6vzY-kqMT9kE/edit?usp=sharing)"
)
st.sidebar.markdown("---")

st.sidebar.header("1. Upload Raw Text Files")

input_files = st.sidebar.file_uploader(
    "Upload one or more **.txt** files for analysis.",
    type=["txt"],
    accept_multiple_files=True,
    key="input_uploader",
    help="The files will be analyzed against the single pre-loaded JLPT word list."
)

st.sidebar.header("2. Word List Used")
st.sidebar.info(f"Using the pre-loaded **Unknown Source** list ({len(ALL_JLPT_LEVELS)} levels).")

# ===============================================
# Main Area: Process and Display
# ===============================================

results = []
pos_percentage_results = []
pos_count_results = []
corpus_data = [] 

if input_files:
    
    # --- PASS 1 & 2: Data Processing ---
    # (Identical to previous two-pass script, populates corpus_data, results, pos_percentage_results)
    
    st.header("2. Analysis Results")
    st.markdown("Coverage columns report the count of **unique words** from the text found in that category.")
    
    progress_bar = st.progress(0, text="--- PASS 1: Analyzing components and raw metrics ---")
    
    for i, uploaded_file in enumerate(input_files):
        filename = uploaded_file.name
        content_bytes = uploaded_file.read()
        try:
             text = content_bytes.decode('utf-8')
        except UnicodeDecodeError:
             st.error(f"Failed to decode {filename}. Ensure it is UTF-8 encoded.")
             progress_bar.progress((i + 1) / len(input_files))
             continue
             
        text = text.strip()
        if not text:
            st.warning(f"File {filename} is empty, skipped.")
            progress_bar.progress((i + 1) / len(input_files))
            continue
        
        tagged_nodes = list(tagger(text))
        jgri_raw_components = analyze_jgri_components(text, tagged_nodes)
        
        corpus_data.append({
            'Filename': filename,
            'Text': text,
            'Tagged_Nodes': tagged_nodes,
            'Tokens': [word.surface for word in tagged_nodes],
            **jgri_raw_components
        })
        progress_bar.progress((i + 1) / len(input_files), text=f"PASS 1: Analyzed {i+1} of {len(input_files)} files.")

    if not corpus_data:
        progress_bar.empty(); st.error("No valid text files were processed."); st.stop()

    df_raw_metrics = pd.DataFrame(corpus_data)
    progress_bar.progress(0, text="--- PASS 2: Calculating JGRI and final results ---")
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
        hdd_value = lex.hdd(draws=min(42, total_tokens)) if total_tokens > 0 else None; mtld_value = lex.mtld()
        jlpt_counts = analyze_jlpt_coverage(data['Tokens'], jlpt_dict_to_use)

        result = {
            "Filename": data['Filename'], "JGRI": jgri_values[i], "MMS": data['MMS'], "LD": data['LD'], "VPS": data['VPS'], "MPN": data['MPN'],
            "Kanji_Density": kanji_density, "Script_Distribution": f"K: {script_distribution['Kanji']}% | H: {script_distribution['Hiragana']}% | T: {script_exposure['Katakana']}% | O: {script_distribution['Other']}%",
            "Tokens": total_tokens, "Types": unique_tokens, "TTR": ttr, "HDD": hdd_value, "MTLD": mtld_value,
        }
        for level in ALL_OUTPUT_LEVELS:
            result[level.replace(" ", "_")] = jlpt_counts.get(level, 0)

        results.append(result)
        pos_percentage_results.append(pos_percentages)
        pos_count_results.append(pos_counts)
        progress_bar.progress((i + 1) / len(corpus_data), text=f"PASS 2: Completed analysis for {data['Filename']}.")

    progress_bar.empty(); st.success("Analysis complete!")
    df_results = pd.DataFrame(results)
    df_pos_percentage = pd.DataFrame(pos_percentage_results)

    # ===============================================
    # --- 3. N-gram Analysis Section (NEW) ---
    # ===============================================
    
    st.header("3. N-gram Frequency Analysis")

    # --- Sidebar N-gram Control ---
    st.sidebar.header("3. N-gram Settings")
    n_gram_size = st.sidebar.radio(
        "Select N-gram Length (N)",
        options=[1, 2, 3, 4, 5],
        index=0,
        key='n_gram_size_radio'
    )
    
    st.markdown(f"**Current N-gram length selected: {n_gram_size}-gram**")
    
    # 1. Generate ALL N-grams across the corpus
    all_n_grams_df = pd.DataFrame(columns=['N_gram', 'POS_Sequence'])
    for data in corpus_data:
        df_n = get_n_grams(data['Tagged_Nodes'], n_gram_size)
        all_n_grams_df = pd.concat([all_n_grams_df, df_n], ignore_index=True)
        
    df_n_gram_freq = calculate_n_gram_frequency(all_n_grams_df)

    # 2. Dynamic Filter UI
    st.markdown("##### Filter N-grams by Word or Part-of-Speech (POS)")
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
    
    # 3. Apply Filters
    df_filtered_n_grams = apply_n_gram_filters(df_n_gram_freq, current_filters, n_gram_size)
    
    # 4. Display Results
    st.markdown(f"**Total unique {n_gram_size}-grams: {len(df_filtered_n_grams):,}**")
    
    # Display table (max 50 rows)
    st.dataframe(
        df_filtered_n_grams[['N_gram', 'Frequency', 'Percentage']].head(50), 
        use_container_width=True,
        height=300,
        column_config={
            "N_gram": st.column_config.Column(f"{n_gram_size}-gram", help="Sequence of words/morphemes."),
            "Frequency": st.column_config.NumberColumn("Frequency", help="Total count of this specific N-gram."),
            "Percentage": st.column_config.TextColumn("Percentage", help="Frequency relative to the total number of filtered N-grams."),
        }
    )
    
    # 5. Download Button
    if not df_filtered_n_grams.empty:
        csv_n_grams = df_filtered_n_grams.to_csv(index=False).encode('utf-8')
        st.download_button(
            label=f"⬇️ Download Full Filtered {n_gram_size}-gram List ({len(df_filtered_n_grams):,} unique items)",
            data=csv_n_grams,
            file_name=f"{n_gram_size}_gram_frequency_full.csv",
            mime="text/csv"
        )
    
    st.markdown("---")

    # ===============================================
    # --- 4. Visualizations ---
    # ===============================================

    st.subheader("4. Visualizations")
    
    if len(df_results) >= 1:
        
        # --- Row 1: JLPT and Scripts ---
        col1, col2 = st.columns(2)
        
        with col1:
            jlpt_plot_file = plot_jlpt_coverage(df_results, filename="jlpt_coverage.png")
            st.image(jlpt_plot_file, caption="JLPT Vocabulary Coverage (Proportion of Unique Words)")
            
        with col2:
            scripts_plot_file = plot_scripts_distribution(df_results, filename="scripts_distribution.png")
            st.image(scripts_plot_file, caption="Scripts Distribution (Kanji, Hiragana, Katakana, Other)")
            
        st.markdown("---")
        
        # --- Row 2: JGRI, MTLD, TTR ---
        col3, col4, col5 = st.columns(3)

        with col3:
            if len(df_results) > 1:
                jgri_plot_file = plot_jgri_comparison(df_results, filename="jgri_comparison.png")
                st.image(jgri_plot_file, caption="JGRI Comparison (Relative Grammatical Complexity)")
            else:
                st.info("JGRI comparison requires at least two files.")

        with col4:
            mtld_plot_file = plot_mtld_comparison(df_results, filename="mtld_comparison.png")
            st.image(mtld_plot_file, caption="MTLD Comparison (Lexical Diversity Score)")

        with col5:
            ttr_plot_file = plot_ttr_comparison(df_results, filename="ttr_comparison.png")
            st.image(ttr_plot_file, caption="Type-Token Ratio (TTR) Comparison")
            
        st.markdown("---")
        
        # --- Row 3: POS and Tokens ---
        col6, col7 = st.columns(2)

        with col6:
            pos_plot_file = plot_pos_comparison(df_pos_percentage, filename="pos_comparison.png")
            st.image(pos_plot_file, caption="Normalized POS Distribution (Top 10 Categories)")
            
        with col7:
            token_count_plot_file = plot_token_count_comparison(df_results, filename="token_count_comparison.png")
            st.image(token_count_plot_file, caption="Total Token Count (Text Length)")
        
        st.markdown("---")

        # --- Row 4: Rolling TTR Curve (Now hidden in an Expander) ---
        with st.expander("Show Rolling Mean TTR Curve (Vocabulary Trend)"):
            st.markdown("This plot shows the trend of vocabulary diversity over the length of the text. A flat, high line indicates sustained rich vocabulary.")
            rolling_ttr_plot_file = plot_rolling_ttr_curve(corpus_data, filename="rolling_ttr_curve.png")
            st.image(rolling_ttr_plot_file, caption="Rolling Mean TTR (Vocabulary Trend)")
        
        st.markdown("---")


    # ===============================================
    # --- 5. Summary Tables ---
    # ===============================================
    st.subheader("5. Summary Table (Lexical, Structural & Readability Metrics)")
    
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
    
    st.markdown("""
        ### JGRI (Japanese Grammatical Readability Index) Explanation
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
    st.subheader("6. Detailed Part-of-Speech (POS) Distribution")
    st.markdown("This table shows the percentage of **all** detected grammatical categories for each file.")
    
    df_pos_percentage = pd.DataFrame(pos_percentage_results)
    df_pos_percentage = df_pos_percentage.set_index('Filename').fillna(0).T 
    df_pos_percentage.columns.name = "POS Distribution (%)"

    st.dataframe(df_pos_percentage.sort_index(), use_container_width=True, height=600)
    
    # --- 2D. RAW JGRI COMPONENTS TABLE (Keeping for debug/full data in Excel) ---
    with st.expander("Show Raw JGRI Components (Original Data for MMS, LD, VPS, MPN)"):
        st.markdown("This table provides the original raw values used to calculate the JGRI index. These values are also in the main table.")
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
        df_n_gram_freq.to_excel(writer, index=False, sheet_name=f'{n_gram_size}_gram_Frequency')
        
    st.download_button(
        label="⬇️ Download All Results as Excel (Includes N-gram Data)",
        data=output.getvalue(),
        file_name="lexical_profile_results_full.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )
    
else:
    st.header("Upload Files to Begin")
    st.info("Please upload your Japanese text files (.txt) using the **sidebar**.")
