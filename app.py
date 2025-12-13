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
        n_gram_words = " ".join(words[i:i + n])
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
# Other Helper Functions (Plotting, etc.)
# ===============================================

# (Plotting functions and analysis functions remain the same as the prior version for brevity)

# Placeholder/stubs for analysis functions (actual code remains unchanged)
def analyze_jgri_components(text, tagged_nodes): pass
def calculate_jgri(metrics_df): pass
def analyze_script_distribution(text): pass
def analyze_kanji_density(text): pass
def analyze_jlpt_coverage(tokens, jlpt_dict): pass
def analyze_pos_distribution(tagged_nodes, filename): pass
def plot_jlpt_coverage(df, filename="jlpt_coverage.png"): pass
def plot_jgri_comparison(df, filename="jgri_comparison.png"): pass
def plot_scripts_distribution(df, filename="scripts_distribution.png"): pass
def plot_mtld_comparison(df, filename="mtld_comparison.png"): pass
def plot_token_count_comparison(df, filename="token_count_comparison.png"): pass
def plot_rolling_ttr_curve(corpus_data, window_size=50, filename="rolling_ttr_curve.png"): pass
def plot_ttr_comparison(df, filename="ttr_comparison.png"): pass
def plot_pos_comparison(df_pos_percentage, filename="pos_comparison.png"): pass

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
            "Kanji_Density": kanji_density, "Script_Distribution": f"K: {script_distribution['Kanji']}% | H: {script_distribution['Hiragana']}% | T: {script_distribution['Katakana']}% | O: {script_distribution['Other']}%",
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
    # --- 3. N-gram Analysis Section ---
    # ===============================================
    
    st.header("3. N-gram Frequency Analysis & Concordance")

    # --- Sidebar N-gram Control ---
    st.sidebar.header("3. N-gram Settings")
    n_gram_size = st.sidebar.radio(
        "Select N-gram Length (N)",
        options=[1, 2, 3, 4, 5],
        index=0,
        key='n_gram_size_radio'
    )
    
    # --- Sidebar KWIC Context Control ---
    st.sidebar.markdown("---")
    st.sidebar.subheader("Concordance Context")
    col_l, col_r = st.sidebar.columns(2)
    left_context_size = col_l.number_input("Words to Left", min_value=1, max_value=20, value=7, key='left_context_size')
    right_context_size = col_r.number_input("Words to Right", min_value=1, max_value=20, value=7, key='right_context_size')
    
    st.markdown(f"**Current N-gram length selected: {n_gram_size}-gram**")
    st.info("Use the `*` symbol in the word filter boxes below to represent zero or more characters (e.g., `*ing` or `本*`).")
    
    # 1. Generate ALL N-grams across the corpus
    all_n_grams_df = pd.DataFrame(columns=['N_gram', 'POS_Sequence'])
    for data in corpus_data:
        df_n = get_n_grams(data['Tagged_Nodes'], n_gram_size)
        all_n_grams_df = pd.concat([all_n_grams_df, df_n], ignore_index=True)
        
    df_n_gram_freq = calculate_n_gram_frequency(all_n_grams_df)

    # 2. Dynamic Filter UI
    st.markdown("##### Filter N-grams by Word (with Wildcards) or Part-of-Speech (POS)")
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
    st.markdown("#### N-gram Frequency List")
    st.markdown(f"**Total unique {n_gram_size}-grams matching filter: {len(df_filtered_n_grams):,}**")
    
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
    
    # Download Button for N-gram list
    if not df_filtered_n_grams.empty:
        csv_n_grams = df_filtered_n_grams.to_csv(index=False).encode('utf-8')
        st.download_button(
            label=f"⬇️ Download Full Filtered {n_gram_size}-gram List ({len(df_filtered_n_grams):,} unique items)",
            data=csv_n_grams,
            file_name=f"{n_gram_size}_gram_frequency_full.csv",
            mime="text/csv"
        )
    
    st.markdown("---")

    # 5. Generate and Display Concordance
    st.markdown("#### Concordance (Keyword In Context - KWIC)")
    
    # Pass all filters and context sizes to generate the KWIC list
    df_concordance = generate_concordance(corpus_data, current_filters, n_gram_size, left_context_size, right_context_size)

    st.markdown(f"**Total concordance lines generated: {len(df_concordance):,}** (based on N-gram filters)")

    # Display KWIC Table
    st.dataframe(
        df_concordance.head(500), # Show more lines for context, still capped by Streamlit
        use_container_width=True,
        height=400,
        column_config={
            "Filename": st.column_config.Column("File", width="small"),
            "Left Context": st.column_config.TextColumn("Left Context", help=f"{left_context_size} words to the left", width="large"),
            "Keyword(s)": st.column_config.TextColumn("Keyword(s)", help=f"The filtered {n_gram_size}-gram", width="large"),
            "Right Context": st.column_config.TextColumn("Right Context", help=f"{right_context_size} words to the right", width="large"),
        }
    )
    
    # Download Button for Concordance
    if not df_concordance.empty:
        csv_concordance = df_concordance.to_csv(index=False).encode('utf-8')
        st.download_button(
            label=f"⬇️ Download Full Concordance List ({len(df_concordance):,} lines)",
            data=csv_concordance,
            file_name="concordance_list_full.csv",
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
        
        # NOTE: Using placeholder functions for plotting logic since they are unchanged from the last step.
        with col1:
            jlpt_plot_file = "jlpt_coverage.png" # Placeholder
            plot_jlpt_coverage(df_results, filename=jlpt_plot_file)
            st.image(jlpt_plot_file, caption="JLPT Vocabulary Coverage (Proportion of Unique Words)")
            
        with col2:
            scripts_plot_file = "scripts_distribution.png" # Placeholder
            plot_scripts_distribution(df_results, filename=scripts_plot_file)
            st.image(scripts_plot_file, caption="Scripts Distribution (Kanji, Hiragana, Katakana, Other)")
            
        st.markdown("---")
        
        # --- Row 2: JGRI, MTLD, TTR ---
        col3, col4, col5 = st.columns(3)

        with col3:
            if len(df_results) > 1:
                jgri_plot_file = "jgri_comparison.png" # Placeholder
                plot_jgri_comparison(df_results, filename=jgri_plot_file)
                st.image(jgri_plot_file, caption="JGRI Comparison (Relative Grammatical Complexity)")
            else:
                st.info("JGRI comparison requires at least two files.")

        with col4:
            mtld_plot_file = "mtld_comparison.png" # Placeholder
            plot_mtld_comparison(df_results, filename=mtld_plot_file)
            st.image(mtld_plot_file, caption="MTLD Comparison (Lexical Diversity Score)")

        with col5:
            ttr_plot_file = "ttr_comparison.png" # Placeholder
            plot_ttr_comparison(df_results, filename=ttr_plot_file)
            st.image(ttr_plot_file, caption="Type-Token Ratio (TTR) Comparison")
            
        st.markdown("---")
        
        # --- Row 3: POS and Tokens ---
        col6, col7 = st.columns(2)

        with col6:
            pos_plot_file = "pos_comparison.png" # Placeholder
            plot_pos_comparison(df_pos_percentage, filename=pos_plot_file)
            st.image(pos_plot_file, caption="Normalized POS Distribution (Top 10 Categories)")
            
        with col7:
            token_count_plot_file = "token_count_comparison.png" # Placeholder
            plot_token_count_comparison(df_results, filename=token_count_plot_file)
            st.image(token_count_plot_file, caption="Total Token Count (Text Length)")
        
        st.markdown("---")

        # --- Row 4: Rolling TTR Curve (Now hidden in an Expander) ---
        with st.expander("Show Rolling Mean TTR Curve (Vocabulary Trend)"):
            st.markdown("This plot shows the trend of vocabulary diversity over the length of the text. A flat, high line indicates sustained rich vocabulary.")
            rolling_ttr_plot_file = "rolling_ttr_curve.png" # Placeholder
            plot_rolling_ttr_curve(corpus_data, filename=rolling_ttr_plot_file)
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
        label="⬇️ Download All Results as Excel (Includes N-gram Data)",
        data=output.getvalue(),
        file_name="lexical_profile_results_full.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )
    
else:
    st.header("Upload Files to Begin")
    st.info("Please upload your Japanese text files (.txt) using the **sidebar**.")
