import streamlit as st
import pandas as pd
import io
import os
import re
from collections import Counter
import numpy as np
import matplotlib.pyplot as plt # <-- New Import

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
    """
    Loads all five JLPT wordlists from local CSV files using pd.read_csv for speed.
    """
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
# Helper: JGRI Component Analysis
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

# ===============================================
# Helper: Plotting Functions (NEW)
# ===============================================

def plot_jlpt_coverage(df, filename="jlpt_coverage.png"):
    """Creates a normalized stacked bar chart of JLPT coverage."""
    
    # 1. Prepare data (normalize to 100%)
    df_plot = df[['Filename', 'JLPT_N5', 'JLPT_N4', 'JLPT_N3', 'JLPT_N2', 'JLPT_N1', 'NA']].copy()
    
    # Calculate total unique words (Types) for normalization
    df_plot['Total_Types'] = df_plot.iloc[:, 1:].sum(axis=1)
    
    # Normalize counts to percentages (0-100)
    for col in df_plot.columns[1:-1]: # Skip Filename and Total_Types
        df_plot[col] = (df_plot[col] / df_plot['Total_Types']) * 100

    df_plot = df_plot.set_index('Filename').drop(columns='Total_Types')

    # Define colors (using a spectrum from light to dark/red for difficulty)
    colors = {
        'JLPT_N5': '#51A3A3', 
        'JLPT_N4': '#51C4D4',
        'JLPT_N3': '#FFD000',
        'JLPT_N2': '#FFA500',
        'JLPT_N1': '#FF6347',
        'NA': '#8B0000' 
    }
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot normalized stacked bar chart
    df_plot.plot(kind='barh', stacked=True, color=[colors[col] for col in df_plot.columns], ax=ax)
    
    # Formatting
    ax.set_title("Normalized JLPT Vocabulary Coverage (%)", fontsize=14)
    ax.set_xlabel("Percentage of Unique Words (%)", fontsize=12)
    ax.set_ylabel("Text File", fontsize=12)
    ax.legend(title="Vocabulary Level", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    
    plt.savefig(filename)
    plt.close(fig)
    return filename

def plot_jgri_comparison(df, filename="jgri_comparison.png"):
    """Creates a bar chart comparing JGRI scores across texts."""
    
    df_plot = df[['Filename', 'JGRI']].set_index('Filename')
    
    # Define colors: Positive/Negative relative to 0
    colors = ['#1f77b4' if x >= 0 else '#d62728' for x in df_plot['JGRI']]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    df_plot['JGRI'].plot(kind='bar', color=colors, ax=ax)
    
    # Add horizontal line at 0 (corpus mean)
    ax.axhline(0, color='gray', linestyle='--')
    
    # Formatting
    ax.set_title("JGRI Comparison (Relative Grammatical Complexity)", fontsize=14)
    ax.set_xlabel("Text File", fontsize=12)
    ax.set_ylabel("JGRI Score (Z-Score Average)", fontsize=12)
    ax.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig(filename)
    plt.close(fig)
    return filename

# ===============================================
# Other Helper Functions (Script, Kanji, JLPT, POS)
# ===============================================
# (These functions remain the same as the prior version)

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

results_raw = []
results = []
pos_percentage_results = []
pos_count_results = []

if input_files:
    st.header("2. Analysis Results")
    st.markdown("Coverage columns report the count of **unique words** from the text found in that category.")
    
    progress_bar = st.progress(0, text="--- PASS 1: Analyzing components and raw metrics ---")
    
    # --- PASS 1: Calculate all raw metrics for the corpus ---
    corpus_data = []
    
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
        
        # 1. Tokenization (Full Fugashi output)
        tagged_nodes = list(tagger(text))
        
        # 2. Calculate JGRI RAW components
        jgri_raw_components = analyze_jgri_components(text, tagged_nodes)
        
        # 3. Store raw data for Pass 2 (Normalization)
        corpus_data.append({
            'Filename': filename,
            'Text': text,
            'Tagged_Nodes': tagged_nodes,
            'Tokens': [word.surface for word in tagged_nodes],
            **jgri_raw_components
        })
        
        progress_bar.progress((i + 1) / len(input_files), text=f"PASS 1: Analyzed {i+1} of {len(input_files)} files.")

    if not corpus_data:
        progress_bar.empty()
        st.error("No valid text files were processed.")
        st.stop()

    df_raw_metrics = pd.DataFrame(corpus_data)
    
    # --- PASS 2: Calculate JGRI and final results ---
    progress_bar.progress(0, text="--- PASS 2: Calculating JGRI and final results ---")
    
    jgri_values = calculate_jgri(df_raw_metrics)
    
    for i, data in enumerate(corpus_data):
        
        # --- Structural and POS Analysis ---
        script_distribution = analyze_script_distribution(data['Text'])
        kanji_density = analyze_kanji_density(data['Text'])
        pos_percentages, pos_counts = analyze_pos_distribution(data['Tagged_Nodes'], data['Filename'])

        # 4. Lexical Richness Metrics
        text_tokenized = " ".join(data['Tokens'])
        lex = LexicalRichness(text_tokenized)
        total_tokens = lex.words
        unique_tokens = lex.terms
        ttr = lex.ttr
        hdd_value = lex.hdd(draws=min(42, total_tokens)) if total_tokens > 0 else None
        mtld_value = lex.mtld()
        
        # 5. JLPT coverage
        jlpt_counts = analyze_jlpt_coverage(data['Tokens'], jlpt_dict_to_use)

        # --- Compile Final Summary Result ---
        result = {
            "Filename": data['Filename'],
            "JGRI": jgri_values[i],
            # JGRI Raw Components
            "MMS": data['MMS'],
            "LD": data['LD'],
            "VPS": data['VPS'],
            "MPN": data['MPN'],
            # Other Structural/Lexical Metrics
            "Kanji_Density": kanji_density,
            "Script_Distribution": f"K: {script_distribution['Kanji']}% | H: {script_distribution['Hiragana']}% | T: {script_distribution['Katakana']}% | O: {script_distribution['Other']}%",
            "Tokens": total_tokens,
            "Types": unique_tokens,
            "TTR": ttr,
            "HDD": hdd_value,
            "MTLD": mtld_value,
        }
        for level in ALL_OUTPUT_LEVELS:
            result[level.replace(" ", "_")] = jlpt_counts.get(level, 0)

        results.append(result)
        pos_percentage_results.append(pos_percentages)
        pos_count_results.append(pos_counts)
        
        progress_bar.progress((i + 1) / len(corpus_data), text=f"PASS 2: Completed analysis for {data['Filename']}.")

    progress_bar.empty()
    st.success("Analysis complete!")

    # =========================================================================
    # --- DISPLAY SECTION ---
    # =========================================================================

    # --- 2A. VISUALIZATIONS (NEW SECTION) ---
    df_results = pd.DataFrame(results)

    st.subheader("3. Visualizations")
    
    if len(df_results) >= 1:
        # Plot 1: JLPT Coverage (Normalized)
        jlpt_plot_file = plot_jlpt_coverage(df_results, filename="jlpt_coverage.png")
        st.image(jlpt_plot_file, caption="Normalized JLPT Vocabulary Coverage (Percentage of Types)")

        # Plot 2: JGRI Comparison (Only useful if more than one file is present)
        if len(df_results) > 1:
            jgri_plot_file = plot_jgri_comparison(df_results, filename="jgri_comparison.png")
            st.image(jgri_plot_file, caption="JGRI Comparison (Relative Grammatical Complexity)")
        else:
            st.info("Upload at least two files to compare JGRI scores effectively.")
    
    st.markdown("---")


    # --- 2B. MAIN SUMMARY TABLE (Lexical, Structural, Coverage) ---
    st.subheader("Summary Table (Lexical, Structural & Readability Metrics)")
    
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
    st.subheader("Detailed Part-of-Speech (POS) Distribution")
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
        
    st.download_button(
        label="⬇️ Download All Results as Excel",
        data=output.getvalue(),
        file_name="lexical_profile_results_full.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )
    
else:
    st.header("Upload Files to Begin")
    st.info("Please upload your Japanese text files (.txt) using the **sidebar**.")
