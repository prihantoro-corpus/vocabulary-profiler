import streamlit as st
import pandas as pd
import io
import os
import re

# --- Configuration ---
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
                 st.warning(f"CSV file {filename} is empty.")
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
# Helper: Script and Density Analysis
# ===============================================

def analyze_script_distribution(text):
    """Calculates the percentage of Kanji, Hiragana, Katakana, and Romaji/Other."""
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
    """Calculates the average number of Kanji characters per sentence."""
    
    # Simple sentence splitting: split by common Japanese punctuation marks
    sentences = re.split(r'[。！？\n]', text.strip())
    sentences = [s.strip() for s in sentences if s.strip()]

    if not sentences:
        return 0.0

    total_kanji = len(re.findall(r'[\u4E00-\u9FFF]', text))
    num_sentences = len(sentences)

    density = total_kanji / num_sentences
    
    return round(density, 2)

# ===============================================
# Helper: JLPT Coverage
# ===============================================
def analyze_jlpt_coverage(tokens, jlpt_dict):
    """
    Calculates word counts for N5-N1 levels and adds an 'NA' category for 
    words not covered by any list.
    """
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

# ===============================================
# Sidebar & Initialization
# ===============================================

# Load essential components
jlpt_dict_to_use = load_jlpt_wordlist()
tagger = initialize_tokenizer()

if jlpt_dict_to_use is None or tagger is None:
    st.stop() # Stop execution if prerequisites fail

# --- Sidebar Configuration (Upload files here) ---
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
if input_files:
    st.header("2. Analysis Results")
    st.markdown("Coverage columns report the count of **unique words** from the text found in that category.")
    
    progress_bar = st.progress(0, text="Processing files...")
    
    for i, uploaded_file in enumerate(input_files):
        filename = uploaded_file.name
        
        # Read file content
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
        
        # --- Core Analysis ---
        
        # Script Distribution and Density
        script_distribution = analyze_script_distribution(text)
        kanji_density = analyze_kanji_density(text)
        
        # Tokenization and Lexical Richness
        tokens = [word.surface for word in tagger(text)]
        text_tokenized = " ".join(tokens)
        lex = LexicalRichness(text_tokenized)
        
        # Lexical metrics
        total_tokens = lex.words
        unique_tokens = lex.terms
        ttr = lex.ttr
        hdd_value = lex.hdd(draws=min(42, total_tokens)) if total_tokens > 0 else None
        mtld_value = lex.mtld()
        
        # JLPT coverage
        jlpt_counts = analyze_jlpt_coverage(tokens, jlpt_dict_to_use)

        # --- Compile Result ---
        result = {
            "Filename": filename,
            # Structural Metrics
            "Kanji_Density": kanji_density,
            "Script_Distribution": f"K: {script_distribution['Kanji']}% | H: {script_distribution['Hiragana']}% | T: {script_distribution['Katakana']}% | O: {script_distribution['Other']}%",
            # Lexical Metrics
            "Tokens": total_tokens,
            "Types": unique_tokens,
            "TTR": ttr,
            "HDD": hdd_value,
            "MTLD": mtld_value,
        }
        # Add N5-N1 and NA distribution
        for level in ALL_OUTPUT_LEVELS:
            result[level.replace(" ", "_")] = jlpt_counts.get(level, 0)

        results.append(result)
        
        progress_bar.progress((i + 1) / len(input_files), text=f"Processed {i+1} of {len(input_files)} files.")

    progress_bar.empty()
    st.success("Analysis complete!")

    # --- Display and Download Output ---
    df_results = pd.DataFrame(results)
    
    st.subheader("Summary Table")
    
    # --- UPDATED DISPLAY LOGIC: Use simple st.dataframe ---
    # We must provide the explanations separately since st.column_config failed.
    
    # Manually rename columns for cleaner display and tooltips
    display_names = {
        "Kanji_Density": "Kanji Density ❓",
        "Script_Distribution": "Script Distribution ❓",
        "Tokens": "Tokens ❓",
        "Types": "Types ❓",
        "TTR": "TTR ❓",
        "HDD": "HDD ❓",
        "MTLD": "MTLD ❓",
        "JLPT_N5": "JLPT_N5 ❓",
        "JLPT_N4": "JLPT_N4 ❓",
        "JLPT_N3": "JLPT_N3 ❓",
        "JLPT_N2": "JLPT_N2 ❓",
        "JLPT_N1": "JLPT_N1 ❓",
        "NA": "NA ❓",
    }
    
    # Apply tooltips using a simplified method for wide compatibility
    st.markdown("""
        **Column Explanations (Hover over the '❓' below):**
        * **Kanji Density:** Average Kanji characters per sentence (Higher = more complex).
        * **Script Distribution:** Percentage breakdown (K=Kanji, H=Hiragana, T=Katakana, O=Other). 
        * **Tokens/Types:** Total words / Unique words.
        * **TTR/HDD/MTLD:** Lexical richness metrics (Higher scores = more diverse vocabulary).
        * **JLPT/NA:** Count of unique words covered by the JLPT level or Not Covered (NA).
    """)

    st.dataframe(df_results.rename(columns=display_names), use_container_width=True)
    # -----------------------------------------------------

    # Convert DataFrame to Excel in memory for download
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer: 
        df_results.to_excel(writer, index=False, sheet_name='Lexical Profile')
    
    st.download_button(
        label="⬇️ Download Results as Excel",
        data=output.getvalue(),
        file_name="lexical_profile_results.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )
    
else:
    st.header("Upload Files to Begin")
    st.info("Please upload your Japanese text files (.txt) using the **sidebar**.")
