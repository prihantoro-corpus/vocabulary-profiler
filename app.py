import streamlit as st
import pandas as pd
import io
import os

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
# Add the new category
ALL_OUTPUT_LEVELS = ALL_JLPT_LEVELS + ["NA"]

# --- Import Libraries (Assuming they are in requirements.txt) ---
try:
    from lexicalrichness import LexicalRichness
except ImportError:
    st.error("The 'lexicalrichness' package is missing.")
    st.stop()

try:
    # Requires 'fugashi' and 'unidic-lite' in requirements.txt
    from fugashi import Tagger
except ImportError:
    st.error("The 'fugashi' package is missing.")
    st.stop()

# --- Layout and Title ---
st.set_page_config(
    page_title="🇯🇵 Japanese Lexical Profiler",
    layout="wide"
)

st.title("🇯🇵 Japanese Lexical Profiler")
st.markdown("Analyze lexical richness (TTR, HDD, MTLD) and **JLPT word coverage distribution**.")

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
                 # Get the name of the first column
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
# Helper: JLPT Coverage (UPDATED)
# ===============================================
def analyze_jlpt_coverage(tokens, jlpt_dict):
    """
    Calculates word counts for N5-N1 levels and adds an 'NA' category for 
    words not covered by any list.
    """
    unique_tokens_in_text = set(tokens)
    result = {}
    total_known_words = set()

    # 1. Calculate counts for N5-N1 and build a master set of known words
    for level, wordset in jlpt_dict.items():
        # Count words in the input that are also in the JLPT word set
        count = sum(1 for w in unique_tokens_in_text if w in wordset)
        result[level] = count
        
        # Add the words found in this level to the total known set
        total_known_words.update(w for w in unique_tokens_in_text if w in wordset)

    # 2. Calculate NA (Not Applicable/Not Covered)
    # NA = Unique tokens in text MINUS all unique tokens found in N5-N1 lists
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

# --- Sidebar Configuration (Upload files here as requested) ---
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
    st.markdown("Coverage columns (JLPT_N5 to NA) report the count of **unique words** from the text found in that category.")
    
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
        
        # --- Tokenize Japanese text with Fugashi ---
        # Note: We use .surface (raw token) for lexical richness and coverage
        tokens = [word.surface for word in tagger(text)]
        text_tokenized = " ".join(tokens)

        # --- Lexical richness ---
        lex = LexicalRichness(text_tokenized)
        total_tokens = lex.words
        unique_tokens = lex.terms
        ttr = lex.ttr

        # HDD Calculation
        hdd_value = None
        try:
            draws = min(42, total_tokens) if total_tokens > 0 else 0
            hdd_value = lex.hdd(draws=draws) if draws > 0 else None
        except Exception:
            pass

        # MTLD Calculation
        mtld_value = None
        try:
            mtld_value = lex.mtld()
        except Exception:
            pass

        # --- JLPT coverage ---
        # Note: analyze_jlpt_coverage now calculates 'NA' using unique tokens
        jlpt_counts = analyze_jlpt_coverage(tokens, jlpt_dict_to_use)

        # --- Compile Result ---
        result = {
            "Filename": filename,
            "Tokens": total_tokens,
            "Types": unique_tokens,
            "TTR": ttr,
            "HDD": hdd_value,
            "MTLD": mtld_value,
        }
        # Add N5-N1 and NA distribution to the result
        for level in ALL_OUTPUT_LEVELS:
            # Replace space with underscore for column name uniformity
            result[level.replace(" ", "_")] = jlpt_counts.get(level, 0)

        results.append(result)
        
        progress_bar.progress((i + 1) / len(input_files), text=f"Processed {i+1} of {len(input_files)} files.")

    progress_bar.empty()
    st.success("Analysis complete!")

    # --- Display and Download Output ---
    df_results = pd.DataFrame(results)
    
    st.subheader("Summary Table")
    st.dataframe(df_results, use_container_width=True)

    # Convert DataFrame to Excel in memory for download
    output = io.BytesIO()
    # Ensure xlsxwriter is available via requirements.txt
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer: 
        df_results.to_excel(writer, index=False, sheet_name='Lexical Profile')
    
    st.download_button(
        label="⬇️ Download Results as Excel",
        data=output.getvalue(),
        file_name="lexical_profile_results.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )
