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
    Assumes the words are in the first data column (index 0).
    """
    jlpt_dict = {}
    
    for level_name, filename in JLPT_FILE_MAP.items():
        if not os.path.exists(filename):
            st.error(f"Required CSV file '{filename}' not found in the repository root.")
            return None

        try:
            # Read CSV quickly. Assume first row is header (header=0).
            # We use usecols=[0, 1] to limit reading only the first two columns, 
            # and then select the appropriate column for words.
            df = pd.read_csv(filename, header=0, encoding='utf-8', keep_default_na=False)
            
            # --- WORD COLUMN LOGIC ---
            # Assume the words are in the first column found in the dataframe (index 0)
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
# Helper: JLPT Coverage
# ===============================================
def analyze_jlpt_coverage(words, jlpt_dict):
    """Calculates the count of words matching each JLPT level."""
    result = {}
    for level, wordset in jlpt_dict.items():
        count = sum(1 for w in words if w in wordset)
        result[level] = count
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
        # Add N5-N1 distribution to the result
        for level in ALL_JLPT_LEVELS:
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
