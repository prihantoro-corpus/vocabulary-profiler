import streamlit as st
import pandas as pd
import io
import os

# --- Configuration ---
# IMPORTANT: These files MUST be in the root of your GitHub repository.
AVAILABLE_WORDLISTS = [
    "_JLPT_Word_List MERGED SOURCE UNKNOWN.xlsx",
    "_MERGED ANKI JLPT WORDLIST.xlsx",
    "_Merged BLUSKYO.xlsx",
    "_jlpt_vocab-Kaggle.xlsx"
]

ALL_JLPT_LEVELS = ["JLPT N5", "JLPT N4", "JLPT N3", "JLPT N2", "JLPT N1"]

# --- Install packages (Ensure these are in requirements.txt) ---
try:
    from lexicalrichness import LexicalRichness
except ImportError:
    st.error("The 'lexicalrichness' package is missing.")
    st.stop()

try:
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
st.markdown("Analyze lexical richness (TTR, HDD, MTLD) and **JLPT word coverage**.")

# ===============================================
# Helper Functions - Caching for Performance
# ===============================================

@st.cache_data(show_spinner="Loading and processing JLPT Wordlist...")
def load_jlpt_wordlist_from_file(filename):
    """
    Loads the JLPT wordlist from the selected file.
    It searches for a column named 'Level' (case-insensitive) in each sheet.
    """
    
    if not os.path.exists(filename):
        st.error(f"Required file '{filename}' not found. Please ensure it is uploaded to the GitHub repo.")
        return None

    try:
        jlpt_dict = {}
        xls = pd.ExcelFile(filename)
        
        for level in ALL_JLPT_LEVELS:
            if level not in xls.sheet_names:
                 st.error(f"Error: Sheet '{level}' not found in the selected file.")
                 return None
            
            # Read the sheet
            df = xls.parse(level)
            
            # --- REVISED LOGIC: Find the 'Level' column ---
            word_column = None
            
            # Search for 'level' header, case-insensitive
            for col in df.columns:
                if isinstance(col, str) and col.strip().lower() == 'level':
                    word_column = col
                    break
            
            if word_column is None:
                st.error(f"Error in sheet '{level}': Could not find a column header named 'Level' (case-insensitive).")
                return None
            # ---------------------------------------------

            # Extract words from the identified column
            words = set(df[word_column].dropna().astype(str).tolist())
            jlpt_dict[level] = words
        
        st.success(f"Wordlist '{filename}' loaded successfully!")
        return jlpt_dict
        
    except Exception as e:
        st.error(f"Error loading JLPT Wordlist from file: {e}")
        st.stop()
        return None

@st.cache_resource(show_spinner="Initializing Fugashi Tokenizer...")
def initialize_tokenizer():
    """Initializes the Fugashi Tagger."""
    try:
        tagger = Tagger()
        st.success("Fugashi tokenizer loaded successfully!")
        return tagger
    except Exception as e:
        # This error is usually fixed by unidic-lite in requirements.txt
        st.error(f"Error loading Fugashi: {e}")
        st.error("Please ensure 'unidic-lite' is in your requirements.txt to fix MeCab initialization.")
        return None

# ===============================================
# Helper: JLPT Coverage
# ===============================================
def analyze_jlpt_coverage(words, jlpt_dict):
    """Calculates the count of words matching each JLPT level."""
    result = {}
    for level, wordset in jlpt_dict.items():
        # Count words in the input that are also in the JLPT word set
        count = sum(1 for w in words if w in wordset)
        result[level] = count
    return result

# ===============================================
# Sidebar Configuration
# ===============================================
st.sidebar.header("⚙️ Configuration")

# 1. Wordlist Selection
selected_wordlist_filename = st.sidebar.selectbox(
    "1. Select JLPT Word List Source:",
    options=AVAILABLE_WORDLISTS,
    index=0,
    help="Choose one of the word list files uploaded to the repository."
)

# Load essential components
jlpt_dict_all = load_jlpt_wordlist_from_file(selected_wordlist_filename)
tagger = initialize_tokenizer()

if jlpt_dict_all is None or tagger is None:
    st.stop() # Stop execution if prerequisites fail

# Use all levels for analysis, achieving the goal of N1-N5 distribution
jlpt_dict_to_use = jlpt_dict_all 

# ===============================================
# Main Area: Process Input Files
# ===============================================

st.header("1. Upload Raw Text Files")
input_files = st.file_uploader(
    "Upload one or more **.txt** files for analysis.",
    type=["txt"],
    accept_multiple_files=True,
    key="input_uploader"
)

results = []
if input_files:
    st.header("2. Analysis Results")
    progress_bar = st.progress(0, text="Processing files...")
    
    for i, uploaded_file in enumerate(input_files):
        filename = uploaded_file.name
        
        # Read file content
        string_data = uploaded_file.getvalue().decode("utf-8")
        text = string_data.strip()

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

# --- Instructions when no files are uploaded ---
else:
    st.info(f"""
        **Instructions:**
        1. Select your preferred JLPT Word List Source using the configuration panel on the **left sidebar**.
        2. Upload your Japanese raw text files (.txt) above.
        3. Results (including N5-N1 distribution) will appear here, along with a download button.
    """)
