import streamlit as st
import pandas as pd
import io
import os

# --- Configuration ---
# You must place these four Excel files in the root of your GitHub repository.
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
    """Loads the JLPT wordlist from the selected file bundled in the repository."""
    
    # Check if file exists in the deployment environment
    if not os.path.exists(filename):
        st.error(f"Required file '{filename}' not found. Please ensure it is uploaded to the GitHub repo.")
        return None

    try:
        jlpt_dict = {}
        xls = pd.ExcelFile(filename)
        for level in ALL_JLPT_LEVELS:
            # Assuming column name = sheet name
            words = set(xls.parse(level)[level].dropna().astype(str).tolist())
            jlpt_dict[level] = words
        st.success(f"Wordlist '{filename}' loaded successfully!")
        return jlpt_dict
    except KeyError as e:
        st.error(f"Error: Missing expected sheet or column '{e}' in '{filename}'. Check the file structure.")
        return None
    except Exception as e:
        st.error(f"Error loading JLPT Wordlist from file: {e}")
        return None

@st.cache_resource(show_spinner="Initializing Fugashi Tokenizer...")
def initialize_tokenizer():
    """Initializes the Fugashi Tagger."""
    try:
        tagger = Tagger()
        st.success("Fugashi tokenizer loaded successfully!")
        return tagger
    except Exception as e:
        st.error(f"Error loading Fugashi: {e}")
        return None

# ===============================================
# Helper: JLPT Coverage
# ===============================================
def analyze_jlpt_coverage(words, selected_levels_dict):
    """Calculates the count of words matching only the selected JLPT levels."""
    result = {}
    for level, wordset in selected_levels_dict.items():
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

# 2. Level Selection
selected_levels = st.sidebar.multiselect(
    "2. Select JLPT Levels for Coverage:",
    options=ALL_JLPT_LEVELS,
    default=ALL_JLPT_LEVELS,
    help="Only words matching the selected lists will be counted in the JLPT columns."
)

# Load essential components
jlpt_dict_all = load_jlpt_wordlist_from_file(selected_wordlist_filename)
tagger = initialize_tokenizer()

# Check for prerequisites and filter dictionary
if jlpt_dict_all is None or tagger is None or not selected_levels:
    if not selected_levels:
        st.sidebar.warning("Please select at least one JLPT level.")
    st.stop() # Stop execution if prerequisites fail

# Filter the master dictionary based on user level selection
jlpt_dict_filtered = {level: jlpt_dict_all[level] for level in selected_levels}


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
        jlpt_counts = analyze_jlpt_coverage(tokens, jlpt_dict_filtered)

        # --- Compile Result ---
        result = {
            "Filename": filename,
            "Tokens": total_tokens,
            "Types": unique_tokens,
            "TTR": ttr,
            "HDD": hdd_value,
            "MTLD": mtld_value,
        }
        # Dynamically add JLPT results (0 for unselected levels)
        for level in ALL_JLPT_LEVELS:
            # Use underscores for column names
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
        1. Select your preferred JLPT Word List and Levels using the configuration panel on the **left sidebar**.
        2. Upload your Japanese raw text files (.txt) above.
        3. Results will appear here, along with a download button.
    """)
