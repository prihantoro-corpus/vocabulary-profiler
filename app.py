@st.cache_data(show_spinner="Loading and processing JLPT Wordlist...")
def load_jlpt_wordlist_from_file(filename):
    """
    Loads the JLPT wordlist from the selected file.
    It searches for a column named 'Word' (case-insensitive) in each sheet 
    to extract the vocabulary list.
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
            
            # --- REVISED LOGIC: Find the 'Word' column ---
            word_column = None
            
            # Search for 'word' header, case-insensitive
            for col in df.columns:
                if isinstance(col, str) and col.strip().lower() == 'word':
                    word_column = col
                    break
            
            # Fallback: If 'Word' isn't found, try 'Level' (your previous requirement)
            if word_column is None:
                for col in df.columns:
                    if isinstance(col, str) and col.strip().lower() == 'level':
                        word_column = col
                        break
            
            if word_column is None:
                st.error(f"Error in sheet '{level}': Could not find a column header named 'Word' or 'Level'.")
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
# ... rest of the code remains the same ...
