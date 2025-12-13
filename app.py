# ===============================================
# Other Helper Functions (Plotting, etc.)
# ===============================================

# --- CORRECTED CODE ---
def analyze_jgri_components(text, tagged_nodes):
    pos_counts = Counter(node.feature.pos1 for node in tagged_nodes if node.surface and node.feature.pos1)
    Nouns = pos_counts.get('名詞', 0)
    Verbs = pos_counts.get('動詞', 0)
    Adjectives = pos_counts.get('形容詞', 0)
    Adverbs = pos_counts.get('副詞', 0)
    Total_Morphemes = len(tagged_nodes)
    sentences = re.split(r'[。！？\n]', text.strip())
    sentences = [s.strip() for s in sentences if s.strip()]
    Num_Sentences = len(sentences)
    if Total_Morphemes == 0 or Nouns == 0 or Num_Sentences == 0: 
        return {'MMS': 0.0, 'LD': 0.0, 'VPS': 0.0, 'MPN': 0.0}
    MMS = Total_Morphemes / Num_Sentences
    LD = (Nouns + Verbs + Adjectives + Adverbs) / Total_Morphemes
    VPS = Verbs / Num_Sentences
    MPN = (Adjectives + Verbs) / Nouns
    return {'MMS': MMS, 'LD': LD, 'VPS': VPS, 'MPN': MPN}

def calculate_jgri(metrics_df): 
    jgri_values = []
    mu = metrics_df[['MMS', 'LD', 'VPS', 'MPN']].mean()
    sigma = metrics_df[['MMS', 'LD', 'VPS', 'MPN']].std()
    sigma = sigma.replace(0, 1e-6) 
    for index, row in metrics_df.iterrows(): 
        raw_values = row[['MMS', 'LD', 'VPS', 'MPN']]
        z_mms = (raw_values['MMS'] - mu['MMS']) / sigma['MMS']
        z_ld = (raw_values['LD'] - mu['LD']) / sigma['LD']
        z_vps = (raw_values['VPS'] - mu['VPS']) / sigma['VPS']
        z_mpn = (raw_values['MPN'] - mu['MPN']) / sigma['MPN']
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
# ... rest of the script is updated similarly ...
