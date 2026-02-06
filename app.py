import streamlit as st
import pandas as pd
import re
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from collections import Counter
import ufal.morphodita

st.title("Tagcloud systém pre abstrakty záverečných prác študentov")
st.write("Aplikácia funguje")

st.subheader("Nahraj abstrakty")

MODEL_PATH = "slovak-morfflex-pdt-170914.tagger"  # súbor je teraz v koreňovom repozitári
tagger = ufal.morphodita.Tagger.load(MODEL_PATH)
if not tagger:
    st.error("Nepodarilo sa načítať Morphodita model!")
    st.stop()

def lemmatize_word(word):
    """
    Funkcia na lematizáciu jedného slova pomocou Morphodita
    """
    forms = ufal.morphodita.Forms()
    lemmas = ufal.morphodita.Lemmas()
    tagger.tag(word, forms, lemmas)
    if lemmas.size() > 0:
        return lemmas[0].lemma
    else:
        return word
        
uploaded_file = st.file_uploader(
    "Vyber CSV súbor s abstraktmi záverečných prác",
    type="csv"

if uploaded_file:
    import pandas as pd
    df = pd.read_csv(uploaded_file)
    st.write("Načítané dáta:")
    st.write(df.head())

    text = " ".join(df["abstrakt"].astype(str))
    words = re.findall(r"\b\w+\b", text.lower())

    st.write("Počet slov:", len(words))
    st.write(words[:20])
    stopwords_input = st.text_area(
        "Vlastné stop slová (oddelené čiarkou)",
        "bakalárska,cieľ,práca,analýza"
    )

    stopwords = [w.strip().lower() for w in stopwords_input.split(",")]

    filtered_words = [
        w for w in words if w not in stopwords and len(w) > 2
    ]
    st.write("Slová po filtrovaní:")
    st.write(filtered_words[:20])

    words_counter = Counter(filtered_words)
    
    from wordcloud import WordCloud
    import matplotlib.pyplot as plt
    
    wc_text = " ".join(filtered_words)
    
    wc = WordCloud(
    width=800,
    height=400,
    background_color="white",
    colormap="viridis", 
    stopwords=None 
    ).generate_from_frequencies(words_counter)
    
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.imshow(wc, interpolation='bilinear')
    ax.axis("off")
    st.pyplot(fig)


