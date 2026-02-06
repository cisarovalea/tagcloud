import streamlit as st
import pandas as pd
import re
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from collections import Counter
import ufal.udpipe

import streamlit as st
import ufal.udpipe

MODEL_PATH = "slovak-snk-ud-2.5-191206.udpipe"  # uprav podľa repozitára
model = ufal.udpipe.Model.load(MODEL_PATH)
if model:
    st.write("UDPipe model sa načítal správne!")
else:
    st.error("Nepodarilo sa načítať UDPipe model! Skontroluj cestu a súbor v repozitári.")
    st.stop()

st.title("Tagcloud systém pre abstrakty záverečných prác študentov")
st.write("Aplikácia funguje")

st.subheader("Nahraj abstrakty")

MODEL_PATH = "slovak-snk-ud-2.5-191206.udpipe"  # cesta v repozitári
model = ufal.udpipe.Model.load(MODEL_PATH)
if not model:
    st.error("Nepodarilo sa načítať UDPipe model!")
    st.stop()

def lemmatize_words_udpipe(text):
    """
    Vstup: text ako string
    Výstup: zoznam lematizovaných slov
    """
    pipeline = ufal.udpipe.Pipeline(model, "tokenize", ufal.udpipe.Pipeline.DEFAULT, ufal.udpipe.Pipeline.DEFAULT, "conllu")
    processed = pipeline.process(text)
    
    lemmas = []
    for line in processed.split("\n"):
        if line.startswith("#") or line.strip() == "":
            continue
        parts = line.split("\t")
        if len(parts) >= 4:
            lemma = parts[2]
            lemmas.append(lemma.lower())
    return lemmas
    
uploaded_file = st.file_uploader(
    "Vyber CSV súbor s abstraktmi záverečných prác",
    type="csv"
)

if uploaded_file:
    import pandas as pd
    df = pd.read_csv(uploaded_file)
    st.write("Načítané dáta:")
    st.write(df.head())

    text = " ".join(df["abstrakt"].astype(str))
    words = re.findall(r"\b\w+\b", text.lower())
    
    st.write("Lematizujem text... (môže chvíľu trvať pri veľkých datasetoch)")
    lemmas = lemmatize_words_udpipe(text)
    st.write("Prvých lematizovaných slov:", len(lemmas))
    st.write("Prvých 50 slov na kontrolu:", lemmas[:50])
    
    stopwords_input = st.text_area(
        "Vlastné stop slová (oddelené čiarkou)",
        "bakalárska,cieľ,práca,analýza"
    )

    stopwords = [w.strip().lower() for w in stopwords_input.split(",")]

    filtered_lemmas = [w for w in lemmas if w not in stopwords and len(w) > 2]
    st.write("Slová po filtrovaní:")
    st.write(filtered_lemmas[:50])

    words_counter = Counter(filtered_lemmas)
    
    most_common = words_counter.most_common(50)
    st.write("50 najčastejších slov:", most_common)
    
    from wordcloud import WordCloud
    import matplotlib.pyplot as plt
    
    wc_text = " ".join(filtered_lemmas)
    
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


