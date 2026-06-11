import streamlit as st
import pandas as pd
import re
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from collections import Counter
import ufal.udpipe
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np


MODEL_PATH = "slovak-snk-ud-2.5-191206.udpipe"
model = ufal.udpipe.Model.load(MODEL_PATH)
if model:
    st.write("UDPipe model sa načítal správne!")
else:
    st.error("Nepodarilo sa načítať UDPipe model! Skontroluj cestu a súbor v repozitári.")
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
    
with st.sidebar:
    st.title("Nastavenia")

    n_words = st.slider(
        "Počet slov v tagcloude",
        10, 200, 50,
        key="n_words"
    )

    min_freq = st.slider(
        "Minimálna frekvencia slova",
        1, 20, 2,
        key="min_freq"
    )

    ngram_type = st.selectbox(
        "Typ výrazov",
        ["Jednoslovné", "Bigramy"]
    )

    colormap = st.selectbox(
        "Farebná schéma",
        ["viridis", "plasma", "magma", "cividis"]
    )
st.title("Tagcloud systém pre abstrakty záverečných prác")
st.write("Aplikácia funguje")

uploaded_file = st.file_uploader(
    "Nahraj CSV súbor s abstraktmi",
    type="csv"
)

slovak_stopwords = [
    "a", "aby", "aj", "ale", "ani", "ako", "ani", "sa", "som", "si", "sú", "je",
    "že", "zo", "za", "na", "v", "do", "po", "pre", "od", "k", "so", "bez", "o",
    "alebo", "tak", "ten", "tá", "to", "s", "sa", "sú", "sa", "už", "do", "pri",
    "sa", "ako", "pretože", "keď", "ktorý", "ktorá", "ktoré", "ale", "teda", "táto",
    "podľa", "napriek", "byť", "zároveň", "vrátane", "mnohý", "určený", "toto", "ona", "on",
    "(", ")", ",", ".", "jeho", "jej", "viacero", "pričom", "ich", "mať"
]

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    text = " ".join(df["abstrakt"].astype(str))

    words = re.findall(r"\b\w+\b", text.lower())
    
    st.subheader("Štatistiky datasetu")
    lemmas = words
    # 1. počet abstraktov
    num_abstracts = len(df)

    # 2. text
    text = " ".join(df["abstrakt"].astype(str))

    # 3. lematizácia
    lemmas = lemmatize_words_udpipe(text)

    # 4. základné štatistiky
    num_tokens = len(re.findall(r"\b\w+\b", text.lower()))
    num_lemmas = len(lemmas)
    num_unique_lemmas = len(set(lemmas))

    st.write("Počet abstraktov:", len(df))
    st.write("Počet tokenov:", len(words))
    st.write("Počet lemát:", num_lemmas)
    st.write("Počet unikátnych slov:", len(set(words)))

    # 1. TEXT
    text = " ".join(df["abstrakt"].astype(str))

    # 2. LEMATIZÁCIA
    st.write("Lematizujem text...")
    lemmas = lemmatize_words_udpipe(text)
    
    st.write("Počet lematizovaných slov:", len(lemmas))
    st.write("Prvých 25 slov:", lemmas[:25])

    # 3. STOP SLOVÁ
    stopwords_input = st.text_area(
        "Vlastné stop slová (oddelené čiarkou)",
        "bakalársky,cieľ,práca,analýza,časť,praktický,teória,teoretický,jednotlivý,prirodzený,uvedený,preskúmať,vhodný,následne,možnosť,zaoberať"
    )

    custom_stopwords = [
        w.strip().lower()
        for w in stopwords_input.split(",")
        if w.strip()
    ]

    all_stopwords = set(slovak_stopwords + custom_stopwords)

    filtered_lemmas = [
        w for w in lemmas
        if w not in all_stopwords and len(w) > 2
]

    st.write("Počet slov po filtrovaní:", len(filtered_lemmas))
    st.write("Prvých 25 slov:", filtered_lemmas[:25])

    ngram_type = st.selectbox(
        "Typ výrazov",
        ["Jednoslovné", "Dvojslovné (bigramy)"]
)
    # 3. frekvencie
    if ngram_type == "Jednoslovné":
        data = filtered_lemmas
        words_counter = Counter(data)

    else:
        text_for_bigrams = " ".join(filtered_lemmas)

        vectorizer = CountVectorizer(ngram_range=(2, 2))
        X = vectorizer.fit_transform([text_for_bigrams])

        words_list = vectorizer.get_feature_names_out()
        counts = X.toarray().sum(axis=0)

        words_counter = Counter(dict(zip(words_list, counts)))

    # 4. filter min frequency
    words_counter = Counter(
        {w: c for w, c in words_counter.items() if c >= min_freq}
    )
    # 5. TOP N (tu sa používa slider)
    words_counter = Counter(
        dict(words_counter.most_common(n_words))
    )
    
    # 5. WORDCLOUD
    wc = WordCloud(
        width=800,
        height=400,
        background_color="white",
        colormap=colormap,
        max_words=n_words,
        random_state=42
    ).generate_from_frequencies(words_counter)

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.imshow(wc, interpolation='bilinear')
    ax.axis("off")

    st.pyplot(fig)
    import io

    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    buf.seek(0)
    st.subheader("Top slová")
    top_df = pd.DataFrame(
        words_counter.most_common(n_words),
        columns=["Slovo", "Frekvencia"]
)

    st.subheader("Frekvencie slov")
    st.dataframe(top_df)

    csv = top_df.to_csv(index=False).encode("utf-8-sig")

    st.download_button(
        label="Stiahnuť frekvencie slov (CSV)",
        data=csv,
        file_name="frekvencie_slov.csv",
        mime="text/csv"
)

    st.download_button(
        label="Stiahnuť tagcloud ako PNG",
        data=buf,
        file_name="tagcloud.png",
        mime="image/png"
)
    


