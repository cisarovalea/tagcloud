import streamlit as st
import pandas as pd
import re
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from collections import Counter
import ufal.udpipe

MODEL_PATH = "slovak-snk-ud-2.5-191206.udpipe"
model = ufal.udpipe.Model.load(MODEL_PATH)
if model:
    st.write("UDPipe model sa načítal správne!")
else:
    st.error("Nepodarilo sa načítať UDPipe model! Skontroluj cestu a súbor v repozitári.")
    st.stop()

st.title("Tagcloud systém pre abstrakty záverečných prác študentov")

st.subheader("Nahraj abstrakty")

slovak_stopwords = [
    "a", "aby", "aj", "ale", "ani", "ako", "ani", "sa", "som", "si", "sú", "je",
    "že", "zo", "za", "na", "v", "do", "po", "pre", "od", "k", "so", "bez", "o",
    "alebo", "tak", "ten", "tá", "to", "s", "sa", "sú", "sa", "už", "do", "pri",
    "sa", "ako", "pretože", "keď", "ktorý", "ktorá", "ktoré", "ale", "teda", "táto",
    "podľa", "napriek", "byť", "zároveň", "vrátane", "mnohý", "určený", "toto", "ona", "on",
    "(", ")", ",", ".", "jeho", "jej", "viacero", "pričom", "ich", "mať"
]

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
    df = pd.read_csv(uploaded_file)
    st.write("Načítané dáta:")
    st.write(df.head())

    # 1. TEXT
    text = " ".join(df["abstrakt"].astype(str))

    # 2. LEMATIZÁCIA
    st.write("Lematizujem text...")
    lemmas = lemmatize_words_udpipe(text)
    n_words = st.slider(
        "Počet slov v tagcloude",
        min_value=10,
        max_value=200,
        value=50,
        step=10
)

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
    
    min_freq = st.slider(
        "Minimálna frekvencia slova",
        1, 20, 2,
        key="min_freq"
)
    # 4. FREKVENCIE
    words_counter = Counter(filtered_lemmas)

    words_counter = Counter(
        dict(words_counter.most_common(n_words))
)

    n_words = st.slider(
        "Počet slov v tagcloude",
        10, 200, 50,
        key="n_words"
)
    colormap = st.selectbox(
    "Farebná schéma",
    [
        "viridis",
        "plasma",
        "inferno",
        "magma",
        "cividis",
        "cool",
        "spring",
        "summer",
        "autumn",
        "winter"
    ]
)
    background_color = st.selectbox(
    "Farba pozadia",
    ["white", "black"]
)
    # 5. WORDCLOUD
    wc = WordCloud(
    width=800,
    height=400,
    background_color=background_color,
    colormap=colormap,
    max_words=n_words
).generate_from_frequencies(words_counter)

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.imshow(wc, interpolation='bilinear')
    ax.axis("off")
    st.pyplot(fig)
    import io

    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    buf.seek(0)

    top_words_df = pd.DataFrame(
        words_counter.most_common(n_words),
        columns=["Slovo", "Frekvencia"]
)

    top_words_df.insert(
        0,
        "Poradie",
        range(1, len(top_words_df) + 1)
)
    
    st.subheader("Frekvencie slov")
    st.dataframe(top_words_df)

    csv = top_words_df.to_csv(index=False).encode("utf-8")

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
    


