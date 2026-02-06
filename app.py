st.title("NLP systém pre abstrakty")
st.write("Aplikácia funguje 🎉")

st.subheader("Nahraj abstrakty")

uploaded_file = st.file_uploader(
    "Vyber CSV súbor s abstraktmi",
    type="csv"
)

if uploaded_file:
    import pandas as pd
    df = pd.read_csv(uploaded_file)
    st.write("Načítané dáta:")
    st.write(df.head())
    import re

    text = " ".join(df["abstrakt"].astype(str))
    words = re.findall(r"\b\w+\b", text.lower())

    st.write("Počet slov:", len(words))
    st.write(words[:20])
    stopwords_input = st.text_area(
        "Vlastné stop slová (oddelené čiarkou)",
        "bakalárska,cieľ,práca,analýza"
    )

    stopwords = [w.strip() for w in stopwords_input.split(",")]

    filtered_words = [
        w for w in words if w not in stopwords and len(w) > 2
    ]

    st.write("Slová po filtrovaní:")
    st.write(filtered_words[:20])
    
    from wordcloud import WordCloud
    import matplotlib.pyplot as plt
    
    wc_text = " ".join(filtered_words)
    
    wc = WordCloud(
    width=800,
    height=400,
    background_color="white",
    colormap="viridis",  # môžeš zmeniť farby
    stopwords=None  # stop slová už máme filtrované manuálne
    ).generate(wc_text)
    
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.imshow(wc, interpolation='bilinear')


