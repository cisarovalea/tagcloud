import spacy
nlp = spacy.load("sk_core_news_sm")
import streamlit as st

st.title("NLP systém pre abstrakty")
st.write("Aplikácia funguje 🎉")

st.subheader("Nahraj abstrakty")

uploaded_file = st.file_uploader(
    "Vyber CSV súbor s abstraktmi",
    type="csv"
)

if uploaded_file:
    text = " ".join(df["abstrakt"].astype(str)) 
    import re
    words = re.findall(r"\b\w+\b", text.lower())
    
    stopwords_input = st.text_area("Vlastné stop slová (oddelené čiarkou)",
                               "bakalárska,cieľ,práca,analýza")
    stopwords = [w.strip() for w in stopwords_input.split(",")]
    filtered_words = [w for w in words if w not in stopwords and len(w) > 2]
    
    doc = nlp(" ".join(filtered_words))
    lemmas = [token.lemma_ for token in doc if not token.is_punct and not token.is_space]
    
    from wordcloud import WordCloud 
    import matplotlib.pyplot as plt 
    
    wc_text = " ".join(lemmas)
    wc = WordCloud(width=800, height=400, background_color="white", colormap="viridis").generate(wc_text)
    
    fig, ax = plt.subplots(figsize=(12,6))
    ax.imshow(wc, interpolation='bilinear')
    ax.axis("off")
    st.pyplot(fig)

