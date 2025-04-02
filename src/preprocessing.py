import string

import contractions
import nltk
from nltk.corpus import stopwords, wordnet
from nltk.tokenize import word_tokenize
from textblob import TextBlob

stop_words = set(stopwords.words("english"))

def pos_tagger(nltk_tag: str) -> str:
    """Map NLTK POS tags to WordNet POS tags."""
    if nltk_tag.startswith("J"):
        return wordnet.ADJ
    elif nltk_tag.startswith("V"):
        return wordnet.VERB
    elif nltk_tag.startswith("N"):
        return wordnet.NOUN
    elif nltk_tag.startswith("R"):
        return wordnet.ADV
    return ""


def lemmatize_tokens(tokens: list[str]) -> list[str]:
    """Lemmatize tokens based on their POS tags."""
    lemmatizer = nltk.WordNetLemmatizer()
    pos_tagged = nltk.pos_tag(tokens)
    return [
        lemmatizer.lemmatize(token, pos_tagger(tag)) if pos_tagger(tag) else lemmatizer.lemmatize(token)
        for token, tag in pos_tagged
    ]

def expand_contractions(text: str) -> str:
    """Expand contractions in a given text."""
    try:
        if not isinstance(text, str) or not text.strip():
            return text
        expanded_text = contractions.fix(text)
        return expanded_text  # type: ignore

    except Exception as e:
        print(f"Error expanding contractions in: {text}, Error: {e}")
        return text


# def preprocess_sentence(sentence: str, add_noun_phrases: bool = False, stopwords: bool = False) -> str:

