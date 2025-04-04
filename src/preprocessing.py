import string
import contractions
import nltk
from nltk.corpus import stopwords, wordnet
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

def get_wordnet_pos(universal_tag):
    if universal_tag == 'ADJ':
        return wordnet.ADJ
    elif universal_tag == 'VERB':
        return wordnet.VERB
    elif universal_tag == 'NOUN':
        return wordnet.NOUN
    elif universal_tag == 'ADV':
        return wordnet.ADV
    else:
        return wordnet.NOUN

def preprocess_text(text):
    text = contractions.fix(text)
    text = text.lower().translate(str.maketrans("", "", string.punctuation))
    tokens = word_tokenize(text)
    pos_tags = nltk.pos_tag(tokens, tagset='universal')
    cleaned_tokens = [
        lemmatizer.lemmatize(token, get_wordnet_pos(pos))
        for token, pos in pos_tags
        if token not in stop_words and len(token) > 2
    ]
    return cleaned_tokens  # ← important: return tokens
