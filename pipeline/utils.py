import logging
import spacy

log = logging.getLogger(__name__)

nlp = spacy.load("en_core_web_sm")
nlp.max_length = int(2e6)

def lemmatize(text):
    words = [w.lemma_ for w in nlp(text)]
    # add space at start and end to match ngrams
    words = " ".join(words).replace(" - ", "-")
    return words


def spaced(text):
    return f" {text.strip()} "