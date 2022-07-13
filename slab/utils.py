import logging
from itertools import product

import requests
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
    return f" {text.strip()}"

def duck(text):
    """ # docker run -d --name duckling --restart=always --port 8000:8000 rasa/duckling
    """
    return requests.post("http://localhost:8000/parse", data=dict(text=text)).json()


def get_ngram_indexes(slen, ngram=999):
    """return indexes of ngrams in sentence
    eg. "the cat sat on the mat" => [(0,1), (0,2)] = ["the", "the cat"]
    """
    ngram_indexes = [
        (x, y)
        for x, y in product(range(0, slen + 1), repeat=2)
        if y > x and y - x <= ngram
    ]
    return ngram_indexes
