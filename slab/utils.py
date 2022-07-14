import logging
from itertools import product

import requests
import spacy
import pandas as pd

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
    """# docker run -d --name duckling --restart=always --port 8000:8000 rasa/duckling"""
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


def filtered(files, sample=1000):
    # aggregate sentences
    log.info(f"{len(files)} pdfs")
    dfs = [pd.read_pickle(f) for f in files]
    df = pd.concat(dfs)
    df = df.rename(columns=dict(text="sent"))

    # filter1 = sample, numbers, sentence structure
    log.info(f"{len(df)} sentences")
    df = df.sample(sample)
    log.info(f"{len(df)} sample sentences")
    df = df[df.accepted]
    log.info(f"{len(df)} accepted with number, verb, 2*nouns")
    return df


def aggregate():
    """aggregate results"""
    out = [
        pd.read_pickle(f"working/{x}")
        for x in ["inscope", "compare_sents", "compare_ngrams", "ducks"]
    ]
    df = pd.concat(out, axis=1)
    df = df[
        [
            "kpi_ngram",
            "kpi_sent",
            "kw_topic1",
            "esg_topic",
            "ducks",
            "ngram",
            "ntopics",
            "keywords1",
            "score_sent",
            "score_ngram",
            "esg_score",
        ]
    ]
    df.to_excel("output/out.xlsx")

    return df
