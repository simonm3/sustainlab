import logging

import numpy as np
import pandas as pd
from prefectx import task
from sklearn.metrics.pairwise import cosine_similarity
from tqdm.auto import tqdm
from transformers import pipeline

from . import doctask as task
from .utils import get_ngram_indexes

log = logging.getLogger(__name__)

pipe = pipeline(
    "feature-extraction",
    model="nbroad/ESG-BERT",
    padding=True,
    truncation=True,
    max_length=512,
)

@task
def token_feats(sents):
    token_feats = [
        np.array(pipe(s)).squeeze() for s in tqdm(sents, desc="token_features")
    ]
    return token_feats


@task
def sent_feats(token_feats):
    sent_feats = [x.mean(axis=0) for x in tqdm(token_feats, desc="sent_features")]
    return sent_feats


# def get_words(s):
#     """get words from a sentences
#     e.g. ["the", "cat", "sat"]
#     """
#     s = s.lower()
#     s = re.sub(r"[^a-zA-Z0-9\s]", " ", s)
#     tokens = [token for token in s.split(" ") if token != ""]
#     return tokens


# def get_ngrams(tokens, maxn=999):
#     """get ngrams from list of words e.g. "the cat" """
#     ngrams = []
#     for start in range(len(tokens)):
#         for end in range(start, len(tokens)):
#             if end + 1 - start > maxn:
#                 break
#             ngram = " ".join(tokens[start : end + 1])
#             ngrams.append(ngram)
#     return list(set(ngrams))
