import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from tqdm.auto import tqdm
from itertools import product
import pandas as pd
import logging

log = logging.getLogger(__name__)


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


class Document:
    def __init__(self, sents, pipe):
        self.sents = sents
        self.pipe = pipe

    def get_feats(self):
        self.token_feats = [np.array(self.pipe(s)).squeeze() for s in tqdm(self.sents)]
        self.sent_feats = [x.mean(axis=0) for x in self.token_feats]

    def compare_sents(self, target):
        """return sents * best"""
        res = cosine_similarity(self.sent_feats, target.sent_feats)
        kpi = [target.sents[x] for x in res.argmax(axis=1)]
        score = res.max(axis=1)
        df = pd.DataFrame(dict(kpi=kpi, score=score), index=self.sents)
        df.index.name = "sent"
        return df

    def compare_ngrams(self, i, target, ngram_limit=999):
        """
        compare a single sentence against ngrams
        :param i: index in sents
        :param target: another document
        :param ngram_limit: e.g. 4 => up to 4grams
        :return: ngram_best, kpi_best, score_best, df (for checking ngrams for a sentence)
        """
        sent = self.sents[i]
        tokens = self.pipe.tokenizer.encode(sent)
        token_feat = self.token_feats[i]

        # get ngrams. note index to tokens is translated to text
        ngram_indexes = get_ngram_indexes(len(token_feat), ngram_limit)
        ngram_feats = [token_feat[x:y].mean(axis=0) for x, y in ngram_indexes]
        ngrams = [self.pipe.tokenizer.decode(tokens[x:y]) for x, y in ngram_indexes]

        # compare ngram features to target sentence features
        res = cosine_similarity(ngram_feats, target.sent_feats)

        # best match ngram text, kpi text, score
        ix = np.unravel_index(res.argmax(), res.shape)
        ngram_best, kpi_best = ngrams[ix[0]], target.sents[ix[1]]
        score_best = res.max()

        # only used for checking results of sentence
        ngramdf = self.get_ngramdf(res, ngrams, target)

        return ngram_best, kpi_best, score_best, ngramdf

    def get_ngramdf(self, res, ngrams, target):
        """get dataframe of all matches for debugging"""

        # create dataframe
        df = pd.DataFrame(res, index=ngrams, columns=target.sents)
        df = df.melt(ignore_index=False)

        # sort and filter
        df.index.name = "ngram"
        df = df.reset_index()
        df = df.sort_values(["ngram", "value"], ascending=False)
        df = df[df.value > 0.5]

        # presentation
        df.columns = ["ngram", "best", "score"]
        df = df.set_index("ngram").sort_values("score", ascending=False)

        return df


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
