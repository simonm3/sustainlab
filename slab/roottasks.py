import logging
import re

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from tqdm.auto import tqdm
from transformers import pipeline, PreTrainedTokenizerFast

from . import roottask as task
from .utils import get_ngram_indexes, lemmatize, spaced, duck

log = logging.getLogger(__name__)


@task
def topic2kw():
    log.info("start")
    keywords = pd.read_excel("keywords.xlsx")

    # get ngrams
    keywords["ngrams"] = (
        keywords.Unigrams.astype(str) + "," + keywords.Bigrams.astype(str)
    )
    keywords.ngrams = keywords.ngrams.apply(lambda x: x.split(","))
    topic2kw = dict(zip(keywords.Subtopic, keywords.ngrams))
    log.info("got ngrams")

    # adjustments
    for k, v in topic2kw.items():
        # correct spelling errors
        out = [x.replace("_", " ").strip() for x in v]
        spell = dict(abseteeism="absenteeism", laeble="label", _emissions="emissions")
        out = [spell.get(x, x) for x in out]

        # lemmatize
        out = [lemmatize(x) for x in out]

        # remove common words that have mixed meanings
        out = [
            x for x in out if x not in ["", "nan", "good", "material", "right", "cycle"]
        ]

        # remove duplicates
        topic2kw[k] = sorted(list(set(out)))
    log.info("done adjustments")

    # remove overlapping
    for k, v in topic2kw.items():
        newv = v.copy()
        for v1 in v:
            for v2 in v:
                if (v1 != v2) and (spaced(v1) in spaced(v2)) and (v2 in newv):
                    # log.info(f"removing {v2} due to {v1}")
                    newv.remove(v2)
        topic2kw[k] = newv
    log.info("finishing")

    return topic2kw


@task
def kwtopics(sents, topic2kw):
    """
    generate topics for sentences using ngram matching
    :param sents: collection of sentences
    :return: dataframe with sentence, kw_topic1, topic2, keywords1, keywords2
    """
    out = []
    for sent in tqdm(sents, desc="kwtopics"):
        topics = dict()
        for k, v in topic2kw.items():
            s = " | ".join(v)
            s = spaced(s)
            ngrams = [x.strip() for x in re.findall(s, lemmatize(sent))]
            ngrams = list(set(ngrams))
            if len(ngrams) > 0:
                topics[k] = ngrams
        topics = dict(sorted(topics.items(), key=lambda x: len(x[1]), reverse=True))
        res = dict(ntopics=len(topics))
        if len(topics) >= 1:
            res["kw_topic1"], res["keywords1"] = list(topics.items())[0]
        # if len(topics) >= 2:
        #     res["topic2"], res["keywords2"] = list(topics.items())[1]
        out.append(res)
    df = pd.DataFrame(out, index=sents)
    df.kw_topic1 = df.kw_topic1.fillna("outofscope")
    return df


@task
def esg(sents):
    classifier = pipeline(model="nbroad/ESG-BERT")
    results = []
    for sent in tqdm(sents, desc="esg"):
        # truncate as bert has 512 token limit
        result = classifier(sent[:512])
        results.append(result)

    df = pd.DataFrame(index=sents)
    df["esg_topic"] = [res[0]["label"] for res in results]
    df["esg_score"] = [res[0]["score"] for res in results]

    return df


@task
def compare_sents(sent_feats, kpi_feats, sents, kpis):
    """add best kpi and score to sents"""
    res = cosine_similarity(sent_feats, kpi_feats)
    kpi = [kpis[x] for x in res.argmax(axis=1)]
    score = res.max(axis=1)
    log.info((len(kpi), len(score), len(sents)))
    return pd.DataFrame(dict(kpi_sent=kpi, score_sent=score), index=sents)


@task
def compare_ngrams(sent_token_feats, kpi_feats, sents, kpis):
    # ngrams embedding
    res = []
    zipped = zip(sent_token_feats, sents)
    for sent_token_feats, sent in tqdm(zipped, total=len(sents), desc="compare ngrams"):
        ngram, kpi, score, _ = compare_ngrams_s(sent_token_feats, kpi_feats, sent, kpis)
        res.append([ngram, kpi, score])
    ngramdf = pd.DataFrame(
        res, columns=["ngram", "kpi_ngram", "score_ngram"], index=sents
    )

    return ngramdf


@task
def ducks(sents):
    """duckling NER"""
    ducks = [duck(x) for x in sents]
    ducks = pd.DataFrame(dict(ducks=ducks), index=sents)
    return ducks


def compare_ngrams_s(token_feats, kpi_feats, sent, kpis, ngram_limit=5):
    """
    compare ngrams in a sentence against kpis
    :param sent_token_feats: token features for sentence
    :param kpi_feats: features for kpis
    :param sent: sentence text
    :param kpis: text of kpis for adding to ngramdf
    :param ngram_limit: max number of tokens in ngrams
    :return: best ngram, kpi, score; ngramdf (ngram * target sentence * score)
    """
    tokenizer = PreTrainedTokenizerFast.from_pretrained("nbroad/ESG-BERT")
    tokens = tokenizer.encode(sent)

    # get ngrams. note index to tokens is translated to text
    ngram_indexes = get_ngram_indexes(len(token_feats), ngram_limit)
    ngram_feats = [token_feats[x:y].mean(axis=0) for x, y in ngram_indexes]
    ngrams = [tokenizer.decode(tokens[x:y]) for x, y in ngram_indexes]

    # compare ngram features to target sentence features
    res = cosine_similarity(ngram_feats, kpi_feats)

    # best match ngram text, kpi text, score
    ix = np.unravel_index(res.argmax(), res.shape)
    ngram, kpi = ngrams[ix[0]], kpis[ix[1]]
    score = res.max()

    # for testing to check scores for all ngrams
    ngramdf = pd.DataFrame()  # get_ngramdf(res, ngrams, kpis)

    return ngram, kpi, score, ngramdf


def get_ngramdf(res, ngrams, kpis):
    """get dataframe of all matches for debugging"""

    # create dataframe
    df = pd.DataFrame(res, index=ngrams, columns=kpis)
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
