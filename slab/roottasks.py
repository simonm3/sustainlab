import logging
import re

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from tqdm.auto import tqdm
from transformers import pipeline, PreTrainedTokenizerFast

from . import task
from .utils import get_ngram_indexes, lemmatize, spaced, duck

log = logging.getLogger(__name__)

pipe = pipeline(
    "feature-extraction",
    model="nbroad/ESG-BERT",
    truncation=True,
    max_length=512,
)

task = task(target="{path}/{funcname}", name_template="{funcname}")

@task
def inscope(kwtopics, esg):
    """select inscope based on kwtopic and esg"""
    df = pd.concat([kwtopics, esg], axis=1)
    df.loc[(df.esg_score < 0.5) & (df.ntopics == 0), "esg_topic"] = "outofscope"
    df = df[df.esg_topic != "outofscope"]
    log.info(f"{len(df)} inscope")

    return df


@task
def topic2kw():
    keywords = pd.read_excel("keywords.xlsx")

    # get ngrams
    keywords["ngrams"] = (
        keywords.Unigrams.astype(str) + "," + keywords.Bigrams.astype(str)
    )
    keywords.ngrams = keywords.ngrams.apply(lambda x: x.split(","))
    topic2kw = dict(zip(keywords.Subtopic, keywords.ngrams))

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

    # remove overlapping
    for k, v in topic2kw.items():
        newv = v.copy()
        for v1 in v:
            for v2 in v:
                if (v1 != v2) and (spaced(v1) in spaced(v2)) and (v2 in newv):
                    # log.info(f"removing {v2} due to {v1}")
                    newv.remove(v2)
        topic2kw[k] = newv

    return topic2kw


@task
def kwtopics(sents, topic2kw):
    """
    generate topics for sentences using keywords matching
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
    classifier = pipeline(
        model="nbroad/ESG-BERT",
        truncation=True,
        max_length=512,
    )
    results = []
    for sent in tqdm(sents, desc="esg"):
        result = classifier(sent)
        results.append(result)

    df = pd.DataFrame(index=sents)
    df["esg_topic"] = [res[0]["label"] for res in results]
    df["esg_score"] = [res[0]["score"] for res in results]

    return df

@task
def token_feats(df):
    df["token_feats"] = [
        np.array(pipe(s)).squeeze() for s in tqdm(df.index, desc="token_features")
    ]
    return df[["token_feats"]]


@task
def sent_feats(df):
    df["sent_feats"] = [
        x.mean(axis=0) for x in tqdm(df.token_feats, desc="sent_features")
    ]
    return df[["sent_feats"]]


@task
def compare_sents(df1, df2):
    """return best sentence embedding"""
    res = cosine_similarity(df1.sent_feats.tolist(), df2.sent_feats.tolist())
    df1["kpi_sent"] = [df2.index[x] for x in res.argmax(axis=1)]
    df1["score_sent"] = res.max(axis=1)
    return df1[["kpi_sent", "score_sent"]]


@task
def compare_ngrams(df1, df2):
    """return best ngram embedding"""
    res = []
    for sent, row in tqdm(df1.iterrows(), total=len(df1), desc="compare ngrams"):
        ngram, kpi, score, _ = compare_ngrams_s(
            row.token_feats, df2.sent_feats.tolist(), sent, df2.index
        )
        res.append([ngram, kpi, score])
    ngramdf = pd.DataFrame(
        res, columns=["ngram", "kpi_ngram", "score_ngram"], index=df1.index
    )

    return ngramdf


@task
def ducks(df):
    """duckling NER"""
    df["ducks"] = [duck(x) for x in df.index]
    return df[["ducks"]]


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
