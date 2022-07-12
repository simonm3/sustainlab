import re
import pandas as pd
from .utils import spaced, lemmatize
from .document import Document
from tqdm.auto import tqdm
from transformers import pipeline
from prefectx import task

import logging

log = logging.getLogger(__name__)

modeltask = task(target="models/{funcname}", name="{funcname}")


@modeltask
def kwtopics(df, topic2kw):
    """
    generate topics for sentences using ngram matching
    :param df: dataframe with sent
    :return: dataframe with sentence, topic1, topic2, keywords1, keywords2
    """
    df = df[["sent"]]
    out = []
    for sent in tqdm(df.sent, desc="kwtopics"):
        topics = dict()
        for k, v in topic2kw.items():
            s = " | ".join(v)
            s = spaced(s)
            ngrams = [x.strip() for x in re.findall(s, lemmatize(sent))]
            ngrams = list(set(ngrams))
            if len(ngrams) > 0:
                topics[k] = ngrams
        topics = dict(sorted(topics.items(), key=lambda x: len(x[1]), reverse=True))
        res = dict(sent=sent, ntopics=len(topics))
        if len(topics) >= 1:
            res["topic1"], res["keywords1"] = list(topics.items())[0]
        # if len(topics) >= 2:
        #     res["topic2"], res["keywords2"] = list(topics.items())[1]
        out.append(res)
    out = pd.DataFrame(out)
    out.topic1 = out.topic1.fillna("outofscope")
    return out


@modeltask
def esg(df):
    from transformers import pipeline

    df = df[["sent", "ntopics"]]
    sents = df.sent.tolist()

    classifier = pipeline(model="nbroad/ESG-BERT")
    results = []
    for sent in tqdm(sents, desc="esg"):
        # truncate as bert has 512 token limit
        result = classifier(sent[:512])
        results.append(result)
    df["esg"] = [res[0]["label"] for res in results]
    df["score"] = [res[0]["score"] for res in results]
    df.loc[(df.score < 0.5) & (df.ntopics == 0), "esg"] = "outofscope"
    df = df.drop("ntopics", axis=1)

    # if running on colab then save to parquet as pickle is pandas version specific
    # df.to_parquet("working/esg.parquet")

    return df


@modeltask
def embeddings(df):
    # load data
    sents = df.sent
    sents = Document(sents)
    kpis = pd.read_excel(
        "SustainLab_Generic_Granular_KPI list.xlsx", sheet_name="Granular KPI list"
    ).KPI.tolist()
    kpis = Document(kpis)

    # get features
    sents.get_feats()
    kpis.get_feats()

    # sentence embedding
    sentdf = sents.compare_sents(kpis)

    # ngrams embedding
    res = []
    for i, sent in enumerate(tqdm(sents.sents)):
        ngram, kpi, score, _ = sents.compare_ngrams(i, kpis)
        res.append([sent, ngram, kpi, score])
    ngramdf = pd.DataFrame(res, columns=["sent", "ngram", "kpi", "score"])
    ngramdf = ngramdf.set_index("sent")

    out = ngramdf.join(sentdf, rsuffix="_sent")[
        ["ngram", "kpi", "score", "kpi_sent", "score_sent"]
    ]

    return out
