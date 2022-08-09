import logging
import re

import pandas as pd
import spacy
from tqdm.auto import tqdm
from transformers import pipeline

from ..prefectx import task

log = logging.getLogger(__name__)

# only need tagger and lemmatizer
nlp = spacy.load("en_core_web_sm")
nlp.max_length = int(2e6)


def lemmatize(text):
    words = list(nlp.pipe([text], n_process=1))[0]
    words = [w.lemma_ for w in words]
    # add space at start and end to match ngrams
    words = " ".join(words).replace(" - ", "-")
    return words


def spaced(text):
    return f" {text.strip()}"


@task
def sentence_filter(text):
    """keep proper sentences with numbers"""
    text = nlp(text)

    accepted = []
    for sent in tqdm(list(text.sents), desc="sentence_filter"):
        nouns = sum([token.pos_ in ["NOUN", "PROPN", "PRON"] for token in sent])
        verbs = sum([token.pos_ in ["AUX", "VERB"] for token in sent])
        labels = [ent.label_ for ent in sent.ents]
        if (
            any(x in ["CARDINAL", "PERCENT", "MONEY"] for x in labels)
            and (nouns >= 2)
            and (verbs >= 1)
            # TODO improve
            and len(re.findall("[0-9]", str(sent))) > 0
        ):
            accepted.append(True)
        else:
            accepted.append(False)
    sents = [str(s) for s in text.sents]
    return pd.DataFrame(dict(text=sents, accepted=accepted))


@task
def topic2kw():
    """convert keywords sheet to topic2kw dict"""
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
    classifier = pipeline(model="nbroad/ESG-BERT", truncation=True, max_length=512,)
    results = []
    for sent in tqdm(sents, desc="esg"):
        result = classifier(sent)
        results.append(result)

    df = pd.DataFrame(index=sents)
    df["esg_topic"] = [res[0]["label"] for res in results]
    df["esg_score"] = [res[0]["score"] for res in results]

    return df


@task
def inscope(kwtopics, esg):
    """select inscope based on kwtopic and esg"""
    df = pd.concat([kwtopics, esg], axis=1)
    df.loc[(df.esg_score < 0.5) & (df.ntopics == 0), "esg_topic"] = "outofscope"
    df = df[df.esg_topic != "outofscope"]
    log.info(f"{len(df)} inscope")

    return df
