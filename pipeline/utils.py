import logging
import pickle
import os

import numpy as np
import pandas as pd
import spacy
import pikepdf
import PyPDF2
from tqdm.auto import tqdm
from glob import glob

log = logging.getLogger(__name__)

nlp = spacy.load("en_core_web_sm")


def decrypt_all():
    for rep in tqdm(list(glob("reports/*.pdf"))):
        try:
            decrypt(rep)
        except:
            log.warning(f"cannot read {rep}")


def decrypt(path):
    """decrypt pdf. pypdf2 cannot read encrypted even if no password
    NOTE: pikepdf does not work in prefect2
    """
    os.makedirs("working/decrypt", exist_ok=True)
    outpath = f"working/decrypt/{os.path.basename(path)}"
    if os.path.exists(outpath):
        return outpath
    if not PyPDF2.PdfFileReader(path).is_encrypted:
        return outpath

    # decrypt
    pikepdf.Pdf.open(path).save(outpath)
    return outpath


def combine(accepted, dropped):
    df1 = pd.DataFrame(dict(text=accepted))
    df1["accepted"] = True
    df2 = pd.DataFrame(dict(text=dropped))
    df2["accepted"] = False
    return pd.concat([df1, df2])


def lemmatize(text):
    words = [w.lemma_ for w in nlp(text)]
    # add space at start and end to match ngrams
    words = " ".join(words).replace(" - ", "-")
    return words


def spaced(text):
    return f" {text.strip()} "


def get_topic2kw():
    # check cache
    cache = "/mnt/d/data1/topic2kw.pkl"
    try:
        with open(cache, "rb") as f:
            return pickle.load(f)
    except:
        pass

    log.info("recreating topic2kw")
    keywords = pd.read_excel("/mnt/d/data1/keywords.xlsx")

    # get ngrams
    keywords["ngrams"] = (
        keywords.Unigrams.astype(str) + "," + keywords.Bigrams.astype(str)
    )
    keywords.ngrams = keywords.ngrams.apply(lambda x: x.split(","))
    topic2kw = dict(zip(keywords.Subtopic, keywords.ngrams))

    # fix
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

    # save to cache and return
    with open(cache, "wb") as f:
        pickle.dump(topic2kw, f)
    return topic2kw
