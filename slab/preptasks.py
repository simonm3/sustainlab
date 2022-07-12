import os
import re
import pandas as pd
from .utils import nlp, spaced, lemmatize
from prefectx import task
import PyPDF2
import pikepdf
import shutil
from tqdm.auto import tqdm

import logging

log = logging.getLogger(__name__)

task = task(target="working/{funcname}/{base}", name="{funcname}_{base[:8]}")
roottask = task(target="working/{funcname}", name="{funcname}")


@task(store=None, target="reports_decrypted/{os.path.basename(path)}")
def decrypt(path):
    """decrypt pdf. pypdf2 cannot read encrypted even if no password"""
    outpath = f"reports_decrypted/{os.path.basename(path)}"
    if PyPDF2.PdfFileReader(path).is_encrypted:
        pikepdf.Pdf.open(path).save(outpath)
    else:
        shutil.move(path, outpath)

    return outpath


@task
def pdf2text(path, pages=9999999):
    """return text from pdf"""

    # try to extract text rather than image
    pdf = PyPDF2.PdfFileReader(path)
    pages = [p.extractText() for p in pdf.pages[:pages]]
    text = " ".join(pages)
    if len(text) < 100:
        raise Exception(f"cannot find text for {path}")

    return text


@task
def pdf2text_sl(path):
    """ text provided by sustainlab """
    base = os.path.splitext(os.path.basename(path))[0]
    with open(f"reports/{base}.txt", encoding="latin1") as f:
        text = f.read()
    return text


@task
def text_filter(text):
    # multiline hyphenated words
    text = re.sub("\s*\-\s*\n", "", text)
    # blank lines
    text = re.sub("\n\s*\n", "\n", text)
    # join lines
    text = re.sub("\s*\n\s*", " ", text)

    return text


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
        ):
            accepted.append(True)
        else:
            accepted.append(False)
    sents = [str(s) for s in text.sents]
    return pd.DataFrame(dict(text=sents, accepted=accepted))


@roottask
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

