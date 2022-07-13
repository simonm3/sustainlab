import logging
import os
import re
import shutil

import pandas as pd
import pikepdf
import PyPDF2
from prefectx import task
from tqdm.auto import tqdm

from . import decrypttask as task
from .utils import nlp

log = logging.getLogger(__name__)


@task
def decrypt(path):
    """decrypt pdf. pypdf2 cannot read encrypted even if no password"""
    outpath = f"reports_decrypted/{os.path.basename(path)}"
    if PyPDF2.PdfFileReader(path).is_encrypted:
        pikepdf.Pdf.open(path).save(outpath)
    else:
        shutil.move(path, outpath)

    return outpath


from . import pdftask as task


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
    """text provided by sustainlab"""
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
