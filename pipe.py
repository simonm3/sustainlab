import pikepdf
import PyPDF2
import os
import re
import pandas as pd
from utils import nlp, spaced

import logging

log = logging.getLogger(__name__)


def decrypt(path):
    """decrypt pdf. pypdf2 cannot read encrypted even if no password
    cached as takes 12 seconds
    """
    os.makedirs("decrypted", exist_ok=True)
    with open(path, "rb") as f:
        pdf = PyPDF2.PdfFileReader(f)
    if pdf.is_encrypted:
        decrypted = f"decrypted/{os.path.basename(path)}"
        if not os.path.exists(decrypted):
            with open(decrypted, "wb") as f:
                pikepdf.Pdf.open(path).save(decrypted)
        return decrypted
    else:
        return path


def pdf2text(path, pages=9999999):
    """return text from pdf"""
    # try to extract text rather than image
    pdf = PyPDF2.PdfFileReader(path)
    pages = [p.extractText() for p in pdf.pages[:pages]]
    text = " ".join(pages)
    basename = os.path.basename(path)

    # extract pdf2text directly.
    if len(text) > 100:
        log.info(f" {basename} text pdf")
        return text

    # sustainlab provided text (from image=>pytesseract OCR)
    base, ext = os.path.splitext(path)
    try:
        with open(f"{base}.txt") as f:
            text = f.read()
            log.info(f" {basename} pytesseract text")
    except:
        log.info(f" {basename} image pdf with no provided text")
    return text


def row_filter(text):
    # remove headers
    dropped = []
    accepted = []
    for r in text.split("\n"):
        if any(
            [
                r.istitle(),
                r.isupper(),
                r.isspace(),
                r.replace(",", "").isnumeric(),
                r.find("....") >= 0,
            ]
        ):
            log.debug(f"ROW DROPPED={r}")
            dropped.append(r)
        else:
            log.debug(f"ROW ACCEPTED={r}")
            accepted.append(r)
    return accepted, dropped


def sentence_filter(rows):
    """split into sentences and filter
    rule based. nouns>=2 and (verbs+aux)>=1
    """
    text = "\n".join(rows)
    # hyphenated words
    text = text.replace("-\n", "")
    # line endings
    text = text.replace("\n\n", ".\n").replace("\n", " ")
    # multiple spaces
    text = re.sub("\s+", " ", text)

    doc = nlp(text)

    sents = []
    dropped = []
    for sent in doc.sents:
        nouns = sum([token.pos_ in ["NOUN", "PROPN", "PRON"] for token in sent])
        verbs = sum([token.pos_ in ["AUX", "VERB"] for token in sent])
        log.debug(sent.text)
        log.debug((nouns, verbs))
        if (nouns >= 2) and (verbs >= 1):
            log.debug("*********** ACCEPTED ******************")
            sents.append(sent)
        else:
            log.debug("*********** DROPPED ******************")
            dropped.append(sent)
    return sents, dropped


def quant_filter(sents):
    valid = []
    for sent in sents:
        nlp1 = nlp(sent)
        labels = [ent.label_ for ent in nlp1.ents]
        if any(x in ["CARDINAL", "ORDINAL", "PERCENT", "MONEY"] for x in labels):
            valid.append(sent)
    return valid


def get_topics(sents, topic2kw):
    """ 
    generate topics for sentences using ngram matching
    :param sents: sentences
    :return: dataframe with sentence, topic1, topic2, keywords1, keywords2
    """
    out = []
    for sent in sents:
        topics = dict()
        for k, v in topic2kw.items():
            s = " | ".join(v)
            s = spaced(s)
            ngrams = [x.strip() for x in re.findall(s, sent)]
            if len(ngrams) > 0:
                topics[k] = ngrams
        topics = dict(sorted(topics.items(), key=lambda x: len(x[1]), reverse=True))
        res = dict(sent=sent, ntopics=len(topics))
        if len(topics) >= 1:
            res["topic1"], res["keywords1"] = list(topics.items())[0]
        if len(topics) >= 2:
            res["topic2"], res["keywords2"] = list(topics.items())[1]
        out.append(res)
    return pd.DataFrame(out)
