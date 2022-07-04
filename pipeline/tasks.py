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

task = task(target="working/{taskname}/{base}", name="{taskname}_{base[:8]}")
roottask = task(target="working/{taskname}", name="{taskname}")


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

    def save_src(src, base):
        """save source of text"""
        src = f"working/pdf2text_{src}"
        os.makedirs(src, exist_ok=True)
        with open(f"{src}/{base}", "w") as f:
            f.write("")

    # try to extract text rather than image
    pdf = PyPDF2.PdfFileReader(path)
    pages = [p.extractText() for p in pdf.pages[:pages]]
    text = " ".join(pages)
    base = os.path.splitext(os.path.basename(path))[0]

    # extract pdf2text directly.
    if len(text) > 100:
        save_src("text", base)
    else:
        # use provided text
        try:
            with open(f"reports/{base}.txt") as f:
                text = f.read()
                log.info(f"{base} pytesseract text")
                save_src("slab", base)
        except:
            log.info(f"{base} image pdf with no provided text")
            save_src("notext", base)

    return text


@task
def row_filter(text):
    accepted = []
    rows = text.split("\n")
    for row in rows:
        if any(
            [
                row.istitle(),
                row.isupper(),
                row.isspace(),
                row.replace(",", "").isnumeric(),
                row.find("....") >= 0,
            ]
        ):
            accepted.append(False)
        else:
            accepted.append(True)
    return pd.DataFrame(dict(text=rows, accepted=accepted))


@task
def sentence_filter(rows):
    """keep proper sentences with numbers"""
    rows = rows[rows.accepted].text
    text = "\n".join(rows)
    # hyphenated words
    text = text.replace("-\n", "")
    # line endings
    text = text.replace("\n\n", ".\n").replace("\n", " ")
    # multiple spaces, tabs and newlines
    text = re.sub("\s+", " ", text)

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


@task
def get_topics(sents, topic2kw):
    """
    generate topics for sentences using ngram matching
    :param sents: sentences
    :return: dataframe with sentence, topic1, topic2, keywords1, keywords2
    """
    sents = sents[sents.accepted].text
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
    out = pd.DataFrame(out)
    out.topic1 = out.topic1.fillna("outofscope")
    return out


@roottask
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


@roottask
def esg(df):
    from transformers import pipeline

    classifier = pipeline(model="nbroad/ESG-BERT")
    results = []
    for sent in tqdm(df.sent.tolist()):
        # truncate as bert has 512 token limit
        result = classifier(sent[:512])
        results.append(result)
    df["esg"] = [res[0]["label"] for res in results]
    df["score"] = [res[0]["score"] for res in results]

    df["esg2"] = df.esg
    df.loc[(df.score < 0.5) & (df.ntopics == 0), "esg2"] = "outofscope"
    df.topic1 = df.topic1.fillna("outofscope")

    # if running on colab then save to parquet as pickle is pandas version specific
    # df.to_parquet("working/esg.parquet")

    return df


# TODO not working
@roottask
def esg_ray(df):
    raise NotImplementedError
    from transformers import pipeline
    import psutil
    import ray

    num_cpus = psutil.cpu_count(logical=True) - 1
    ray.init(num_cpus=num_cpus, ignore_reinit_error=True)
    classifier = pipeline(model="nbroad/ESG-BERT")
    pipe_id = ray.put(classifier)

    @ray.remote
    def predict(pipe_id, sent):
        return pipe_id(sent)

    # bert only accepts 512
    sents = [s[:512] for s in df.sent.tolist()]
    results = ray.get([predict.remote(pipe_id, sent) for sent in sents])
    df["esg"] = [res[0]["label"] for res in results]
    df["score"] = [res[0]["score"] for res in results]

    return df
