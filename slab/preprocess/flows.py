from . import extract
from . import filter
from ..prefectx import flow
import logging

log = logging.getLogger(__name__)

def flow1(pdf, first_page, last_page):

    # at top because no upstream
    topic2kw = filter.topic2kw()

    # extract images
    pdf = extract.decrypt(pdf)
    images = extract.pdf2images(pdf, first_page, last_page)
    images = images.wait().result()

    # ocr
    all_text = []
    for image in images:
        text = extract.image2text(image)
        all_text.append(text)

    # merge and filter
    merged = extract.merge_pages(all_text)
    sents = filter.sentence_filter(merged)
    sents = sents.wait().result()
    sents = sents[sents.accepted].text

    # inscope
    kwtopics = filter.kwtopics(sents, topic2kw)
    esg = filter.esg(sents)
    df = filter.inscope(kwtopics, esg)

    return df
