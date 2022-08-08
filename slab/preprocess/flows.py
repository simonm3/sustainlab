from . import extract
from . import filter
from . import flow
import logging

log = logging.getLogger(__name__)


@flow
def flow1(pdf, first_page, last_page):

    log.info("decrypt")
    pdf = extract.decrypt(pdf)

    log.info("pdf2text")
    text = extract.pdf_to_text(pdf, first_page, last_page)

    log.info("filter")
    sents = filter.sentence_filter(text)
    sents = sents[sents.accepted].text

    log.info("kwtopics")
    topic2kw = filter.topic2kw()
    kwtopics = filter.kwtopics(sents, topic2kw)

    log.info("esg")
    esg = filter.esg(sents)

    log.info("inscope")
    df = filter.inscope(kwtopics, esg)

    return df
