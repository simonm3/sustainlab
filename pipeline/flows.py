from prefectx.makefile import flow
from prefectx import keepalive
from prefect.task_runners import DaskTaskRunner

from .tasks import *
from . import utils


@flow(
    version=os.getenv("GIT_COMMIT_SHA"),
    task_runner=DaskTaskRunner(cluster_kwargs=dict(n_workers=4)),
)
def testflow(files):
    keepalive()
    topic2kw = utils.get_topic2kw()
    for path in files:
        text = pdf2text(path)
        rows = row_filter(text, path)
        sents = sentence_filter(rows, path)
        quant = quant_filter(sents, path)
        kwtopics = get_topics(quant, topic2kw, path)
