from prefectx.makefile import flow
from prefectx import keepalive
from prefect.task_runners import DaskTaskRunner, SequentialTaskRunner
import dask

from .tasks import *
from . import utils

runner = DaskTaskRunner(cluster_kwargs=dict(n_workers=4, resources=dict(process=1)))
# runner = SequentialTaskRunner()


@flow(version=os.getenv("GIT_COMMIT_SHA"), task_runner=runner)
def extract(files):
    keepalive()
    topic2kw = utils.get_topic2kw()

    # restrict tasks per worker else memory issues where using spacy with lot of text
    with dask.annotate(resources=dict(process=1)):
        for path in files:
            text = pdf2text(path)
            rows = row_filter(text, path)
            sents = sentence_filter(rows, path)
            quant = quant_filter(sents, path)
            kwtopics = get_topics(quant, topic2kw, path)


@flow(version=os.getenv("GIT_COMMIT_SHA"), task_runner=runner)
def sents_flow(files):
    keepalive()
    with dask.annotate(resources=dict(process=1)):
        for path in files:
            base = os.path.splitext(os.path.basename(path))[0]
            from prefectx.filepath import Filepath

            srcpath = f"working/row_filter/{base}"
            # print(srcpath, os.path.isfile(srcpath))
            if os.path.isfile(srcpath):
                sents = sentence_filter(Filepath(srcpath), path)

