from prefectx import flow
from prefect.task_runners import (
    DaskTaskRunner,
    SequentialTaskRunner,
    ConcurrentTaskRunner,
)
import dask
from prefectx import gcontext

from .tasks import *

runner = DaskTaskRunner(cluster_kwargs=dict(n_workers=1, resources=dict(process=4)))
# runner = SequentialTaskRunner()
# runner = ConcurrentTaskRunner()


@flow(version=os.getenv("GIT_COMMIT_SHA"), task_runner=runner)
def extract(files):
    topic2kw_ = topic2kw()

    # restrict tasks per worker else memory issues where using spacy with lot of text
    with dask.annotate(resources=dict(process=1)):
        for path in files:
            gcontext["base"] = os.path.basename(os.path.splitext(path)[0])
            text = pdf2text(path)
            rows = row_filter(text)
            with dask.annotate(resources=dict(process=4)):
                sents = sentence_filter(rows)
            kwtopics = get_topics(sents, topic2kw_)


@flow(version=os.getenv("GIT_COMMIT_SHA"), task_runner=runner)
def sents_flow(files):
    with dask.annotate(resources=dict(process=1)):
        for path in files:
            base = os.path.splitext(os.path.basename(path))[0]
            from prefectx.stores import Filestore

            srcpath = f"working/row_filter/{base}"
            # print(srcpath, os.path.isfile(srcpath))
            if os.path.isfile(srcpath):
                sents = sentence_filter(Filestore(srcpath), path)
