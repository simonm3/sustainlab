from prefectx import flow
from prefect import tags
from prefect.task_runners import (
    DaskTaskRunner,
    SequentialTaskRunner,
    ConcurrentTaskRunner,
)
import dask
from prefectx import gcontext

from .tasks import *

# runner = DaskTaskRunner(cluster_kwargs=dict(n_workers=1, resources=dict(process=4)))
runner = SequentialTaskRunner()
# runner = ConcurrentTaskRunner()


@flow(version=os.getenv("GIT_COMMIT_SHA"), task_runner=runner)
def extract(files):
    with tags("sentence"):
        topic2kw_ = topic2kw()

    for path in files:
        gcontext["base"] = os.path.basename(os.path.splitext(path)[0])
        text = pdf2text(path)
        text_filter_ = text_filter(text)
        with tags("sentence"):
            sents = sentence_filter(text_filter_)
        kwtopics = get_topics(sents, topic2kw_)


@flow(version=os.getenv("GIT_COMMIT_SHA"), task_runner=runner)
def extract_sl(files):
    for path in files:
        gcontext["base"] = os.path.basename(os.path.splitext(path)[0])
        sl = pdf2text_sl(path)
