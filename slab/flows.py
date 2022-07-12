from glob import glob
import numpy as np

from prefectx import flow
from prefect import tags
from prefect.client import get_client
from prefect.task_runners import (
    DaskTaskRunner,
    SequentialTaskRunner,
    ConcurrentTaskRunner,
)
import dask
from prefectx import gcontext

from .preptasks import *
from .models import *

"""
TODO 
spacy tasks use a lot of RAM which prevents use of multiprocessing on single machine
    research dask n_workers, n_threads and limiting RAM usage with concurrency limits
    try kubernetes or other way to run on multiple servers
    research why concurrent much slower than sequential
"""

# runner = DaskTaskRunner(cluster_kwargs=dict(n_workers=1, resources=dict(process=4)))
# runner = ConcurrentTaskRunner()

# small scale tests faster without multiprocessing
runner = SequentialTaskRunner()

# TODO simpler way
async def setlimits(tag, limit=1):
    async with get_client() as client:
        limit_id = await client.create_concurrency_limit(
            tag=tag, concurrency_limit=limit
        )


@flow(task_runner=runner)
def add_filters(files, prefix=""):
    """splits files into sentences and adds filters

    set prefix="sl/" to use sustainlab text instead of pypdf2
    """
    # sentence uses a lot of RAM so run on own
    if isinstance(runner, DaskTaskRunner):
        setlimits("sentence", 1)
    for path in files:
        gcontext["base"] = prefix + os.path.basename(os.path.splitext(path)[0])
        text = pdf2text(path)
        text_filter_ = text_filter(text)
        with tags("sentence"):
            sents = sentence_filter(text_filter_)


@flow(task_runner=runner)
def create_features(files=None, sample=1000):
    """
    each feature is df of sentences so can be rerun separately
    :param sample: sample size for testing
    """
    # aggregate sentences
    files = files or glob("working/sentence_filter/*")
    log.info(f"{len(files)} pdfs")
    dfs = [pd.read_pickle(f) for f in files]
    df = pd.concat(dfs)
    df = df.rename(columns=dict(text="sent"))

    # filter
    log.info(f"{len(df)} sentences")
    df = df.sample(sample)
    log.info(f"{len(df)} sample sentences")
    df = df[df.accepted]
    log.info(f"{len(df)} accepted with number, verb, 2*nouns")

    # kw and topic
    topic2kw = pd.read_pickle("working/topic2kw")
    kwtopics_ = kwtopics(df, topic2kw)
    esg_ = esg(kwtopics_)
    esg_.wait()

    # aggregate kw/esg
    items = [kwtopics_, esg_]
    items = [i.result().load().set_index("sent") for i in items]
    df = pd.concat(items, axis=1).reset_index()

    # filter inscope
    df = df[df.esg != "outofscope"]
    log.info(f"{len(df)} inscope")
    df.to_pickle("inscope")

    # embedddings
    embeddings_ = embeddings(df)
    # TODO ner(df)

    # aggregate
    embeddings_.wait()
    items = [kwtopics_, esg_, embeddings_]
    items = [i.result().load().set_index("sent") for i in items]
    df = pd.concat(items, axis=1)
    # TODO df = df.add_ensemble()

    df.to_excel("output/agg.xlsx")
