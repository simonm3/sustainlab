import dask
from prefect import tags
from prefect.client import get_client
from prefect.task_runners import (
    ConcurrentTaskRunner,
    DaskTaskRunner,
    SequentialTaskRunner,
)

from . import flow, gcontext
from . import utils
from .pdftasks import *
from .roottasks import *

dask.config.set({"distributed.comm.timeouts.connect": 600})
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
def preprocess(files):
    """splits files into sentences and adds filters
    :param files: pdf files
    """
    # sentence uses a lot of RAM so run on own
    if isinstance(runner, DaskTaskRunner):
        setlimits("sentence", 1)
    for path in files:
        gcontext["base"] = os.path.basename(os.path.splitext(path)[0])
        text = pdf2text(path)
        text_filter_ = text_filter(text)
        with tags("sentence"):
            sents = sentence_filter(text_filter_)


@flow(task_runner=runner)
def create_features(sent_path, kpi_path, kpis, sample=1000):
    """
    :param sents_path: path to sentences
    :param sample: number pre-filter sentences. post-filter will be approx 1/4 of that.
    """
    gcontext.path = f"working/{sent_path}"

    # filter and sample
    np.random.seed(0)
    df = utils.filtered(f"{gcontext.path}/sentence_filter/*", sample)

    # keyword and esg models
    kwtopics_ = kwtopics(df.sent, topic2kw())
    esg_ = esg(df.sent)

    # select inscope
    inscope_ = inscope(kwtopics_, esg_)

    # sentence embeddings
    token_feats_ = token_feats(inscope_)
    sent_feats_ = sent_feats(token_feats_)

    # NER duckling
    ducks_ = ducks(inscope_)

    ###########################################

    # kpi embeddings
    kpis = pd.DataFrame(index=kpis)
    gcontext.path = f"working/{kpi_path}"
    kpi_token_feats_ = token_feats(kpis)
    kpi_feats_ = sent_feats(kpi_token_feats_)

    ###########################################

    # best kpi embeddings based on sent and ngrams
    gcontext.path = "_".join([sent_path, kpi_path])
    gcontext.path = f"working/{gcontext.path}"
    compare_sents_ = compare_sents(sent_feats_, kpi_feats_)
    compare_ngrams_ = compare_ngrams(token_feats_, kpi_feats_)
