from glob import glob

from prefect import tags
from prefect.client import get_client
from prefect.task_runners import (
    ConcurrentTaskRunner,
    DaskTaskRunner,
    SequentialTaskRunner,
)

from . import Store, flow, gcontext
from .doctasks import *
from .roottasks import *
from .pdftasks import *

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
def preprocess(files, prefix=""):
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

    # filter1 = sample, numbers, sentence structure
    log.info(f"{len(df)} sentences")
    df = df.sample(sample)
    log.info(f"{len(df)} sample sentences")
    df = df[df.accepted]
    log.info(f"{len(df)} accepted with number, verb, 2*nouns")

    ##################################################

    # kw and topic
    kwtopics_ = kwtopics(df.sent, topic2kw())
    esg_ = esg(df.sent)

    # aggregate kw/esg
    items = [kwtopics_, esg_]
    items = [i.result().load() for i in items]
    df = pd.concat(items, axis=1)
    df.loc[(df.esg_score < 0.5) & (df.ntopics == 0), "esg_topic"] = "outofscope"

    # filter2 = inscope keyword matches and score threshold
    df = df[df.esg_topic != "outofscope"]
    log.info(f"{len(df)} inscope")
    df.to_pickle("working/inscope")

    #######################################################

    # sentence embeddings
    gcontext["docname"] = "sents"
    token_feats_ = token_feats(df.index)
    sent_feats_ = sent_feats(token_feats_)

    # kpi embeddings
    kpis = pd.read_excel(
        "SustainLab_Generic_Granular_KPI list.xlsx", sheet_name="Granular KPI list"
    )
    kpis = kpis.KPI.tolist()
    gcontext["docname"] = "kpis"
    kpi_token_feats_ = token_feats(kpis)
    kpi_feats_ = sent_feats(kpi_token_feats_)

    # best kpi embeddings based on sent and ngrams
    compare_sents_ = compare_sents(sent_feats_, kpi_feats_, df.index, kpis)
    compare_ngrams_ = compare_ngrams(token_feats_, kpi_feats_, df.index, kpis)

    # NER duckling
    ducks_ = ducks(df.index)

    # aggregate results
    out = [x.result() for x in [compare_sents_, compare_ngrams_, ducks_]]
    out = [x.load() if isinstance(x, Store) else x for x in out]
    df = pd.concat([df, *out], axis=1)
    df = df[
        [
            "kpi_ngram",
            "kpi_sent",
            "kw_topic1",
            "esg_topic",
            "ducks",
            "ngram",
            "ntopics",
            "keywords1",
            "score_sent",
            "score_ngram",
            "esg_score",
        ]
    ]
    df.to_excel("output/out.xlsx")

    return df
