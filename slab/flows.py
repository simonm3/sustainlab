import dask
from prefect import tags
from prefect.client import get_client

# from prefect_dask.task_runners import DaskTaskRunner
# from prefect_ray.task_runners import RayTaskRunner

from prefect.task_runners import SequentialTaskRunner, ConcurrentTaskRunner

from . import flow, gcontext
from . import utils
from .pdftasks import *
from .roottasks import *

dask.config.set({"distributed.comm.timeouts.connect": 600})

# small scale tests faster without multiprocessing
runner = SequentialTaskRunner()
# runner = DaskTaskRunner(cluster_kwargs=dict(n_workers=1, resources=dict(process=4)))
# runner = ConcurrentTaskRunner()

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
    try:
        if isinstance(runner, DaskTaskRunner):
            setlimits("sentence", 1)
    except:
        log.exception("error setting dask limits")

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
    raw_sents = utils.filtered(f"{gcontext.path}/sentence_filter/*", sample)

    # keyword and esg models
    kwtopics_ = kwtopics(raw_sents, topic2kw())
    esg_ = esg(raw_sents)

    # select inscope
    inscope_ = inscope(kwtopics_, esg_)
    sents = inscope_.result()

    ############################################ sentences

    # sentence embeddings
    token_feats_ = token_feats(sents)
    sent_feats_ = sent_feats(token_feats_)

    # NER duckling
    ducks_ = ducks(sents)

    ########################################### kpis

    # kpi embeddings
    gcontext.path = f"working/{kpi_path}"
    kpi_token_feats_ = token_feats(kpis)
    kpi_feats_ = sent_feats(kpi_token_feats_)

    ########################################### sentences + kpis

    # best kpi embeddings based on sent and ngrams
    path = "_".join([sent_path, kpi_path])
    gcontext.path = f"working/{path}"
    compare_sents_ = compare_sents(sent_feats_, kpi_feats_, kpis)
    compare_ngrams_ = compare_ngrams(token_feats_, kpi_feats_, sents, kpis)


@flow(task_runner=runner)
def compare_clean():
    """run embeddings on clean sentences"""

    cleancols = lambda cols: [x.lower().replace(" ", "_") for x in cols]

    # kpis
    kpis = pd.read_excel(
        "SustainLab_KPIs_and_labeled_clean sentences_v2.xlsx", sheet_name="KPIs"
    )
    kpis.columns = cleancols(kpis.columns)
    kpis = kpis.rename(columns=dict(kpi_topic="topic", value_field="value"))
    kpis = kpis[["kpi", "topic", "value"]].dropna()
    kpi_feats = []
    for k in kpis.columns:
        gcontext.path = f"working/clean_{k}"
        token_feats_ = token_feats(kpis[k])
        out = sent_feats(token_feats_)
        kpi_feats.append(out)

    # sents
    labsents = pd.read_excel(
        "SustainLab_KPIs_and_labeled_clean sentences_v2.xlsx",
        sheet_name="Labled Sentences",
    )
    labsents.columns = cleancols(labsents.columns)
    labsents = labsents.rename(
        columns=dict(
            single_kpi_sentence="sent",
            kpi_label="Akpi",
            kpi_topic_label="Atopic",
            value_field="Avalue",
        )
    )
    labsents = labsents[["sent", "Akpi", "Atopic", "Avalue"]]
    sents = labsents.sent
    sent_path = "clean"
    gcontext.path = f"working/{sent_path}"
    token_feats_ = token_feats(sents)
    sent_feats_ = sent_feats(token_feats_)

    # compare
    for k, kpi_feats_ in zip(kpis.columns, kpi_feats):
        path = "_".join([sent_path, f"clean_{k}"])
        gcontext.path = f"working/{path}"
        compare_sents_ = compare_sents(sent_feats_, kpi_feats_, kpis[k].tolist())
        compare_ngrams_ = compare_ngrams(
            token_feats_, kpi_feats_, sents, kpis[k].tolist()
        )

    # aggregate sent,Akpi, Akpi_topic, Avalue_field,  kpi, kpi_topic, value_field, Mkpi, Mkpi_topic, Mvalue_field,
    out = []
    for k in kpis.columns:
        # df = pd.read_pickle(f"working/clean_clean_{k}/compare_sents")
        # df = df.rename(columns=dict(kpi_sent=k))
        df = pd.read_pickle(f"working/clean_clean_{k}/compare_ngrams")
        df = df.rename(columns=dict(kpi_ngram=k))
        out.append(df)
    out = pd.concat([labsents] + out, axis=1)
    out["Mkpi"] = out.kpi == out.Akpi
    out["Mtopic"] = out.topic == out.Atopic
    out["Mvalue"] = out.value == out.Avalue
    outcols = [x for x in out.columns if x.find("score") < 0]
    out.to_excel("output/clean_ngrams.xlsx", index=False)
