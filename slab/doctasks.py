import logging

import numpy as np
from prefectx import task
from tqdm.auto import tqdm
from transformers import pipeline

from . import doctask as task

log = logging.getLogger(__name__)

pipe = pipeline(
    "feature-extraction",
    model="nbroad/ESG-BERT",
    truncation=True,
    max_length=512,
)


@task
def token_feats(df):
    df["token_feats"] = [
        np.array(pipe(s)).squeeze() for s in tqdm(df.index, desc="token_features")
    ]
    return df[["token_feats"]]


@task
def sent_feats(df):
    df["sent_feats"] = [
        x.mean(axis=0) for x in tqdm(df.token_feats, desc="sent_features")
    ]
    return df[["sent_feats"]]
