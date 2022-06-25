from prefect.flows import flow
from prefect.tasks import task
import pikepdf

import logging

log = logging.getLogger(__name__)


@flow
def testflow(path):
    decrypt(path)


@task
def decrypt(path):
    log.info("starting pike")
    pdf = pikepdf.Pdf.open(path)
    log.info("finished pike")
