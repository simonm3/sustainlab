import os
from functools import partial
import logging
from time import sleep
import subprocess
from threading import Thread
import json
from cloudpickle import pickle
import base64

log = logging.getLogger(__name__)

# TODO remove when dask bug fixed https://github.com/dask/distributed/issues/5971
os.environ["ENV MALLOC_TRIM_THRESHOLD_"] = "65536"

# set flag for prefect use
prefect = True
if "PREFECT_DISABLED" in os.environ:
    log.warning(
        "prefect is disabled as PREFECT_DISABLED environment variable exists"
    )
    prefect = False
else:
    try:
        import prefect
    except ModuleNotFoundError:
        log.warning("prefect is disabled as not installed")
        prefect = False

# TODO remove when fixed. likely there should be a timeout setting somewhere.
def keepalive():
    """poll the orion server to stop connection timeout"""

    def target():
        while True:
            subprocess.Popen("prefect storage ls >/dev/null", shell=True)
            sleep(100)

    Thread(target=target, daemon=True).start()


def load_cache(filename):
    """load file from cache"""
    res = json.load(open(filename))
    blob = res["blob"]
    return pickle.loads(base64.b64decode(blob))

if prefect:
    from .makefile import task
else:
    from .filetask import filetask as task
