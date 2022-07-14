import os
from datetime import timedelta
import logging

log = logging.getLogger()

os.environ["PREFECT_LOGGING_EXTRA_LOGGERS"] = "slab"
# os.environ["PREFECT_API_URL"] = "http://127.0.0.1:4200/api"

# remove any parallelism in pytorch to avoid a clash
# TODO does this also disable prefect parallelism?
os.environ["TOKENIZERS_PARALLELISM"] = "False"
os.environ["OMP_NUM_THREADS"] = "1"

from transformers import logging as tlogging

# avoid warning message re unused weights
tlogging.set_verbosity_error()

if os.environ.get("PREFECTX", "prefect") == "prefect":
    # prefect2 better for production as well resourced and maintained
    from prefect import task, flow
    from prefect.tasks import task_input_hash

    task = task(cache_key_fn=task_input_hash, cache_expiration=timedelta(days=7))

    decrypttask = task
    pdftask = task
    roottask = task
    doctask = task
    gcontext = dict()

    class Store:
        pass

else:
    # prefectx better for development as saves interim outputs for checking
    from prefectx import task, flow, gcontext
    from prefectx.store import Store

    decrypttask = task(store=None, target="reports_decrypted/{os.path.basename(path)}")
    pdftask = task(target="working/{funcname}/{base}", name="{funcname}_{base[:8]}")
    roottask = task(target="working/{funcname}", name="{funcname}")
    doctask = task(target="working/{docname}_{funcname}", name="{funcname}")
