import logging
import os

log = logging.getLogger()

os.environ["PREFECT_LOGGING_EXTRA_LOGGERS"] = os.path.basename(
    os.path.dirname(__file__)
)
os.environ["PREFECT_API_URL"] = "http://127.0.0.1:4200/api"

# remove pytorch parallelism as conflicts with prefect2 even in sequential runner
os.environ["TOKENIZERS_PARALLELISM"] = "False"
os.environ["OMP_NUM_THREADS"] = "1"

# avoid warning message re unused weights
from transformers import logging as tlogging

tlogging.set_verbosity_error()

from pipex import task, flow, gcontext

# note still uses pipex to ignore any kwargs not used by prefect
if os.environ.get("PIPEX") == "prefect":
    from datetime import timedelta
    from functools import partial
    from prefect.tasks import task_input_hash

    task = partial(
        task, cache_key_fn=task_input_hash, cache_expiration=timedelta(days=7)
    )
