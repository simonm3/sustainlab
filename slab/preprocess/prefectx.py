from ast import Import
import os
import logging
from functools import partial

log = logging.getLogger(__name__)

# dummy decorator
def do_nothing(fn=None, **kwargs):
    return fn if fn else do_nothing

try:
    if os.environ.get("DISABLE_PREFECT")=="True":
        raise
    from prefect import task, flow
    from prefect.tasks import task_input_hash

    # cache results by default
    task = partial(task, cache_key_fn=task_input_hash)
except:
    log.warning("not using prefect")
    task = do_nothing
    flow = do_nothing
