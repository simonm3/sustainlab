import os
import logging
from functools import partial
from datetime import timedelta

log = logging.getLogger(__name__)

# dummy decorator
def do_nothing(fn=None, **kwargs):
    return fn if fn else do_nothing

try:
    if os.environ.get("DISABLE_PREFECT")=="True":
        raise
    from prefect import task, flow, Task
    from prefect.tasks import task_input_hash

    # cache results
    # expire = None
    expire = timedelta(minutes=1)
    task = partial(task, cache_key_fn=task_input_hash, cache_expiration=expire)
    
    # automatically submit to enable same code to be used with or without prefect
    Task.__call__ = Task.submit

except:
    log.warning("not using prefect")
    task = do_nothing
    flow = do_nothing
