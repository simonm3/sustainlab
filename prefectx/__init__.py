import os
import logging
from time import sleep
import subprocess
from threading import Thread
import yaml
import shutil
import re

log = logging.getLogger(__name__)

gcontext = {}

# TODO remove when dask bug fixed https://github.com/dask/distributed/issues/5971
os.environ["ENV MALLOC_TRIM_THRESHOLD_"] = "65536"

# TODO remove when fixed. likely there should be a timeout setting somewhere.
def keepalive():
    """poll the orion server to stop connection timeout"""

    def target():
        while True:
            subprocess.Popen("prefect storage ls >/dev/null", shell=True)
            sleep(100)

    Thread(target=target, daemon=True).start()


# TODO ideally dask or prefect will do this automatically
def setup_dask_logging():
    # read log settings
    with open(os.environ["PREFECT_LOGGING_SETTINGS_PATH"]) as f:
        logset = f.read()
    for x in set(re.findall("\${.*}", logset)):
        logset = logset.replace(x, os.environ.get(x[2:-1], "INFO"))
    logset = yaml.safe_load(logset)

    # set extra loggers
    extra_settings = logset["loggers"]["prefect.extra"].copy()
    extras = [
        x.strip()
        for x in os.environ.get("PREFECT_LOGGING_EXTRA_LOGGERS", "").split(",")
    ]
    for extra in extras:
        logset["loggers"][extra] = extra_settings

    # TODO update rather than overwrite?
    # save and copy to dask location
    HOME = os.path.expanduser("~")
    PREFECTX = os.path.abspath(os.path.dirname(__file__))
    DASK_SRC = f"{PREFECTX}/logging_daskdistributed.yaml"
    DASK_TGT = f"{HOME}/.config/dask/distributed.yaml"
    with open(DASK_SRC, "w") as f:
        f.write(yaml.dump(dict(logging=logset)))
    os.makedirs(os.path.dirname(DASK_TGT), exist_ok=True)
    shutil.copy(DASK_SRC, DASK_TGT)


#####################################################################

"""
Normally @task and @flow use prefect.skiptask if prefect installed else cache
This can be overridden for a session by the PREFECTX environment variable
  - for testing
  - to run locally without prefect

These are the options for PREFECTX
----------------------------------
function                accept data and return data. @task and @flow do nothing.
cache                   check target exists; load Items; save output; return Item
                        save=False. return data instead of saving.
                        can be tested and used without prefect but has compatible api
prefect                 accept data; return future. @task and @flow imported from prefect.
prefect.cachetask       cachetask + prefect. send all tasks to orion.
prefect.skiptask        cachetask + prefect. skip the scheduler for completed tasks.
"""


def do_nothing(fn=None, **kwargs):
    """ dummy decorator e.g. for @task, @flow """
    return fn if fn else do_nothing


# default task_type
task_type = os.environ.get("PREFECTX", "prefect.skiptask")
if task_type.startswith("prefect"):
    try:
        import prefect
    except ModuleNotFoundError:
        log.warning("prefect is disabled as not installed")
        task_type = "cachetask"

# set behaviour of @task and @flow depending on task_type
if task_type == "function":
    flow = do_nothing
    task = do_nothing
elif task_type == "cache":  ###### DEFAULT IF PREFECT NOT INSTALLED
    flow = do_nothing
    from .cache import task
elif task_type == "prefect":
    from prefect.flows import flow
    from prefect.tasks import task
elif task_type == "prefect.cachetask":
    from prefect.flows import flow
    from .cachetask import task

    task = task(skip=False)
elif task_type == "prefect.skiptask":  ###### DEFAULT IF PREFECT INSTALLED
    from prefect.flows import flow
    from .cachetask import task

    task = task(skip=True)
else:
    raise Exception(f"PREFECTX={task_type} is not valid")
log.info(f"{task_type} will be used to control task flows")

# workarounds
if task_type.startswith("prefect"):
    keepalive()
    setup_dask_logging()

