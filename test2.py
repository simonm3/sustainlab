import os

prefectx_path = os.path.abspath(os.path.join(__file__, os.pardir, "prefectx"))

# extra_loggers logs to orion
os.environ.update(
    PREFECT_ORION_DATABASE_CONNECTION_TIMEOUT="60.0",
    PREFECT_LOGGING_SETTINGS_PATH=f"{prefectx_path}/logging.yml",
    PREFECT_LOGGING_EXTRA_LOGGERS="xxx,test2",
    PREFECT_API_URL="http://127.0.0.1:4200/api",
)

from prefect.flows import flow
from prefect.tasks import task

# from prefect_dask.task_runners import DaskTaskRunner as runner
from prefect.task_runners import SequentialTaskRunner as runner

# from prefect_ray.task_runners import RayTaskRunner as runner
from prefect import get_run_logger
import logging


@flow(task_runner=runner())
def testflow():
    testtask()


@task
def testtask():
    # from defaultlog import log
    log = logging.getLogger("xxx")
    log.warning(f"warning xxx {log.getEffectiveLevel()}")
    log.debug("debug xxx log")

    log = logging.getLogger()
    log.warning(f"warning root {log.getEffectiveLevel()}")
    log.debug("debug root log")

    log = get_run_logger()
    log.warning(f"warning prefect {log.getEffectiveLevel()}")
    log.debug("debug hi")

    log = logging.getLogger("test2.pppp")
    log.warning(f"warning extra {log.getEffectiveLevel()}")
    log.debug("debug extra log")


if __name__ == "__main__":
    testflow()
