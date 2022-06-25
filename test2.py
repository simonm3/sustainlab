import os

os.environ.update(
    PREFECT_ORION_DATABASE_CONNECTION_TIMEOUT="60.0",
    PREFECT_LOGGING_EXTRA_LOGGERS="test1",
    PREFECT_API_URL="http://127.0.0.1:4200/api",
)

from prefect.flows import flow
from prefect.tasks import task

from prefect_dask.task_runners import DaskTaskRunner as runner

# from prefect.task_runners import SequentialTaskRunner as runner
# from prefect_ray.task_runners import RayTaskRunner as runner
from prefect import get_run_logger
import logging


@flow(task_runner=runner())
def testflow():
    testtask()


@task
def testtask():
    log = get_run_logger()
    log.warning("warning hi")
    log.debug("debug hi")

    log = logging.getLogger()
    log.warning("warning std root log")
    log.debug("debug std root log")

    log = logging.getLogger("test1")
    log.warning("warning extra log")
    log.debug("debug extra log")


if __name__ == "__main__":
    testflow()
