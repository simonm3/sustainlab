import logging


log = logging.getLogger(__name__)

from prefect import flow, task

@task
def say_hello():
    log.info("running the task *******")
    print("Hello, World! I'm Marvin!")


@flow
def marvin_flow():
    log.info("running the flow ******** ")
    say_hello()

marvin_flow()  # "Hello, World! I'm Marvin!"

