import os

os.environ["PREFECT_LOGGING_EXTRA_LOGGERS"] = "slab"
os.environ.setdefault("PREFECT_LOGGING_SETTINGS_PATH", f"{os.path.dirname(__file__)}/logging.yml")
