import os

PREFECTX = os.path.abspath(os.path.join(__file__, os.pardir, os.pardir, "prefectx"))

# extra_loggers logs to orion
os.environ.update(
    PREFECT_ORION_DATABASE_CONNECTION_TIMEOUT="60.0",
    PREFECT_LOGGING_SETTINGS_PATH=f"{PREFECTX}/logging.yml",
    PREFECT_LOGGING_EXTRA_LOGGERS="pipeline",
    PREFECT_API_URL="http://127.0.0.1:4200/api",
)
