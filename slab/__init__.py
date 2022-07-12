import os

os.environ["PREFECT_LOGGING_EXTRA_LOGGERS"] = "slab"
# os.environ["PREFECT_API_URL"] = "http://127.0.0.1:4200/api"
os.environ["TOKENIZERS_PARALLELISM"] = "False"
