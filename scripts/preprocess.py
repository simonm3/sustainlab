import os
import warnings

###########################################################
# TODO SETTINGS FOR slab.__init__.py once finalised
os.environ["PREFECT_LOGGING_SETTINGS_PATH"] = f"{os.path.dirname(__file__)}/logging.yml"
# os.environ["PREFECT_API_URL"] = "http://127.0.0.1:4200/api"
# os.environ["DISABLE_PREFECT"] = "True"
# TODO may be needed if there is a conflict
# os.environ["TOKENIZERS_PARALLELISM"] = "False"
# os.environ["OMP_NUM_THREADS"] = "1"

#######################################################
# import after settings
from slab.prefectx import flow
from slab.preprocess import flows
from prefect.task_runners import SequentialTaskRunner, ConcurrentTaskRunner
from prefect_dask import DaskTaskRunner

# TODO use decorator once finalised
runner = SequentialTaskRunner
# runner = ConcurrentTaskRunner
# runner = DaskTaskRunner(cluster_kwargs=dict(n_workers=1))
flow1 = flow(flows.flow1, task_runner=runner)


def main():
    # PARAMETERS
    data = "/mnt/d/data1"
    pdf = "Boskalis_Sustainability_Report_2020.pdf"
    first_page = 5
    last_page = 9

    # run flow
    warnings.filterwarnings("ignore")
    os.chdir(data)
    res = flow1(pdf, first_page, last_page)
    res.result().to_excel("output.xlsx")


if __name__ == "__main__":
    main()
