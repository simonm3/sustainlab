import os
import warnings

def main():
    # PARAMETERS
    data = "/mnt/d/data1"
    pdf = "Boskalis_Sustainability_Report_2020.pdf"
    page = 5

    # SETTINGS
    # os.environ["DISABLE_PREFECT"] = "True"
    # os.environ["PREFECT_API_URL"] = "http://127.0.0.1:4200/api"

    # import after settings
    from slab.preprocess.flows import flow1

    # run the flow
    warnings.filterwarnings("ignore")
    os.chdir(data)
    res = flow1(pdf, page, page)
    res.to_excel("output.xlsx")

if __name__ == "__main__":
    main()
