import pandas as pd
from pathlib import Path
import numpy as np

from sentence_transformers import SentenceTransformer, util
from models.textual_similarity import Textual_Similarity

class Sentence_Transformer(Textual_Similarity):

    def __init__(self, model_name):
        self.model = SentenceTransformer(model_name)

    def _get_sentence_embeddings(self, sentences):
        embeddings = self.model.encode(sentences, convert_to_tensor=True)
        return embeddings

    def get_similarity_matrix(self, sentences_1, sentences_2):
        embeddings_sent_1 = self._get_sentence_embeddings(sentences_1)
        embeddings_sent_2 = self._get_sentence_embeddings(sentences_2)
        cosine_scores = util.cos_sim(embeddings_sent_1, embeddings_sent_2)
        return cosine_scores.cpu().numpy()

def load_data_to_df(file_path, sheet_name = None):
    if "xls" in file_path.suffix:
        df = _load_excel(file_path, sheet_name)
    elif "csv" in file_path.suffix:
        df = _load_csv(file_path)
    else:
        supported_formats = ["xls", "xlsx", "csv"]
        raise ValueError("File format '{}' not supported. Supported formats '{}'.".format(file_path.suffix, ", ".join(supported_formats)))
    return df

def _load_excel(file_path, sheet_name):
    return pd.read_excel(file_path, sheet_name = sheet_name)

def _load_csv(file_path):
    return pd.read_csv(file_path)

def kpi_matcher(is_labeled_data=False):

    # defining file paths
    data_folder_name = Path("/mnt/d/data1")
    input_folder_name = data_folder_name
    output_folder_name = data_folder_name / "output/"

    # Load data-sets
    kpi_df = load_data_to_df(
        file_path=input_folder_name
        / "SustainLab_KPIs_and_labeled_clean sentences_v2.xlsx",
        sheet_name="KPIs",
    )
    data_df = load_data_to_df(
        file_path=input_folder_name
        / "SustainLab_KPIs_and_labeled_clean sentences_v2.xlsx",
        sheet_name="Labled Sentences",
    )
    kpi_df = kpi_df.dropna()

    # Extract Sentences KPI
    sentences = list(data_df["Single KPI sentence"])
    kpi = list(kpi_df["KPI"])

    # Load Model
    model = Sentence_Transformer("all-MiniLM-L6-v2")

    # Get KPI Matchings Matrix
    cosine_matrix = model.get_similarity_matrix(sentences, kpi)

    # Get KPI Matchings
    sim_index = [sim.argmax() for sim in cosine_matrix]
    sim_score = [sim.max() for sim in cosine_matrix]
    df = pd.DataFrame()
    df["sentences"] = sentences
    df["KPI"] = list(kpi_df["KPI"].iloc[sim_index])
    df["Score"] = list(sim_score)

    if is_labeled_data:
        df["KPI Label"] = list(data_df["KPI Label"])
        accuracy = len(df[df["KPI Label"] == df["KPI"]]) / len(df) * 100
        print("Accuracy: ", accuracy, "%")
        columns = ["Single KPI sentence", "KPI Label", "KPI", "Score"]
        df = df[columns]

    # save in excel
    output_file_name = "kpi_matching_" + str(model.__class__.__name__) + ".xlsx"
    df.to_excel(output_folder_name / output_file_name)


if __name__ == "__main__":
    kpi_matcher(True)
