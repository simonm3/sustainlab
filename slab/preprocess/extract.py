import logging
import os
import shutil

import layoutparser as lp
import numpy as np
import pandas as pd
import pikepdf
import PyPDF2
from pdf2image import convert_from_path, pdfinfo_from_path
from sklearn.cluster import DBSCAN
from tqdm.auto import tqdm

from ..prefectx import task

log = logging.getLogger(__name__)

model = lp.Detectron2LayoutModel(
    "lp://PubLayNet/mask_rcnn_X_101_32x8d_FPN_3x/config",
    extra_config=["MODEL.ROI_HEADS.SCORE_THRESH_TEST", 0.8],
    label_map={0: "Text", 1: "Title", 2: "List", 3: "Table", 4: "Figure"},
)


@task
def decrypt(path):
    """decrypt pdf. pypdf2 cannot read encrypted even if no password"""
    outpath = f"reports_decrypted/{os.path.basename(path)}"
    os.makedirs(os.path.dirname(outpath), exist_ok=True)
    if PyPDF2.PdfFileReader(path).is_encrypted:
        pikepdf.Pdf.open(path).save(outpath)
    else:
        shutil.copy(path, outpath)

    return outpath


@task
def pdf2images(pdf, first_page=None, last_page=None):
    """
    pdf: path to the pdf that you want to extract.
    """
    # extract pages
    if first_page is not None:
        images = convert_from_path(
            pdf, fmt="jpeg", first_page=first_page, last_page=last_page
        )
    else:
        pages = pdfinfo_from_path(pdf)
        images = convert_from_path(pdf, fmt="jpeg", thread_count=pages)

    return images


@task
def image2text(image):
    image = np.array(image)
    layout = model.detect(image)

    # filter
    text_blocks = lp.Layout([b for b in layout if b.type == "Text"])
    figure_blocks = lp.Layout([b for b in layout if b.type == "Figure"])
    text_blocks = lp.Layout(
        [b for b in text_blocks if not any(b.is_in(b_fig) for b_fig in figure_blocks)]
    )
    if len(text_blocks) == 0:
        return []

    # OCR
    ocr_agent = lp.TesseractAgent(languages="eng")
    for block in text_blocks:
        segment_image = block.pad(left=5, right=5, top=5, bottom=5).crop_image(image)
        text = ocr_agent.detect(segment_image)
        block.set(text=text, inplace=True)

    # cluster columns and sort by column then row
    df = pd.DataFrame()
    df["text"] = [b.text for b in text_blocks]
    df[["x0", "y0", "x1", "y1"]] = [b.coordinates for b in text_blocks]
    db = DBSCAN(eps=15, min_samples=1)
    df["cluster"] = db.fit_predict(np.array(df.x0.values).reshape(-1, 1))
    df["x0group"] = df.groupby("cluster").x0.transform(np.median).astype(int)
    df = df.sort_values(["x0group", "y0"])

    # remove special chars at end of block
    df.text = df.text.str.replace(r"\s*$", "", regex=True)

    return df.text.tolist()


@task
def merge_pages(all_text):
    # list of lists to list
    all_text = sum(all_text, [])
    # merge blocks with no sentence end (.!?) with next block. can run to next page.
    merged = []
    buffer = []
    for text in all_text:
        buffer.append(text)
        # TODO bullet points without fullstops?
        if text.endswith((".", "!", "?")):
            # remove extra blank lines in block as OCR error?
            merged.append("\n".join(buffer).replace("\n\n", "\n"))
            buffer = []
    # if last sentence does not end properly
    if buffer:
        merged.append("".join(buffer).replace("\n\n", "\n"))

    # create output
    text = "\n\n".join(merged)

    return text
