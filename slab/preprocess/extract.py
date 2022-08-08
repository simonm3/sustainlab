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

from . import task

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
    if PyPDF2.PdfFileReader(path).is_encrypted:
        pikepdf.Pdf.open(path).save(outpath)
    else:
        shutil.copy(path, outpath)

    return outpath


@task
def pdf_to_text(pdf, first_page=None, last_page=None):
    """
    pdf: path to the pdf that you want to extract.
    output_file: output of  the text file
    """
    # extract pages
    if first_page is not None:
        images = convert_from_path(
            pdf, fmt="jpeg", first_page=first_page, last_page=last_page
        )
    else:
        pages = pdfinfo_from_path(pdf)
        images = convert_from_path(pdf, fmt="jpeg", thread_count=pages)

    # process pages
    all_text = []
    for image in tqdm(images):
        # get layout
        from time import time

        start = time()

        image = np.array(image)
        layout = model.detect(image)

        # filter
        text_blocks = lp.Layout([b for b in layout if b.type == "Text"])
        figure_blocks = lp.Layout([b for b in layout if b.type == "Figure"])
        text_blocks = lp.Layout(
            [
                b
                for b in text_blocks
                if not any(b.is_in(b_fig) for b_fig in figure_blocks)
            ]
        )
        if len(text_blocks) == 0:
            continue

        # OCR
        log.info(time() - start)
        start = time()
        ocr_agent = lp.TesseractAgent(languages="eng")
        log.info(time() - start)
        start = time()
        for block in text_blocks:
            segment_image = block.pad(left=5, right=5, top=5, bottom=5).crop_image(
                image
            )
            text = ocr_agent.detect(segment_image)
            block.set(text=text, inplace=True)
        log.info(time() - start)
        start = time()

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

        all_text.extend(df.text.values)

        log.info(time() - start)
        start = time()

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
