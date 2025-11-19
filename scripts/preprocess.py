# scripts/preprocess.py

import pyreadr
import pandas as pd
import re
import unicodedata
from bs4 import BeautifulSoup

def clean_html(raw_html):
    """
    Remove HTML tags, scripts, styles, and normalize whitespace.
    """
    # Parse HTML
    soup = BeautifulSoup(raw_html, "html.parser")

    # Remove script and style elements
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()

    text = soup.get_text(separator=" ")

    return text


def normalize_text(s):
    if pd.isna(s):
        return ""

    # Unicode normalization
    s = unicodedata.normalize("NFKD", s)

    # Lowercase
    s = s.lower()

    # Remove URLs
    s = re.sub(r"http\S+|www\.\S+", " ", s)

    # Remove non-text characters (keep punctuation and numbers)
    s = re.sub(r"[^a-z0-9\s.,!?;:/()\[\]\-']", " ", s)

    # Collapse whitespace
    s = re.sub(r"\s+", " ", s).strip()

    return s


def preprocess_rdata(input_path, output_path):
    print(f"Loading {input_path} ...")
    result = pyreadr.read_r(input_path)
    df = result[list(result.keys())[0]]

    # Clean raw HTML into plain text
    print("Stripping HTML from text_tmp...")
    df["text_clean"] = df["text_tmp"].astype(str).apply(clean_html)

    # Normalize text
    print("Normalizing text...")
    df["text_clean"] = df["text_clean"].apply(normalize_text)

    print(f"Saving processed CSV to {output_path}")
    df.to_csv(output_path, index=False)
    print("Done.")


if __name__ == "__main__":
    preprocess_rdata(
        input_path="data/claims-raw.RData",
        output_path="data/claims_clean_processed.csv"
    )

    preprocess_rdata(
        input_path="data/claims-test.RData",
        output_path="data/claims_test_processed.csv"
    )
