"""
Simple Data Preprocessing for Webpage Classification
Author: Lucas Childs
"""

import pandas as pd
import pyreadr
from bs4 import BeautifulSoup
import re
from sklearn.model_selection import train_test_split

# Load RData and Extract Text

def load_data(data_path):
    """Load RData file and return dataframe"""
    result = pyreadr.read_r(data_path)
    # Acess the dataframe from the RData file
    df = result[list(result.keys())[0]]
    return df

def clean_html(html_text):
    """Extract and clean text from HTML"""
    if pd.isna(html_text):
        return ""

    try:
        # Parse HTML
        soup = BeautifulSoup(html_text, 'html.parser')

        # Get text from paragraphs and headers
        text = soup.get_text(separator=' ')

        # Clean up
        text = re.sub(r'http\S+', '', text)  # Remove URLs
        text = re.sub(r'\S+@\S+', '', text)  # Remove emails
        text = re.sub(r'[^\w\s]', ' ', text)  # Remove punctuation
        text = re.sub(r'\d+', '', text)  # Remove numbers
        text = re.sub(r'\s+', ' ', text).strip().lower()  # Clean whitespace and lowercase

        return text
    except:
        return ""

def preprocess_data(data_path):
    """Load and preprocess the webpage data"""
    print("Loading data...")
    df = load_data(data_path)

    print(f"Loaded {len(df)} samples")
    print(f"Columns: {list(df.columns)}")

    # Clean HTML and extract text
    print("\nCleaning HTML and extracting text...")
    df['text_clean'] = df['text_tmp'].apply(clean_html)

    # Remove empty texts
    df = df[df['text_clean'].str.len() > 0].copy()

    # Convert labels to binary (0/1)
    unique_labels = df['bclass'].unique()
    print(f"\nOriginal labels: {unique_labels}")
    df['label'] = (df['bclass'] == unique_labels[1]).astype(int)

    print(f"\nAfter cleaning: {len(df)} samples")
    print(f"Class distribution: {df['label'].value_counts().to_dict()}")

    return df

# Split Data

def split_data(df, test_size=0.2, val_size=0.1, random_state=42):
    """Split data into train, validation, and test sets"""

    # First split: separate test set
    train_val, test = train_test_split(
        df, test_size=test_size, random_state=random_state, stratify=df['label']
    )

    # Second split: separate validation from training
    val_ratio = val_size / (1 - test_size)
    train, val = train_test_split(
        train_val, test_size=val_ratio, random_state=random_state, stratify=train_val['label']
    )

    print(f"\nData split:")
    print(f"  Train: {len(train)} samples")
    print(f"  Validation: {len(val)} samples")
    print(f"  Test: {len(test)} samples")

    return train, val, test

# Main execution

def load_and_split_data(data_path="data/claims-raw.RData", test_size=0.2, val_size=0.1, random_state=42):
    """
    Load, preprocess, and split data into train/val/test sets.

    Returns:
        tuple: (train_df, val_df, test_df) - DataFrames with 'text_clean' and 'label' columns
    """
    df = preprocess_data(data_path)
    train_df, val_df, test_df = split_data(df, test_size=test_size, val_size=val_size, random_state=random_state)
    return train_df, val_df, test_df

if __name__ == "__main__":
    # Load and preprocess data
    data_path = "data/claims-raw.RData"
    train_df, val_df, test_df = load_and_split_data(data_path)

    print("\nPreprocessing complete!")
    print(f"\nSample text (first 200 chars):")
    print(train_df['text_clean'].iloc[0][:200] + "...")
