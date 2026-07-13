# src/preprocessing.py
import re
import nltk
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
import os

# Download required NLTK data
nltk.download('stopwords', quiet=True)
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)

from nltk.corpus import stopwords

# ── Constants ──────────────────────────────────────────────
MAX_VOCAB_SIZE = 30000
MAX_SEQ_LEN    = 80
EMOTIONS       = ['Bored', 'Confident', 'Confused', 'Curious', 'Frustrated']

# Stopwords — keep emotion-relevant words
stop_words = set(stopwords.words('english'))
keep_words = {
    'not', 'no', 'never', "don't", "can't", "won't",
    "doesn't", "isn't", "aren't", "very", "really"
}
stop_words = stop_words - keep_words


# ── Text Cleaning ───────────────────────────────────────────
def clean_text(text: str) -> str:
    """
    Clean and normalize input text.
    Steps:
        1. Lowercase
        2. Remove URLs
        3. Remove special characters
        4. Tokenize
        5. Remove stopwords
    """
    text = str(text).lower()
    text = re.sub(r'https?://\S+', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    tokens = text.split()
    tokens = [t for t in tokens
              if t not in stop_words and len(t) > 1]
    return " ".join(tokens)


# ── Load Custom Dataset ─────────────────────────────────────
def load_custom_dataset(path: str = 'data/emotion_text_dataset.xlsx'):
    """Load the custom academic emotion dataset."""
    if path.endswith('.xlsx'):
        df = pd.read_excel(path)
    else:
        df = pd.read_csv(path)
    print(f"Loaded dataset: {len(df)} rows")
    print(df['emotion'].value_counts())
    return df


# ── Preprocess Pipeline ─────────────────────────────────────
def preprocess(df: pd.DataFrame,
               tokenizer=None,
               label_encoder=None,
               fit=True):
    """
    Full preprocessing pipeline.
    Args:
        df            : DataFrame with 'text' and 'emotion' columns
        tokenizer     : existing Keras tokenizer (None = create new)
        label_encoder : existing LabelEncoder (None = create new)
        fit           : True = fit new, False = transform only
    Returns:
        X_padded, y, tokenizer, label_encoder
    """
    # Clean text
    df = df.copy()
    df['clean_text'] = df['text'].apply(clean_text)

    # Label encoding
    if label_encoder is None:
        label_encoder = LabelEncoder()
    if fit:
        y = label_encoder.fit_transform(df['emotion'])
    else:
        y = label_encoder.transform(df['emotion'])

    # Tokenization
    if tokenizer is None:
        tokenizer = Tokenizer(
            num_words=MAX_VOCAB_SIZE,
            oov_token='<OOV>'
        )
    if fit:
        tokenizer.fit_on_texts(df['clean_text'])

    sequences = tokenizer.texts_to_sequences(df['clean_text'])
    X_padded  = pad_sequences(
        sequences,
        maxlen=MAX_SEQ_LEN,
        padding='post',
        truncating='post'
    )

    return X_padded, y, tokenizer, label_encoder


# ── Train/Val/Test Split ────────────────────────────────────
def split_data(X, y, val_size=0.1, test_size=0.1):
    """Split data into train, val, test sets."""
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y,
        test_size=val_size + test_size,
        random_state=42,
        stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp,
        test_size=0.5,
        random_state=42,
        stratify=y_temp
    )
    print(f"Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}")
    return X_train, X_val, X_test, y_train, y_val, y_test


# ── Save/Load Artifacts ─────────────────────────────────────
def save_artifacts(tokenizer, label_encoder,
                   path: str = 'models/bltsm/'):
    """Save tokenizer and label encoder."""
    os.makedirs(path, exist_ok=True)
    with open(os.path.join(path, 'tokenizer.pkl'), 'wb') as f:
        pickle.dump(tokenizer, f)
    with open(os.path.join(path, 'label_encoder.pkl'), 'wb') as f:
        pickle.dump(label_encoder, f)
    print(f"Artifacts saved to {path} ✅")


def load_artifacts(path: str = 'models/bltsm/'):
    """Load tokenizer and label encoder."""
    with open(os.path.join(path, 'tokenizer.pkl'), 'rb') as f:
        tokenizer = pickle.load(f)
    with open(os.path.join(path, 'label_encoder.pkl'), 'rb') as f:
        label_encoder = pickle.load(f)
    print(f"Artifacts loaded from {path} ✅")
    return tokenizer, label_encoder


# ── Quick Test ──────────────────────────────────────────────
if __name__ == "__main__":
    # Test clean_text
    tests = [
        "I have NO idea what recursion means!!",
        "I'm so frustrated I can't fix this bug https://stackoverflow.com",
        "Finally understood neural networks, feeling confident!",
    ]
    print("Testing clean_text():")
    for t in tests:
        print(f"  Input:  {t}")
        print(f"  Output: {clean_text(t)}")
        print()

    # Test load artifacts
    print("Testing load_artifacts():")
    try:
        tokenizer, le = load_artifacts('models/bltsm/')
        print(f"  Label classes: {le.classes_}")
        print(f"  Vocab size: {len(tokenizer.word_index)}")
    except Exception as e:
        print(f"  Error: {e}")