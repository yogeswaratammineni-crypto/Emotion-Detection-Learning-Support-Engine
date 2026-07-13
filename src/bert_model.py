# src/bert_model.py
import torch
import torch.nn as nn
from transformers import BertTokenizer, BertForSequenceClassification
import os

# ── Constants ──────────────────────────────────────────────
MAX_SEQ_LEN = 80
NUM_CLASSES = 5
DEVICE      = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# ── Load BERT Model ─────────────────────────────────────────
def load_bert(model_path: str = 'models/bert_emotion_model_final/bert_model.pt'):
    """
    Load fine-tuned BERT model.
    Args:
        model_path: path to saved bert_model.pt
    Returns:
        model, tokenizer, device
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"BERT model not found at {model_path}")

    # Load tokenizer
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    # Load model architecture
    model = BertForSequenceClassification.from_pretrained(
        'bert-base-uncased',
        num_labels=NUM_CLASSES,
        ignore_mismatched_sizes=True
    )

    # Load trained weights
    model.load_state_dict(
        torch.load(model_path, map_location=DEVICE)
    )
    model.to(DEVICE)
    model.eval()

    print(f"BERT loaded from {model_path} ✅")
    print(f"Device: {DEVICE}")
    return model, tokenizer, DEVICE


# ── Predict With BERT ───────────────────────────────────────
def bert_predict(text: str,
                  model,
                  tokenizer,
                  device,
                  max_len: int = MAX_SEQ_LEN):
    """
    Get emotion probabilities from BERT.
    Args:
        text      : input text string
        model     : loaded BERT model
        tokenizer : BERT tokenizer
        device    : cuda or cpu
        max_len   : max sequence length
    Returns:
        numpy array of probabilities for each emotion
    """
    import numpy as np

    enc = tokenizer(
        text,
        max_length=max_len,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    )

    input_ids      = enc['input_ids'].to(device)
    attention_mask = enc['attention_mask'].to(device)

    with torch.no_grad():
        logits = model(
            input_ids=input_ids,
            attention_mask=attention_mask
        ).logits

    probs = torch.softmax(logits, dim=1)
    return probs.cpu().numpy()[0]


# ── Quick Test ──────────────────────────────────────────────
if __name__ == "__main__":
    import numpy as np

    print("Loading BERT model...")
    model, tokenizer, device = load_bert(
        'models/bert_emotion_model_final/bert_model.pt'
    )

    print("\nTesting bert_predict()...")
    tests = [
        "I have no idea what recursion means",
        "I finally understand neural networks",
        "I am so frustrated with this bug",
        "I wonder how transformers work",
        "This lecture is so boring and repetitive"
    ]

    emotions = ['Bored', 'Confident', 'Confused', 'Curious', 'Frustrated']

    for text in tests:
        probs   = bert_predict(text, model, tokenizer, device)
        top_idx = int(np.argmax(probs))
        print(f"\nText:      {text[:50]}")
        print(f"Predicted: {emotions[top_idx]} ({probs[top_idx]:.2f})")