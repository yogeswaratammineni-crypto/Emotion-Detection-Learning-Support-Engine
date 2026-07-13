# src/predict.py
import numpy as np
import pickle
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.preprocessing import clean_text, load_artifacts
from src.model import load_bilstm
from src.bert_model import load_bert, bert_predict
from tensorflow.keras.preprocessing.sequence import pad_sequences

# ── Constants ──────────────────────────────────────────────
MAX_SEQ_LEN = 80
EMOTIONS    = ['Bored', 'Confident', 'Confused', 'Curious', 'Frustrated']

# ── Load All Models Once ────────────────────────────────────
print("Loading models...")

# BiLSTM artifacts
_tokenizer, _le = load_artifacts('models/bltsm/')

# BiLSTM model
_bilstm = load_bilstm('models/bltsm/bilstm_final.keras')

# BERT model
_bert, _bert_tokenizer, _device = load_bert(
    'models/bert_emotion_model_final/bert_model.pt'
)

print("All models loaded! ✅")


# ── Core Prediction Function ────────────────────────────────
def predict_emotion(text: str, model: str = "bert") -> dict:
    """
    Predict emotion from student text input.

    Args:
        text  : student's text input
        model : "bert" or "bilstm"

    Returns:
        {
            "emotion"       : "Confused",
            "confidence"    : 0.87,
            "all_scores"    : {
                "Bored"     : 0.03,
                "Confident" : 0.05,
                "Confused"  : 0.87,
                "Curious"   : 0.03,
                "Frustrated": 0.02
            },
            "mixed_emotions": ["Confused", "Curious"]
        }
    """
    if not text or not text.strip():
        return {
            "emotion"       : "Confused",
            "confidence"    : 0.0,
            "all_scores"    : {e: 0.0 for e in EMOTIONS},
            "mixed_emotions": ["Confused"]
        }

    # Get probabilities
    if model.lower() == "bert":
        probs = bert_predict(text, _bert,
                              _bert_tokenizer, _device)
    else:
        probs = _bilstm_predict(text)

    # Build result
    top_idx    = int(np.argmax(probs))
    emotion    = _le.classes_[top_idx]
    confidence = float(probs[top_idx])
    all_scores = {
        _le.classes_[i]: round(float(probs[i]), 4)
        for i in range(len(_le.classes_))
    }
    mixed = _get_mixed_emotions(probs)

    return {
        "emotion"       : emotion,
        "confidence"    : round(confidence, 4),
        "all_scores"    : all_scores,
        "mixed_emotions": mixed
    }


# ── BiLSTM Prediction ───────────────────────────────────────
def _bilstm_predict(text: str) -> np.ndarray:
    """Get emotion probabilities from BiLSTM."""
    cleaned  = clean_text(text)
    seq      = _tokenizer.texts_to_sequences([cleaned])
    padded   = pad_sequences(
        seq,
        maxlen=MAX_SEQ_LEN,
        padding='post',
        truncating='post'
    )
    probs = _bilstm.predict(padded, verbose=0)[0]
    return probs


# ── Mixed Emotion Detection ─────────────────────────────────
def _get_mixed_emotions(probs: np.ndarray,
                         threshold: float = 0.15) -> list:
    """
    Return emotions above threshold.
    Also always include top 2 if they're close to each other.
    """
    top_idx = int(np.argmax(probs))
    second_idx = int(np.argsort(probs)[-2])

    mixed = [_le.classes_[top_idx]]

    # Add second emotion if it's within 20% of top emotion
    if probs[second_idx] >= probs[top_idx] * 0.25:
        mixed.append(_le.classes_[second_idx])

    # Also add any emotion above threshold
    for i, p in enumerate(probs):
        emotion = _le.classes_[i]
        if p >= threshold and emotion not in mixed:
            mixed.append(emotion)

    return mixed if len(mixed) > 1 else [_le.classes_[top_idx]]


# ── Compare Both Models ─────────────────────────────────────
def compare_models(text: str) -> dict:
    """
    Run both BiLSTM and BERT on same text.
    Returns both results for side by side comparison.
    Used by Member 3's Streamlit UI.
    """
    bert_result   = predict_emotion(text, model="bert")
    bilstm_result = predict_emotion(text, model="bilstm")

    return {
        "text"  : text,
        "bert"  : bert_result,
        "bilstm": bilstm_result,
        "agreement": bert_result["emotion"] == bilstm_result["emotion"]
    }


# ── Quick Test ──────────────────────────────────────────────
if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("Testing predict_emotion()")
    print("=" * 60)

    tests = [
        ("I have no idea what recursion means no matter how many times I read it",
         "Confused"),
        ("I finally understand how neural networks work and it all makes sense",
         "Confident"),
        ("I have been stuck on this bug for five hours and want to give up",
         "Frustrated"),
        ("I wonder how transformers handle very long documents",
         "Curious"),
        ("This lecture is just repeating last week with nothing new",
         "Bored"),
    ]

    print("\n--- BERT Results ---")
    correct = 0
    for text, expected in tests:
        result = predict_emotion(text, model="bert")
        match  = "✅" if result["emotion"] == expected else "❌"
        if result["emotion"] == expected:
            correct += 1
        print(f"{match} Expected: {expected:12} "
              f"Got: {result['emotion']:12} "
              f"Conf: {result['confidence']:.2f}")
        print(f"   Mixed: {result['mixed_emotions']}")

    print(f"\nBERT Accuracy: {correct}/{len(tests)}")

    print("\n--- BiLSTM Results ---")
    correct = 0
    for text, expected in tests:
        result = predict_emotion(text, model="bilstm")
        match  = "✅" if result["emotion"] == expected else "❌"
        if result["emotion"] == expected:
            correct += 1
        print(f"{match} Expected: {expected:12} "
              f"Got: {result['emotion']:12} "
              f"Conf: {result['confidence']:.2f}")

    print(f"\nBiLSTM Accuracy: {correct}/{len(tests)}")

    print("\n--- Model Comparison ---")
    sample = "I don't understand this concept at all and I'm very confused"
    comparison = compare_models(sample)
    print(f"Text: {sample}")
    print(f"BERT:   {comparison['bert']['emotion']} "
          f"({comparison['bert']['confidence']:.2f})")
    print(f"BiLSTM: {comparison['bilstm']['emotion']} "
          f"({comparison['bilstm']['confidence']:.2f})")
    print(f"Models agree: {comparison['agreement']}")