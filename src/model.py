# src/model.py
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (Embedding, Bidirectional,
                                      LSTM, Dense, Dropout,
                                      SpatialDropout1D)
from tensorflow.keras.optimizers import Adam
import os

# ── Constants ──────────────────────────────────────────────
MAX_VOCAB_SIZE = 30000
MAX_SEQ_LEN    = 80
EMBED_DIM      = 128
LSTM_UNITS     = 128
NUM_CLASSES    = 5


# ── Focal Loss ──────────────────────────────────────────────
def focal_loss(gamma=2.0, alpha=0.25):
    """
    Focal loss — handles class imbalance better
    than standard cross entropy.
    """
    def loss_fn(y_true, y_pred):
        y_true       = tf.cast(y_true, tf.int32)
        y_true_oh    = tf.one_hot(y_true, depth=NUM_CLASSES)
        ce           = -y_true_oh * tf.math.log(
                        tf.clip_by_value(y_pred, 1e-7, 1.0))
        weight       = alpha * y_true_oh * tf.pow(
                        1 - tf.clip_by_value(y_pred, 1e-7, 1.0),
                        gamma)
        return tf.reduce_mean(tf.reduce_sum(weight * ce, axis=1))
    return loss_fn


# ── Build BiLSTM Model ──────────────────────────────────────
def build_bilstm(vocab_size=MAX_VOCAB_SIZE,
                  embed_dim=EMBED_DIM,
                  lstm_units=LSTM_UNITS,
                  num_classes=NUM_CLASSES,
                  dropout_rate=0.3,
                  use_focal_loss=True):
    """
    Build BiLSTM model for emotion classification.
    Args:
        vocab_size    : vocabulary size
        embed_dim     : embedding dimensions
        lstm_units    : LSTM hidden units
        num_classes   : number of emotion classes
        dropout_rate  : dropout rate
        use_focal_loss: use focal loss or sparse categorical
    Returns:
        compiled Keras model
    """
    model = Sequential([
        Embedding(input_dim=vocab_size,
                  output_dim=embed_dim,
                  mask_zero=False),
        SpatialDropout1D(0.2),
        Bidirectional(LSTM(lstm_units,
                           dropout=dropout_rate,
                           return_sequences=True)),
        Bidirectional(LSTM(lstm_units // 2,
                           dropout=dropout_rate)),
        Dense(128, activation='relu'),
        Dropout(dropout_rate),
        Dense(64,  activation='relu'),
        Dropout(dropout_rate),
        Dense(num_classes, activation='softmax')
    ])

    loss = (focal_loss(gamma=2.0, alpha=0.25)
            if use_focal_loss
            else 'sparse_categorical_crossentropy')

    model.compile(
        optimizer=Adam(learning_rate=1e-3, clipnorm=1.0),
        loss=loss,
        metrics=['accuracy']
    )

    return model


# ── Load Saved BiLSTM ───────────────────────────────────────
def load_bilstm(path: str = 'models/bltsm/bilstm_final.keras'):
    """Load saved BiLSTM model."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model not found at {path}")
    model = tf.keras.models.load_model(
        path,
        custom_objects={'loss_fn': focal_loss()}
    )
    print(f"BiLSTM loaded from {path} ✅")
    return model


# ── Quick Test ──────────────────────────────────────────────
if __name__ == "__main__":
    print("Building BiLSTM model...")
    model = build_bilstm()
    model.summary()

    print("\nTesting load_bilstm()...")
    try:
        loaded = load_bilstm('models/bltsm/bilstm_final.keras')
        print(f"Model loaded successfully ✅")
        print(f"Input shape:  {loaded.input_shape}")
        print(f"Output shape: {loaded.output_shape}")
    except Exception as e:
        print(f"Load error: {e}")