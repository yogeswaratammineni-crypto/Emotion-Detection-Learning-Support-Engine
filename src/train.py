# src/train.py
import numpy as np
import pandas as pd
import pickle
import os
import sys

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.preprocessing import (clean_text, preprocess,
                                 split_data, save_artifacts,
                                 load_artifacts)
from src.model import build_bilstm, focal_loss
import tensorflow as tf
from tensorflow.keras.callbacks import (EarlyStopping,
                                         ModelCheckpoint,
                                         ReduceLROnPlateau)


# ── Train BiLSTM ────────────────────────────────────────────
def train_bilstm(data_path: str = 'data/emotion_text_dataset.xlsx',
                  save_path: str = 'models/bltsm/',
                  epochs: int = 60,
                  batch_size: int = 8):
    """
    Full BiLSTM training pipeline.
    Args:
        data_path  : path to dataset
        save_path  : where to save model
        epochs     : max training epochs
        batch_size : training batch size
    """
    print("=" * 50)
    print("BiLSTM Training Pipeline")
    print("=" * 50)

    # Load data
    print("\n1. Loading dataset...")
    if data_path.endswith('.xlsx'):
        df = pd.read_excel(data_path)
    else:
        df = pd.read_csv(data_path)
    print(f"   Loaded {len(df)} samples")
    print(f"   Distribution:\n{df['emotion'].value_counts()}")

    # Preprocess
    print("\n2. Preprocessing...")
    X, y, tokenizer, le = preprocess(df, fit=True)
    print(f"   X shape: {X.shape}")
    print(f"   Classes: {le.classes_}")

    # Split
    print("\n3. Splitting data...")
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y)

    # Save artifacts
    print("\n4. Saving artifacts...")
    os.makedirs(save_path, exist_ok=True)
    save_artifacts(tokenizer, le, save_path)

    # Build model
    print("\n5. Building model...")
    model = build_bilstm(use_focal_loss=False)
    model.summary()

    # Callbacks
    callbacks = [
        EarlyStopping(monitor='val_accuracy',
                      patience=8,
                      restore_best_weights=True,
                      verbose=1),
        ModelCheckpoint(
            os.path.join(save_path, 'bilstm_final.keras'),
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        ),
        ReduceLROnPlateau(monitor='val_loss',
                          factor=0.5,
                          patience=3,
                          verbose=1)
    ]

    # Train
    print("\n6. Training...")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=1
    )

    best_val = max(history.history['val_accuracy'])
    print(f"\nBest val accuracy: {best_val:.4f}")

    # Evaluate on test set
    print("\n7. Evaluating on test set...")
    from sklearn.metrics import classification_report
    y_pred = np.argmax(model.predict(X_test), axis=1)
    print(classification_report(y_test, y_pred,
          target_names=le.classes_))

    print(f"\nBiLSTM saved to {save_path} ✅")
    return model, history


# ── Train BERT ──────────────────────────────────────────────
def train_bert(data_path: str = 'data/emotion_text_dataset.xlsx',
                save_path: str = 'models/bert_emotion_model_final/',
                epochs: int = 15,
                batch_size: int = 16):
    """
    Full BERT training pipeline.
    Args:
        data_path  : path to dataset
        save_path  : where to save model
        epochs     : max training epochs
        batch_size : training batch size
    """
    import torch
    from torch.utils.data import Dataset, DataLoader
    from torch.optim import AdamW
    from transformers import (BertTokenizer,
                               BertForSequenceClassification,
                               get_linear_schedule_with_warmup)
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import LabelEncoder
    from sklearn.metrics import classification_report
    from tqdm import tqdm

    print("=" * 50)
    print("BERT Training Pipeline")
    print("=" * 50)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Load data
    print("\n1. Loading dataset...")
    if data_path.endswith('.xlsx'):
        df = pd.read_excel(data_path)
    else:
        df = pd.read_csv(data_path)

    # Encode labels
    le = LabelEncoder()
    df['label'] = le.fit_transform(df['emotion'])

    # Split
    texts_train, texts_val, y_train, y_val = train_test_split(
        df['text'].values,
        df['label'].values,
        test_size=0.2,
        random_state=42,
        stratify=df['label'].values
    )
    print(f"   Train: {len(texts_train)} | Val: {len(texts_val)}")

    # Dataset class
    class EmotionDataset(Dataset):
        def __init__(self, texts, labels, tokenizer, max_len=80):
            self.texts     = texts
            self.labels    = labels
            self.tokenizer = tokenizer
            self.max_len   = max_len

        def __len__(self):
            return len(self.texts)

        def __getitem__(self, idx):
            enc = self.tokenizer(
                str(self.texts[idx]),
                max_length=self.max_len,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )
            return {
                'input_ids':      enc['input_ids'].squeeze(),
                'attention_mask': enc['attention_mask'].squeeze(),
                'labels':         torch.tensor(
                                    self.labels[idx],
                                    dtype=torch.long)
            }

    # Tokenizer and loaders
    bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    train_loader   = DataLoader(
        EmotionDataset(texts_train, y_train, bert_tokenizer),
        batch_size=batch_size, shuffle=True
    )
    val_loader = DataLoader(
        EmotionDataset(texts_val, y_val, bert_tokenizer),
        batch_size=batch_size, shuffle=False
    )

    # Model
    print("\n2. Loading BERT...")
    model = BertForSequenceClassification.from_pretrained(
        'bert-base-uncased',
        num_labels=5,
        ignore_mismatched_sizes=True
    ).to(device)

    # Optimizer and scheduler
    optimizer   = AdamW(model.parameters(), lr=2e-5, weight_decay=0.01)
    total_steps = len(train_loader) * epochs
    scheduler   = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=total_steps // 5,
        num_training_steps=total_steps
    )

    # Training loop
    print("\n3. Training BERT...")
    best_val_acc = 0

    for epoch in range(epochs):
        # Train
        model.train()
        total_loss, correct, total = 0, 0, 0
        for batch in tqdm(train_loader,
                          desc=f"Epoch {epoch+1}/{epochs}",
                          leave=False):
            input_ids      = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels         = batch['labels'].to(device)
            optimizer.zero_grad()
            outputs = model(input_ids=input_ids,
                            attention_mask=attention_mask,
                            labels=labels)
            outputs.loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            preds    = torch.argmax(outputs.logits, dim=1)
            correct  += (preds == labels).sum().item()
            total    += labels.size(0)
            total_loss += outputs.loss.item()

        train_acc = correct / total

        # Validate
        model.eval()
        val_correct, val_total = 0, 0
        all_preds, all_labels  = [], []
        with torch.no_grad():
            for batch in val_loader:
                input_ids      = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels         = batch['labels'].to(device)
                outputs = model(input_ids=input_ids,
                                attention_mask=attention_mask,
                                labels=labels)
                preds       = torch.argmax(outputs.logits, dim=1)
                val_correct += (preds == labels).sum().item()
                val_total   += labels.size(0)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        val_acc = val_correct / val_total
        print(f"Epoch {epoch+1}: "
              f"train_acc={train_acc:.4f} "
              f"val_acc={val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            os.makedirs(save_path, exist_ok=True)
            torch.save(model.state_dict(),
                       os.path.join(save_path, 'bert_model.pt'))
            print(f"  ✅ Best model saved!")

    print(f"\nBest Val Accuracy: {best_val_acc:.4f}")
    print(f"BERT saved to {save_path} ✅")

    # Final report
    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds,
          target_names=le.classes_))

    return model


# ── Main ────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Select training mode:")
    print("1. BiLSTM only")
    print("2. BERT only")
    print("3. Both")
    choice = input("Enter choice (1/2/3): ").strip()

    if choice == "1":
        train_bilstm()
    elif choice == "2":
        train_bert()
    elif choice == "3":
        train_bilstm()
        train_bert()
    else:
        print("Invalid choice")