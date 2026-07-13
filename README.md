# AI-Driven Emotion Detection & Personalized Learning Support Platform

## Project Overview
An end-to-end AI platform that detects student emotions from text
and provides personalized learning support using BiLSTM, BERT,
and Gemini AI.

## Team Members
| Member | Name | Role | Responsibilities |
|--------|------|------|-----------------|
| Member 1 | Yogeswararao | ML Engineer | Dataset creation, BiLSTM & BERT model training, emotion detection pipeline, preprocessing, predict.py |
| Member 2 | Lalith Prava | AI & Backend Engineer | Gemini AI integration, CSV interaction logging, emotion response mapping, fallback responses |
| Member 3 | Chaitanya Prasanna Kumar | Frontend & Analytics Engineer | Streamlit UI, Plotly analytics dashboard, session history, model comparison UI |

## Detected Emotions
| Emotion | Description |
|---------|-------------|
| 😕 Confused | Student doesn't understand the concept |
| 😤 Frustrated | Student is stuck and overwhelmed |
| 🤔 Curious | Student wants to explore deeper |
| 😊 Confident | Student has mastered the topic |
| 😴 Bored | Student finds content unchallenging |

## 🎬 Project Demo Video

[![Watch Demo](https://img.shields.io/badge/Watch%20Demo-Google%20Drive-green)](https://drive.google.com/file/d/19SdGmrIqo09pDewHkQREtD5oumFLQ2m3/view?usp=drive_link)

**[Click here to watch the demo video](https://drive.google.com/file/d/19SdGmrIqo09pDewHkQREtD5oumFLQ2m3/view?usp=drive_link)**

## Project Structure
New folder/

├── data/

│   ├── GoEmotions/               ← Auto-downloaded via HuggingFace

│   ├── cleansed_emocontext/

│   ├── empatheticdialogues/

│   ├── ISEAR Dataset/

│   └── emotion_text_dataset.xlsx ← Custom academic dataset (250 sentences)

├── models/

│   ├── bert_emotion_model_final/

│   │   └── bert_model.pt         ← Fine-tuned BERT (70% accuracy)

│   └── bltsm/

│       ├── bilstm_final.keras    ← Trained BiLSTM (60% accuracy)

│       ├── tokenizer.pkl         ← Keras tokenizer

│       └── label_encoder.pkl     ← Sklearn label encoder

├── notebooks/                    ← Kaggle training notebooks

├── src/

│   ├── preprocessing.py          ← Text cleaning & tokenization

│   ├── model.py                  ← BiLSTM architecture & focal loss

│   ├── bert_model.py             ← BERT model loading & inference

│   ├── train.py                  ← Training pipeline for both models

│   └── predict.py                ← Core prediction & comparison functions

├── app.py                        ← Streamlit web application (Member 3)

├── analytics.py                  ← Plotly charts & dashboard (Member 3)

├── gemini_handler.py             ← Gemini AI integration (Member 2)

├── logger.py                     ← CSV interaction logging (Member 2)

├── emotion_response_mapping.csv  ← Emotion to response strategy mapping

├── emotion_response_examples.csv ← Example responses per emotion

├── requirements.txt              ← All dependencies

├── .env                          ← API keys (not committed to GitHub)

└── README.md                     ← This file

## Setup Instructions

### 1. Clone The Repository
```bash
git clone https://github.com/yogeswaratammineni-crypto/Emotion-Detection-Learning-Support-Engine
cd Emotion-Detection-Learning-Support-Engine
```
### 2. Download Model Files
Models are too large for GitHub. Download from Google Drive:

🔗 **https://drive.google.com/drive/folders/1W_Co04AUu9EIgEO7dpxMmeS3AsokCevY?usp=sharing**

After downloading, place files in these exact locations:
 
models/

├── bert_emotion_model_final/

│   └── bert_model.pt

└── bltsm/

├── bilstm_final.keras

├── baseline_bilstm.keras

├── tokenizer.pkl

└── label_encoder.pkl

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Configure Environment
Create a `.env` file in the root folder:

GEMINI_API_KEY=your_gemini_api_key_here

BILSTM_MODEL_PATH=models/bltsm/bilstm_final.keras

BERT_MODEL_PATH=models/bert_emotion_model_final/bert_model.pt

TOKENIZER_PATH=models/bltsm/tokenizer.pkl

LABEL_ENCODER_PATH=models/bltsm/label_encoder.pkl

### 5. Run The App
```bash
python -m streamlit run app.py
```

App opens at: http://localhost:8501

## Model Performance
| Model | Accuracy | Parameters | Speed | Best For |
|-------|----------|-----------|-------|----------|
| BiLSTM | 60% | ~5M | Fast | Lightweight deployment |
| BERT | 70% | 109M | Slower | Higher accuracy needed |

## Core Functions

### Member 1 — Emotion Detection (src/predict.py)
```python
from src.predict import predict_emotion, compare_models

# Single model prediction
result = predict_emotion("I don't understand recursion", model="bert")
# Returns:
# {
#   "emotion"       : "Confused",
#   "confidence"    : 0.87,
#   "all_scores"    : {"Bored": 0.03, "Confident": 0.05,
#                      "Confused": 0.87, "Curious": 0.03,
#                      "Frustrated": 0.02},
#   "mixed_emotions": ["Confused", "Curious"]
# }

# Compare both models
comparison = compare_models("I don't understand recursion")
# Returns BiLSTM and BERT results side by side
```

### Member 2 — Gemini AI Response (gemini_handler.py)
```python
from gemini_handler import get_gemini_response

response = get_gemini_response(
    emotion="Confused",
    text="I don't understand recursion"
)
# Returns personalized guidance string from Gemini AI
```

### Member 2 — CSV Logging (logger.py)
```python
from logger import log_interaction, get_all_logs, get_stats

# Log a student interaction
log_interaction(
    student_text="I don't understand recursion",
    emotion="Confused",
    confidence=0.87,
    gemini_response="Try breaking it into smaller steps...",
    bilstm_emotion="Confused",
    bilstm_confidence=1.0,
    bert_emotion="Bored",
    bert_confidence=0.34
)

# Get all past interactions
df = get_all_logs()

# Get summary statistics
stats = get_stats()
# Returns total sessions, emotion counts, avg confidence
```

### Member 3 — Streamlit App (app.py)
```python
# Run the full web application
# python -m streamlit run app.py
#
# Pages:
# 🏠 Home          → Input text, get emotion + Gemini response
# 📊 Analytics     → Plotly charts and emotion statistics
# 📋 Session History → Past interactions with delete option
```

## App Features

🏠 Home Page

├── Text input for student learning problem

├── Emotion detection (BiLSTM + BERT)

├── Confidence score with progress bars

├── Mixed emotion detection

├── BiLSTM vs BERT model comparison

└── Gemini AI personalized guidance
📊 Analytics Dashboard

├── Emotion frequency bar chart

├── Emotion distribution pie chart

├── Confidence score timeline

└── BiLSTM vs BERT comparison chart
📋 Session History

├── All past interactions table

├── Filter by emotion

├── Sort by date/confidence

└── Delete individual sessions

## Tech Stack
| Technology | Version | Purpose |
|-----------|---------|---------|
| Python | 3.12 | Core language |
| TensorFlow/Keras | 2.21.0 | BiLSTM model |
| PyTorch | 2.12.1 | BERT model |
| HuggingFace Transformers | 5.0.0 | BERT architecture |
| Streamlit | 1.58.0 | Web UI |
| Plotly | 6.8.0 | Analytics charts |
| Google Generative AI | 0.8.6 | Gemini responses |
| NLTK | 3.9.4 | Text preprocessing |
| Scikit-learn | 1.9.0 | Label encoding |
| Pandas | 2.2.2 | Data handling |

## Dataset
- **Source**: GoEmotions (Google) — 43,410 Reddit comments
- **Custom**: 250 academic emotion sentences (generated specifically for learning context)
- **Classes**: 5 emotions — Bored, Confident, Confused, Curious, Frustrated
- **Training**: 200 sentences | **Validation**: 50 sentences

## How It Works

Student types learning problem

↓

clean_text()              ← preprocessing.py

↓

predict_emotion()         ← predict.py

↓

┌──────┴──────┐

BiLSTM        BERT          ← model.py / bert_model.py

└──────┬──────┘

↓

{emotion, confidence,       ← structured result

all_scores, mixed}

↓

get_gemini_response()       ← gemini_handler.py

↓

log_interaction()           ← logger.py

↓

Streamlit UI Display        ← app.py


## License
This project was built as part of an academic NLP/ML course project.
