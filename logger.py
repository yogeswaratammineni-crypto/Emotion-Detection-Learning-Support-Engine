# logger.py
import os
import csv
import pandas as pd
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

# ── Constants ──────────────────────────────────────────────
LOG_FILE = 'data/interactions_log.csv'

COLUMNS  = [
    'timestamp',
    'student_text',
    'emotion',
    'confidence',
    'mixed_emotions',
    'bilstm_emotion',
    'bilstm_confidence',
    'bert_emotion',
    'bert_confidence',
    'gemini_response',
    'model_used'
]


# ── Initialize Log File ─────────────────────────────────────
def init_log():
    """Create log file with headers if it doesn't exist."""
    os.makedirs('data', exist_ok=True)
    if not os.path.exists(LOG_FILE):
        with open(LOG_FILE, 'w', newline='',
                  encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=COLUMNS)
            writer.writeheader()
        print(f"Log file created: {LOG_FILE} ✅")
    else:
        print(f"Log file exists: {LOG_FILE} ✅")


# ── Log Single Interaction ──────────────────────────────────
def log_interaction(student_text: str,
                     emotion: str,
                     confidence: float,
                     gemini_response: str,
                     mixed_emotions: list = None,
                     bilstm_emotion: str = "",
                     bilstm_confidence: float = 0.0,
                     bert_emotion: str = "",
                     bert_confidence: float = 0.0,
                     model_used: str = "bert") -> bool:
    """
    Log one student interaction to CSV.

    Args:
        student_text      : what the student typed
        emotion           : final detected emotion
        confidence        : confidence score (0-1)
        gemini_response   : Gemini's personalized response
        mixed_emotions    : list of mixed emotions detected
        bilstm_emotion    : BiLSTM's prediction
        bilstm_confidence : BiLSTM's confidence
        bert_emotion      : BERT's prediction
        bert_confidence   : BERT's confidence
        model_used        : which model was primary

    Returns:
        True if logged successfully, False otherwise
    """
    init_log()

    try:
        row = {
            'timestamp'        : datetime.now().strftime(
                                  '%Y-%m-%d %H:%M:%S'),
            'student_text'     : student_text[:500],
            'emotion'          : emotion,
            'confidence'       : round(confidence, 4),
            'mixed_emotions'   : str(mixed_emotions or []),
            'bilstm_emotion'   : bilstm_emotion,
            'bilstm_confidence': round(bilstm_confidence, 4),
            'bert_emotion'     : bert_emotion,
            'bert_confidence'  : round(bert_confidence, 4),
            'gemini_response'  : gemini_response[:1000],
            'model_used'       : model_used
        }

        with open(LOG_FILE, 'a', newline='',
                  encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=COLUMNS)
            writer.writerow(row)

        return True

    except Exception as e:
        print(f"Logging error: {e}")
        return False


# ── Read All Logs ───────────────────────────────────────────
def get_all_logs() -> pd.DataFrame:
    """
    Read all logged interactions.

    Returns:
        DataFrame of all interactions
    """
    init_log()
    try:
        df = pd.read_csv(LOG_FILE)
        return df
    except Exception as e:
        print(f"Error reading logs: {e}")
        return pd.DataFrame(columns=COLUMNS)


# ── Get Stats ───────────────────────────────────────────────
def get_stats() -> dict:
    """
    Get summary statistics from logs.

    Returns:
        dict with emotion counts, avg confidence, total sessions
    """
    df = get_all_logs()

    if df.empty:
        return {
            'total_sessions'   : 0,
            'emotion_counts'   : {},
            'avg_confidence'   : 0.0,
            'most_common'      : 'None',
            'recent_sessions'  : []
        }

    return {
        'total_sessions' : len(df),
        'emotion_counts' : df['emotion'].value_counts().to_dict(),
        'avg_confidence' : round(df['confidence'].mean(), 4),
        'most_common'    : df['emotion'].value_counts().index[0],
        'recent_sessions': df.tail(5).to_dict('records')
    }


# ── Delete Log Entry ────────────────────────────────────────
def delete_log_entry(index: int) -> bool:
    """
    Delete a specific log entry by index.

    Args:
        index : row index to delete

    Returns:
        True if deleted successfully
    """
    try:
        df = get_all_logs()
        if index < 0 or index >= len(df):
            return False
        df = df.drop(index=index).reset_index(drop=True)
        df.to_csv(LOG_FILE, index=False)
        return True
    except Exception as e:
        print(f"Delete error: {e}")
        return False


# ── Clear All Logs ──────────────────────────────────────────
def clear_all_logs() -> bool:
    """Clear all logged interactions."""
    try:
        with open(LOG_FILE, 'w', newline='',
                  encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=COLUMNS)
            writer.writeheader()
        print("All logs cleared ✅")
        return True
    except Exception as e:
        print(f"Clear error: {e}")
        return False


# ── Quick Test ──────────────────────────────────────────────
if __name__ == "__main__":
    print("Testing logger.py...")
    print("=" * 50)

    # Test logging
    print("\n1. Testing log_interaction()...")
    success = log_interaction(
        student_text="I have no idea what recursion means",
        emotion="Confused",
        confidence=0.87,
        gemini_response="Try breaking it into smaller steps...",
        mixed_emotions=["Confused", "Frustrated"],
        bilstm_emotion="Confused",
        bilstm_confidence=1.0,
        bert_emotion="Bored",
        bert_confidence=0.34,
        model_used="bert"
    )
    print(f"Logged: {success} ✅")

    # Test another entry
    log_interaction(
        student_text="I finally understand neural networks!",
        emotion="Confident",
        confidence=0.92,
        gemini_response="Amazing progress! Now challenge yourself...",
        mixed_emotions=["Confident"],
        bilstm_emotion="Confident",
        bilstm_confidence=0.74,
        bert_emotion="Curious",
        bert_confidence=0.53,
        model_used="bert"
    )

    log_interaction(
        student_text="I am so frustrated with this bug",
        emotion="Frustrated",
        confidence=0.81,
        gemini_response="Take a break and try rubber duck debugging...",
        mixed_emotions=["Frustrated", "Curious"],
        bilstm_emotion="Frustrated",
        bilstm_confidence=0.81,
        bert_emotion="Frustrated",
        bert_confidence=0.51,
        model_used="bert"
    )

    # Test read
    print("\n2. Testing get_all_logs()...")
    df = get_all_logs()
    print(f"Total logs: {len(df)}")
    print(f"Columns: {list(df.columns)}")

    # Test stats
    print("\n3. Testing get_stats()...")
    stats = get_stats()
    print(f"Total sessions:  {stats['total_sessions']}")
    print(f"Emotion counts:  {stats['emotion_counts']}")
    print(f"Avg confidence:  {stats['avg_confidence']}")
    print(f"Most common:     {stats['most_common']}")

    # Test delete
    print("\n4. Testing delete_log_entry()...")
    deleted = delete_log_entry(0)
    print(f"Deleted entry 0: {deleted}")
    print(f"Remaining logs:  {len(get_all_logs())}")

    print("\n✅ Logger working perfectly!")