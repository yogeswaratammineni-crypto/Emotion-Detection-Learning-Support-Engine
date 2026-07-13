# app.py
import streamlit as st
import pandas as pd
import sys
import os
from dotenv import load_dotenv

load_dotenv()

# ── Page Configuration ──────────────────────────────────────
st.set_page_config(
    page_title="Emotion Detection & Learning Support Engine",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Imports ─────────────────────────────────────────────────
from src.predict import predict_emotion, compare_models
from gemini_handler import get_gemini_response
from logger import log_interaction, get_all_logs, get_stats, delete_log_entry
from analytics import (plot_emotion_frequency,
                        plot_confidence_timeline,
                        plot_emotion_pie,
                        plot_model_comparison)

# ── Custom CSS ───────────────────────────────────────────────
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1E88E5;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .emotion-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 12px;
        color: white;
        text-align: center;
        margin: 1rem 0;
    }
    .confidence-bar {
        background: #f0f2f6;
        border-radius: 8px;
        padding: 1rem;
        margin: 0.5rem 0;
    }
    .response-box {
        background: #f8f9fa;
        border-left: 4px solid #1E88E5;
        padding: 1.5rem;
        border-radius: 0 8px 8px 0;
        margin: 1rem 0;
        color: black;
    }
    .metric-card {
        background: white;
        border-radius: 8px;
        padding: 1rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        text-align: center;
        color: black;
    }
    .stButton > button {
        width: 100%;
        border-radius: 8px;
        height: 3rem;
        font-size: 1rem;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

# ── Emotion Emojis ───────────────────────────────────────────
EMOTION_EMOJI = {
    'Confused'   : '😕',
    'Frustrated' : '😤',
    'Curious'    : '🤔',
    'Confident'  : '😊',
    'Bored'      : '😴'
}

EMOTION_COLOR = {
    'Confused'   : '#FF6B6B',
    'Frustrated' : '#FF8E53',
    'Curious'    : '#4ECDC4',
    'Confident'  : '#45B7D1',
    'Bored'      : '#96CEB4'
}

# ── Sidebar Navigation ───────────────────────────────────────
with st.sidebar:
    st.markdown("## 🧠 Emotion Detection & Learning Support Engine")
    st.markdown("---")
    page = st.radio(
        "Navigate",
        ["🏠 Home", "📊 Analytics", "📋 Session History"],
        label_visibility="collapsed"
    )
    st.markdown("---")
    st.markdown("### 🤖 Model Settings")
    model_choice = st.selectbox(
        "Primary Model",
        ["bert", "bilstm"],
        help="Choose which model to use for emotion detection"
    )
    show_comparison = st.checkbox(
        "Show Model Comparison",
        value=True,
        help="Show BiLSTM vs BERT side by side"
    )
    st.markdown("---")
    stats = get_stats()
    st.markdown("### 📈 Quick Stats")
    st.metric("Total Sessions", stats['total_sessions'])
    st.metric("Avg Confidence",
              f"{stats['avg_confidence']:.0%}"
              if stats['avg_confidence'] else "0%")
    if stats['most_common'] != 'None':
        emoji = EMOTION_EMOJI.get(stats['most_common'], '❓')
        st.metric("Most Common Emotion",
                  f"{emoji} {stats['most_common']}")

# ════════════════════════════════════════════════════════════
# PAGE 1 — HOME
# ════════════════════════════════════════════════════════════
if page == "🏠 Home":

    st.markdown(
        '<div class="main-header">🧠 Emotion Detection & Learning Support Engine</div>',
        unsafe_allow_html=True
    )
    st.markdown(
        '<div class="sub-header">Describe your learning challenge and get personalized AI guidance</div>',
        unsafe_allow_html=True
    )

    # Input Section
    col1, col2, col3 = st.columns([1, 3, 1])
    with col2:
        student_text = st.text_area(
            "What are you struggling with?",
            placeholder="Example: I have no idea what recursion means no matter how many times I read it...",
            height=120,
            key="student_input"
        )

        col_btn1, col_btn2 = st.columns(2)
        with col_btn1:
            submit = st.button("🔍 Analyze & Get Help",
                               type="primary")
        with col_btn2:
            clear = st.button("🗑️ Clear", type="secondary")

        if clear:
            st.rerun()

    # Results Section
    if submit and student_text.strip():
        with st.spinner("Analyzing your emotion..."):

            # Get predictions
            if show_comparison:
                comparison = compare_models(student_text)
                bert_result   = comparison['bert']
                bilstm_result = comparison['bilstm']

                if model_choice == "bert":
                    main_result = bert_result
                else:
                    main_result = bilstm_result
            else:
                main_result   = predict_emotion(
                    student_text, model=model_choice
                )
                bert_result   = main_result
                bilstm_result = main_result

            emotion    = main_result['emotion']
            confidence = main_result['confidence']
            mixed      = main_result['mixed_emotions']

        with st.spinner("Generating personalized guidance..."):
            gemini_response = get_gemini_response(
                emotion, student_text
            )

        # Log interaction
        log_interaction(
            student_text=student_text,
            emotion=emotion,
            confidence=confidence,
            gemini_response=gemini_response,
            mixed_emotions=mixed,
            bilstm_emotion=bilstm_result['emotion'],
            bilstm_confidence=bilstm_result['confidence'],
            bert_emotion=bert_result['emotion'],
            bert_confidence=bert_result['confidence'],
            model_used=model_choice
        )

        st.markdown("---")

        # Main emotion result
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            emoji = EMOTION_EMOJI.get(emotion, '❓')
            color = EMOTION_COLOR.get(emotion, '#888')
            st.markdown(f"""
            <div style="background: linear-gradient(135deg,
                {color}88 0%, {color} 100%);
                padding: 2rem; border-radius: 16px;
                color: white; text-align: center;
                margin: 1rem 0;">
                <div style="font-size: 4rem;">{emoji}</div>
                <div style="font-size: 2rem;
                     font-weight: 700;">{emotion}</div>
                <div style="font-size: 1.2rem;
                     opacity: 0.9;">
                     Confidence: {confidence:.0%}</div>
            </div>
            """, unsafe_allow_html=True)

        # Mixed emotions
        if len(mixed) > 1:
            st.info(f"🔀 Mixed emotions detected: "
                    f"{' + '.join(mixed)}")

        # All emotion scores
        st.markdown("#### 📊 Emotion Scores")
        scores = main_result['all_scores']
        for em, score in sorted(scores.items(),
                                 key=lambda x: x[1],
                                 reverse=True):
            col1, col2 = st.columns([1, 4])
            with col1:
                st.write(f"{EMOTION_EMOJI.get(em,'')} {em}")
            with col2:
                st.progress(score,
                            text=f"{score:.0%}")

        # Model comparison
        if show_comparison:
            st.markdown("#### 🤖 Model Comparison")
            col1, col2 = st.columns(2)
            with col1:
                b_color = EMOTION_COLOR.get(
                    bilstm_result['emotion'], '#888'
                )
                st.markdown(f"""
                <div style="border: 2px solid {b_color};
                     border-radius: 8px; padding: 1rem;
                     text-align: center;">
                    <b>BiLSTM</b><br>
                    {EMOTION_EMOJI.get(bilstm_result['emotion'],'')}
                    {bilstm_result['emotion']}<br>
                    <small>Confidence:
                    {bilstm_result['confidence']:.0%}</small>
                </div>
                """, unsafe_allow_html=True)
            with col2:
                b_color = EMOTION_COLOR.get(
                    bert_result['emotion'], '#888'
                )
                st.markdown(f"""
                <div style="border: 2px solid {b_color};
                     border-radius: 8px; padding: 1rem;
                     text-align: center;">
                    <b>BERT</b><br>
                    {EMOTION_EMOJI.get(bert_result['emotion'],'')}
                    {bert_result['emotion']}<br>
                    <small>Confidence:
                    {bert_result['confidence']:.0%}</small>
                </div>
                """, unsafe_allow_html=True)

            agree = bert_result['emotion'] == bilstm_result['emotion']
            if agree:
                st.success("✅ Both models agree!")
            else:
                st.warning("⚠️ Models disagree — "
                           "showing primary model result")

        # Gemini Response
        st.markdown("#### 💡 Personalized Guidance")
        st.markdown(f"""
        <div class="response-box">
            {gemini_response}
        </div>
        """, unsafe_allow_html=True)

    elif submit and not student_text.strip():
        st.warning("Please enter your learning challenge first!")

    # Tips section when no input
    if not submit:
        st.markdown("---")
        st.markdown("#### 💡 Try these examples:")
        examples = [
            "I have no idea what recursion means no matter how many times I read it",
            "I have been stuck on this bug for 5 hours and want to give up",
            "I wonder how neural networks actually learn from data",
            "I finally understand gradient descent and it makes total sense",
            "This lecture keeps repeating what I already know"
        ]
        for example in examples:
            if st.button(f"💬 {example[:60]}...",
                         key=example):
                st.session_state.student_input = example
                st.rerun()

# ════════════════════════════════════════════════════════════
# PAGE 2 — ANALYTICS
# ════════════════════════════════════════════════════════════
elif page == "📊 Analytics":

    st.markdown("## 📊 Analytics Dashboard")
    st.markdown("Real-time insights from all student interactions")
    st.markdown("---")

    # Stats row
    stats = get_stats()
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Sessions",
                  stats['total_sessions'])
    with col2:
        st.metric("Avg Confidence",
                  f"{stats['avg_confidence']:.0%}"
                  if stats['avg_confidence'] else "0%")
    with col3:
        most = stats.get('most_common', 'None')
        emoji = EMOTION_EMOJI.get(most, '❓')
        st.metric("Most Common", f"{emoji} {most}")
    with col4:
        counts = stats.get('emotion_counts', {})
        unique = len(counts)
        st.metric("Unique Emotions", f"{unique}/5")

    st.markdown("---")

    # Charts row 1
    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(
            plot_emotion_frequency(),
            use_container_width=True
        )
    with col2:
        st.plotly_chart(
            plot_emotion_pie(),
            use_container_width=True
        )

    # Charts row 2
    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(
            plot_confidence_timeline(),
            use_container_width=True
        )
    with col2:
        st.plotly_chart(
            plot_model_comparison(),
            use_container_width=True
        )

    # Refresh button
    if st.button("🔄 Refresh Dashboard"):
        st.rerun()

# ════════════════════════════════════════════════════════════
# PAGE 3 — SESSION HISTORY
# ════════════════════════════════════════════════════════════
elif page == "📋 Session History":

    st.markdown("## 📋 Session History")
    st.markdown("All past student interactions")
    st.markdown("---")

    df = get_all_logs()

    if df.empty:
        st.info("No sessions yet — go to Home and submit your first query!")
    else:
        # Summary
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Sessions", len(df))
        with col2:
            st.metric("Avg Confidence",
                      f"{df['confidence'].mean():.0%}")
        with col3:
            st.metric("Most Recent",
                      df['timestamp'].iloc[-1]
                      if len(df) > 0 else "None")

        st.markdown("---")

        # Filter
        col1, col2 = st.columns(2)
        with col1:
            filter_emotion = st.selectbox(
                "Filter by Emotion",
                ["All"] + list(df['emotion'].unique())
            )
        with col2:
            sort_order = st.selectbox(
                "Sort By",
                ["Newest First", "Oldest First",
                 "Highest Confidence", "Lowest Confidence"]
            )

        # Apply filters
        filtered_df = df.copy()
        if filter_emotion != "All":
            filtered_df = filtered_df[
                filtered_df['emotion'] == filter_emotion
            ]

        if sort_order == "Newest First":
            filtered_df = filtered_df.iloc[::-1]
        elif sort_order == "Highest Confidence":
            filtered_df = filtered_df.sort_values(
                'confidence', ascending=False
            )
        elif sort_order == "Lowest Confidence":
            filtered_df = filtered_df.sort_values(
                'confidence', ascending=True
            )

        st.markdown(f"Showing **{len(filtered_df)}** sessions")
        st.markdown("---")

        # Display each session
        for idx, row in filtered_df.iterrows():
            emoji = EMOTION_EMOJI.get(row['emotion'], '❓')
            color = EMOTION_COLOR.get(row['emotion'], '#888')

            with st.expander(
                f"{emoji} {row['emotion']} | "
                f"{float(row['confidence']):.0%} confident | "
                f"{row['timestamp']}"
            ):
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.markdown(f"**Student Input:**")
                    st.write(row['student_text'])
                    st.markdown(f"**AI Response:**")
                    st.write(row['gemini_response'])
                with col2:
                    st.markdown(f"**Details:**")
                    st.write(f"Emotion: {emoji} {row['emotion']}")
                    st.write(f"Confidence: "
                             f"{float(row['confidence']):.0%}")
                    st.write(f"BiLSTM: {row['bilstm_emotion']}")
                    st.write(f"BERT: {row['bert_emotion']}")
                    st.write(f"Model: {row['model_used']}")

                    if st.button("🗑️ Delete",
                                 key=f"del_{idx}"):
                        delete_log_entry(idx)
                        st.success("Deleted!")
                        st.rerun()

        # Clear all button
        st.markdown("---")
        if st.button("🗑️ Clear All Sessions",
                     type="secondary"):
            from logger import clear_all_logs
            clear_all_logs()
            st.success("All sessions cleared!")
            st.rerun()
            