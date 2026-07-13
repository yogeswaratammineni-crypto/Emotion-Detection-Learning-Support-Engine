# analytics.py
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from logger import get_all_logs, get_stats

# ── Color Theme ─────────────────────────────────────────────
EMOTION_COLORS = {
    'Confused'   : '#FF6B6B',
    'Frustrated' : '#FF8E53',
    'Curious'    : '#4ECDC4',
    'Confident'  : '#45B7D1',
    'Bored'      : '#96CEB4'
}


# ── Emotion Frequency Bar Chart ─────────────────────────────
def plot_emotion_frequency():
    """
    Bar chart showing how often each emotion appears.
    Returns plotly figure.
    """
    df = get_all_logs()

    if df.empty:
        fig = go.Figure()
        fig.add_annotation(
            text="No data yet — start using the app!",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=16, color="gray")
        )
        fig.update_layout(
            title="Emotion Frequency",
            height=400
        )
        return fig

    counts = df['emotion'].value_counts().reset_index()
    counts.columns = ['emotion', 'count']

    fig = px.bar(
        counts,
        x='emotion',
        y='count',
        color='emotion',
        color_discrete_map=EMOTION_COLORS,
        title='Emotion Frequency Distribution',
        labels={'emotion': 'Emotion', 'count': 'Count'},
        text='count'
    )
    fig.update_traces(textposition='outside')
    fig.update_layout(
        showlegend=False,
        height=400,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(size=13)
    )
    return fig


# ── Confidence Score Timeline ────────────────────────────────
def plot_confidence_timeline():
    """
    Line chart showing confidence scores over time.
    Returns plotly figure.
    """
    df = get_all_logs()

    if df.empty:
        fig = go.Figure()
        fig.add_annotation(
            text="No data yet — start using the app!",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=16, color="gray")
        )
        fig.update_layout(
            title="Confidence Score Timeline",
            height=400
        )
        return fig

    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['session']   = range(1, len(df) + 1)

    fig = go.Figure()

    # Main confidence line
    fig.add_trace(go.Scatter(
        x=df['session'],
        y=df['confidence'],
        mode='lines+markers',
        name='Confidence',
        line=dict(color='#45B7D1', width=2),
        marker=dict(
            size=8,
            color=[EMOTION_COLORS.get(e, '#888')
                   for e in df['emotion']],
            line=dict(width=1, color='white')
        ),
        hovertemplate=(
            'Session %{x}<br>'
            'Confidence: %{y:.2%}<br>'
            '<extra></extra>'
        )
    ))

    # Average line
    avg = df['confidence'].mean()
    fig.add_hline(
        y=avg,
        line_dash="dash",
        line_color="orange",
        annotation_text=f"Avg: {avg:.2%}"
    )

    fig.update_layout(
        title='Confidence Score Timeline',
        xaxis_title='Session Number',
        yaxis_title='Confidence Score',
        yaxis=dict(tickformat='.0%', range=[0, 1.1]),
        height=400,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(size=13)
    )
    return fig


# ── Emotion Distribution Pie Chart ──────────────────────────
def plot_emotion_pie():
    """
    Pie chart showing emotion distribution.
    Returns plotly figure.
    """
    df = get_all_logs()

    if df.empty:
        fig = go.Figure()
        fig.add_annotation(
            text="No data yet — start using the app!",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=16, color="gray")
        )
        fig.update_layout(
            title="Emotion Distribution",
            height=400
        )
        return fig

    counts = df['emotion'].value_counts()

    fig = px.pie(
        values=counts.values,
        names=counts.index,
        title='Emotion Distribution',
        color=counts.index,
        color_discrete_map=EMOTION_COLORS,
        hole=0.4
    )
    fig.update_traces(
        textposition='inside',
        textinfo='percent+label'
    )
    fig.update_layout(
        height=400,
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(size=13)
    )
    return fig


# ── BiLSTM vs BERT Comparison Chart ─────────────────────────
def plot_model_comparison():
    """
    Bar chart comparing BiLSTM vs BERT predictions.
    Returns plotly figure.
    """
    df = get_all_logs()

    if df.empty or 'bilstm_emotion' not in df.columns:
        fig = go.Figure()
        fig.add_annotation(
            text="No comparison data yet!",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=16, color="gray")
        )
        fig.update_layout(
            title="BiLSTM vs BERT Comparison",
            height=400
        )
        return fig

    # Agreement rate
    df['agree'] = df['bilstm_emotion'] == df['bert_emotion']
    agreement   = df['agree'].mean() * 100

    # Confidence comparison
    fig = go.Figure()

    fig.add_trace(go.Bar(
        name='BiLSTM Confidence',
        x=list(range(1, len(df) + 1)),
        y=df['bilstm_confidence'],
        marker_color='#FF6B6B',
        opacity=0.7
    ))

    fig.add_trace(go.Bar(
        name='BERT Confidence',
        x=list(range(1, len(df) + 1)),
        y=df['bert_confidence'],
        marker_color='#45B7D1',
        opacity=0.7
    ))

    fig.update_layout(
        title=f'BiLSTM vs BERT Confidence (Agreement: {agreement:.1f}%)',
        xaxis_title='Session',
        yaxis_title='Confidence',
        yaxis=dict(range=[0, 1.1]),
        barmode='group',
        height=400,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(size=13)
    )
    return fig


# ── Full Dashboard ───────────────────────────────────────────
def show_dashboard():
    """
    Returns all 4 charts and stats for Streamlit.
    """
    return {
        'emotion_frequency'  : plot_emotion_frequency(),
        'confidence_timeline': plot_confidence_timeline(),
        'emotion_pie'        : plot_emotion_pie(),
        'model_comparison'   : plot_model_comparison(),
        'stats'              : get_stats()
    }


# ── Quick Test ───────────────────────────────────────────────
if __name__ == "__main__":
    print("Testing analytics.py...")
    dashboard = show_dashboard()
    print(f"Charts generated: {len(dashboard) - 1}")
    stats = dashboard['stats']
    print(f"Total sessions:  {stats['total_sessions']}")
    print(f"Emotion counts:  {stats['emotion_counts']}")
    print(f"Avg confidence:  {stats['avg_confidence']}")
    print("Analytics working! ✅")