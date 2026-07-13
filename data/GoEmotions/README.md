GoEmotions Dataset
==================
This dataset is automatically downloaded via HuggingFace datasets library.

To download manually, run:
    from datasets import load_dataset
    dataset = load_dataset("go_emotions", "simplified")

Source: https://huggingface.co/datasets/go_emotions
Paper: Google Research - GoEmotions (2020)
Classes used: 28 emotions remapped to 5 academic emotions

Remapping:
    confusion, surprise, realization  → Confused
    curiosity, excitement             → Curious
    joy, pride, optimism, gratitude   → Confident
    anger, annoyance, fear, remorse   → Frustrated
    neutral, sadness, boredom         → Bored

    