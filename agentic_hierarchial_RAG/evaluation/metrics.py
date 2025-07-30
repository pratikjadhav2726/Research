import re
from typing import List


def normalize(text: str) -> str:
    """Lower text and remove punctuation/articles/extra whitespace."""
    text = text.lower()
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    text = re.sub(r"[^a-z0-9 ]", " ", text)
    text = " ".join(text.split())
    return text


def exact_match(prediction: str, ground_truth: str) -> int:
    return int(normalize(prediction) == normalize(ground_truth))


def f1_score(prediction: str, ground_truth: str) -> float:
    pred_tokens = normalize(prediction).split()
    gt_tokens = normalize(ground_truth).split()
    common = set(pred_tokens) & set(gt_tokens)
    if not common:
        return 0.0
    precision = len(common) / len(pred_tokens)
    recall = len(common) / len(gt_tokens)
    return 2 * precision * recall / (precision + recall)