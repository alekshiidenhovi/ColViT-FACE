import pytest
import sys
import torch
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
from src.metrics import recall_at_k


def test_recall_at_k_perfect_match():
    scores = torch.tensor([[10.0, 5.0, 3.0], [8.0, 4.0, 2.0]])
    assert recall_at_k(scores, k=1) == 1.0
    assert recall_at_k(scores, k=2) == 1.0
    assert recall_at_k(scores, k=3) == 1.0


def test_recall_at_k_no_match():
    scores = torch.tensor([[1.0, 5.0, 8.0], [2.0, 6.0, 9.0]])
    assert recall_at_k(scores, k=1) == 0.0
    assert recall_at_k(scores, k=2) == 0.0
    assert recall_at_k(scores, k=3) == 1.0


def test_recall_at_k_partial_match():
    scores = torch.tensor([[5.0, 8.0, 3.0], [10.0, 7.0, 4.0]])
    assert recall_at_k(scores, k=1) == 0.5
    assert recall_at_k(scores, k=2) == 1.0
