from torch._tensor import Tensor
from pathlib import Path
import pytest
import torch
import sys
sys.path.append(str(Path(__file__).parent.parent))

from src.training.scoring import maxsim


@pytest.mark.parametrize("batch_size, seq_len, dim", [(2, 4, 8), (1, 3, 16)])
def test_colbert_batch_size(batch_size, seq_len, dim) -> None:
    """
    Test colbert_score batch size matching with random inputs of various sizes.
    """
    query: Tensor = torch.randn(batch_size, seq_len, dim)
    gallery: Tensor = torch.randn(batch_size, seq_len, dim)
    scores: Tensor = maxsim(query, gallery)
    assert scores.shape == (batch_size, ), f"Expected output shape {(batch_size,)} but got {scores.shape}"


def test_colbert_score_known_equal_dim() -> None:
    """
    Test colbert_score with a small, known example, with the same dimensions between tensors.
    """
    query = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]])
    gallery = torch.tensor([[[0.5, 2.0], [2.0, 0.5]]])
    expected_score = torch.tensor([4.0])
    scores = maxsim(query, gallery)
    assert torch.allclose(scores, expected_score), f"Expected {expected_score}, but got {scores}"


def test_colbert_score_known_different_dims() -> None:
    """
    Test colbert_score with a small, known example, with different dimensions between tensors.
    """
    query = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]])
    gallery = torch.tensor([[[0.5, 2.0], [2.0, 0.5], [3.0, 0.0]]])
    expected_score = torch.tensor([5.0])
    scores = maxsim(query, gallery)
    assert torch.allclose(scores, expected_score), f"Expected {expected_score}, but got {scores}"


def test_colbert_score_mismatch_batch_size() -> None:
    """
    Confirm that colbert_score raises an error when the batch sizes do not match.
    """
    query = torch.randn(2, 4, 8)
    gallery = torch.randn(3, 4, 8)
    with pytest.raises(RuntimeError):
        maxsim(query, gallery)
