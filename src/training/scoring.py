from torch._tensor import Tensor
import torch
from einops import einsum

def maxsim(query_image: Tensor, gallery_image: Tensor) -> Tensor:
    """
    Computes a ColBERT-style maximum similarity score between query image and gallery image token representation.
    Both inputs are expected to be tensors, with dimensions [batch_size, query_len, reduced_dim] and [batch_size, gallery_len, reduced_dim].
    """
    similarities: Tensor = einsum(query_image, gallery_image, "batch_size query_len reduced_dim, batch_size gallery_len reduced_dim -> batch_size query_len gallery_len")
    max_similarities, _ = torch.max(similarities, dim=2)
    scores: Tensor = max_similarities.sum(dim=1)
    return scores