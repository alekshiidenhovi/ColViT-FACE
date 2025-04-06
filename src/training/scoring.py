from torch._tensor import Tensor
import torch
from einops import einsum, rearrange


def maxsim(query_image: Tensor, gallery_images: Tensor) -> Tensor:
    """
    Computes a single ColBERT-style maximum similarity score between query image and multiple gallery image token representation.
    Both inputs are expected to be tensors, with dimensions [batch_size, 1, query_len, reduced_dim] and [batch_size, num_images, gallery_len, reduced_dim], respectively. Output is a tensor with dimensions [batch_size, num_images].
    """
    query_image = rearrange(query_image, "b 1 l d -> b l d")
    similarities: Tensor = einsum(
        query_image,
        gallery_images,
        "batch_size query_len reduced_dim, batch_size num_images gallery_len reduced_dim -> batch_size num_images query_len gallery_len",
    )
    logger.info(f"Similarities shape: {similarities.shape}")
    max_similarities, _ = torch.max(similarities, dim=-1)
    logger.info(f"Max similarities shape: {max_similarities.shape}")
    scores: Tensor = max_similarities.sum(dim=-1)
    return scores
