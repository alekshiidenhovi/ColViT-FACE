import torch


def recall_at_k(scores: torch.Tensor, k: int):
    """Compute recall@k metric for similarity scores.

    Parameters
    ----------
    scores : torch.Tensor
        Similarity scores between queries and gallery items. 0th index is the positive match.
        Shape: [batch_size, num_gallery_items]
    k : int
        Number of top gallery items to consider for each query.

    Returns
    -------
    float
        Mean recall@k score across the batch. A score of 1.0 means the positive
        match was in the top k results for every query.
    """
    _, top_indices = torch.topk(scores, k=k, dim=1)
    return torch.any(top_indices == 0, dim=1).float().mean()
