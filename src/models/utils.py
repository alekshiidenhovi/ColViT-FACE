import torch
from src.training.scoring import maxsim
from src.metrics import recall_at_k


def compute_metrics(batch: torch.Tensor, encoder: torch.nn.Module):
    query_image, positive_image, negative_images, _, _, _ = batch

    query_repr = encoder(query_image)  # [batch, 1, seq_len, token_embedding_dim]
    all_gallery = torch.cat(
        [positive_image, negative_images], dim=1
    )  # [batch, 1+num_neg, H, W]
    gallery_repr = encoder(
        all_gallery
    )  # [batch, 1+num_neg, seq_len, token_embedding_dim]

    scores = maxsim(query_repr, gallery_repr)  # [batch, 1+num_neg]

    targets = torch.zeros_like(scores)
    targets[:, 0] = 1
    loss = torch.nn.functional.cross_entropy(scores, targets)

    recall_at_1 = recall_at_k(scores, 1)
    recall_at_3 = recall_at_k(scores, 3)

    return loss, recall_at_1, recall_at_3
