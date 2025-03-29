import torch
from training.scoring import maxsim


def compute_similarity_scores(batch: torch.Tensor, encoder: torch.nn.Module):
    query_image, positive_image, negative_images, _, _, _ = batch

    query_repr = encoder(query_image)  # [batch, 1, seq_len, token_embedding_dim]
    all_gallery = torch.cat(
        [positive_image, negative_images], dim=1
    )  # [batch, 1+num_neg, H, W]
    gallery_repr = encoder(
        all_gallery
    )  # [batch, 1+num_neg, seq_len, token_embedding_dim]

    scores = maxsim(query_repr, gallery_repr)  # [batch, 1+num_neg]
    return scores
