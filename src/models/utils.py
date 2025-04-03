import torch
import torch.nn.functional as F
from training.scoring import maxsim
from einops import repeat, rearrange


def compute_similarity_scores(batch: torch.Tensor, encoder: torch.nn.Module):
    pixel_values, _ = batch
    batch_size, num_images = pixel_values.shape[:2]
    all_images_flat = rearrange(
        pixel_values,
        "batch_size num_images channel height width -> (batch_size num_images) channel height width",
    )
    all_reprs = encoder(
        all_images_flat
    )  # [(batch_size * num_images), seq_len, reduced_dim]
    all_reprs = F.normalize(all_reprs, p=2, dim=-1)
    all_reprs = rearrange(
        all_reprs,
        "(batch_size num_images) seq_len reduced_dim -> batch_size num_images seq_len reduced_dim",
        batch_size=batch_size,
        num_images=num_images,
    )
    query_repr = all_reprs[:, 0]  # [batch_size, seq_len, reduced_dim]
    query_repr = repeat(
        query_repr,
        "batch_size seq_len reduced_dim -> batch_size num_images seq_len reduced_dim",
        num_images=1,
    )
    gallery_reprs = all_reprs[:, 1:]  # [batch_size, 1+num_neg, seq_len, reduced_dim]
    scores = maxsim(query_repr, gallery_reprs)  # [batch, 1+num_neg]
    return scores
