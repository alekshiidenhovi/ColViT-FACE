import torch
import torch.nn.functional as F
import typing as T
from training.scoring import maxsim
from einops import repeat


def compute_similarity_scores(batch: torch.Tensor, encoder: torch.nn.Module):
    query_image, positive_image, negative_images, _, _, _ = batch

    query_repr: torch.Tensor = encoder(query_image)  # [batch, seq_len, reduced_dim]
    positive_repr: torch.Tensor = encoder(
        positive_image
    )  # [batch, seq_len, reduced_dim]
    negative_reprs: T.List[torch.Tensor] = [
        encoder(neg_image) for neg_image in negative_images
    ]  # num_negative_images * [batch, seq_len, reduced_dim]

    query_repr = F.normalize(query_repr, p=2, dim=-1)
    positive_repr = F.normalize(positive_repr, p=2, dim=-1)
    negative_reprs = [F.normalize(neg_repr, p=2, dim=-1) for neg_repr in negative_reprs]

    query_repr = repeat(
        query_repr, "batch_size seq_len reduced_dim -> batch_size 1 seq_len reduced_dim"
    )
    positive_repr = repeat(
        positive_repr,
        "batch_size seq_len reduced_dim -> batch_size 1 seq_len reduced_dim",
    )
    negative_reprs = [
        repeat(
            neg_repr,
            "batch_size seq_len reduced_dim -> batch_size 1 seq_len reduced_dim",
        )
        for neg_repr in negative_reprs
    ]

    gallery_reprs = torch.cat(
        [positive_repr, *negative_reprs], dim=1
    )  # [batch, 1+num_neg, seq_len, reduced_dim]

    scores = maxsim(query_repr, gallery_reprs)  # [batch, 1+num_neg]
    return scores
