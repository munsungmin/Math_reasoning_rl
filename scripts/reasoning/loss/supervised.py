"""Model forward and supervised objectives have independent implementations."""


class CausalTokenLoss:
    def __call__(self, logits, labels):
        import torch
        import torch.nn.functional as F

        shifted = labels[:, 1:].contiguous()
        mask = shifted != -100
        count = int(mask.sum())
        if count == 0:
            raise ValueError("Supervised batch has no target tokens")
        token_losses = F.cross_entropy(
            logits[:, :-1][mask].float(), shifted[mask], reduction="none"
        )
        row_ids = torch.arange(shifted.shape[0], device=logits.device)[
            :, None
        ].expand_as(shifted)[mask]
        by_row = torch.zeros(shifted.shape[0], device=logits.device).scatter_add_(
            0, row_ids, token_losses
        )
        total = token_losses.sum()
        return {
            "loss": total / count,
            "loss_sum": total.detach(),
            "target_tokens": count,
            "sample_loss_sums": by_row.detach(),
            "sample_token_counts": mask.sum(-1),
        }
