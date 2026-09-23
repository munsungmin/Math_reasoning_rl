"""Parameter geometry. These functions do not know trainers, datasets or loggers."""


def lora_singular_values(a, b, scale=1.0):
    import torch

    # The nonzero spectrum of B A comes from a rank-sized matrix, not a dense LM weight.
    _, rb = torch.linalg.qr(b.detach().float().cpu(), mode="reduced")
    _, ra = torch.linalg.qr(a.detach().float().cpu().T, mode="reduced")
    return torch.linalg.svdvals(rb @ ra.T) * abs(scale)


def spectrum_statistics(singular_values):
    import torch

    values = singular_values.detach().double()
    total = values.sum()
    probabilities = values / total if total > 0 else values
    nonzero = probabilities[probabilities > 0]
    return {
        "frobenius_norm": float(torch.linalg.vector_norm(values)),
        "spectral_norm": float(values.max()) if values.numel() else 0.0,
        "effective_rank": float(torch.exp(-(nonzero * nonzero.log()).sum()))
        if total > 0
        else 0.0,
    }
