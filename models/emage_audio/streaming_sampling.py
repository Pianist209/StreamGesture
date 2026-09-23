import torch
import torch.nn.functional as F


def sample_token(logits, temperature=1.0, top_k=0):
    """Sample VQ tokens instead of always taking argmax.

    This reduces mode collapse in streaming generation where argmax decoding
    tends to select conservative motion codes.
    """
    if temperature <= 0:
        raise ValueError("temperature must be positive")

    logits = logits / temperature

    if top_k > 0:
        values, _ = torch.topk(logits, min(top_k, logits.shape[-1]), dim=-1)
        threshold = values[..., -1:].contiguous()
        logits = torch.where(
            logits < threshold,
            torch.full_like(logits, float("-inf")),
            logits,
        )

    probs = F.softmax(logits, dim=-1)
    return torch.multinomial(
        probs.reshape(-1, probs.shape[-1]),
        num_samples=1,
    ).reshape(*probs.shape[:-1])


def logits_to_token(logits, deterministic=False, temperature=0.8, top_k=50):
    if deterministic:
        return logits.argmax(dim=-1)
    return sample_token(logits, temperature=temperature, top_k=top_k)
