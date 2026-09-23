import torch


def causal_attention_mask(seq_len, device=None):
    """Mask future positions for causal temporal attention."""
    return torch.triu(
        torch.ones(seq_len, seq_len, dtype=torch.bool, device=device),
        diagonal=1,
    )


def causal_cross_attention_mask(query_len, memory_len, device=None):
    """Mask future memory positions in aligned streaming attention."""
    if query_len != memory_len:
        return None
    return torch.triu(
        torch.ones(query_len, memory_len, dtype=torch.bool, device=device),
        diagonal=1,
    )
