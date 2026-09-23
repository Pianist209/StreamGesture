import torch
import torch.nn as nn


def build_causal_mask(length, device=None):
    return torch.triu(
        torch.ones(length, length, dtype=torch.bool, device=device),
        diagonal=1,
    )


class CausalTransformerEncoder(nn.Module):
    """Transformer encoder wrapper with future-frame blocking.

    Designed for streaming gesture generation. The API matches the existing
    motion_self_encoder usage while adding causal temporal attention.
    """

    def __init__(self, encoder_layer, num_layers=1):
        super().__init__()
        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
        )

    def forward(self, src):
        seq_len = src.shape[0]
        mask = build_causal_mask(seq_len, src.device)
        return self.encoder(src, mask=mask)


class CausalTransformerDecoder(nn.Module):
    """Transformer decoder wrapper for streaming cross attention."""

    def __init__(self, decoder_layer, num_layers=1):
        super().__init__()
        self.decoder = nn.TransformerDecoder(
            decoder_layer,
            num_layers=num_layers,
        )

    def forward(self, tgt, memory):
        tgt_mask = build_causal_mask(tgt.shape[0], tgt.device)
        return self.decoder(
            tgt,
            memory,
            tgt_mask=tgt_mask,
        )
