from dataclasses import dataclass


@dataclass
class StreamingConfig:
    """Runtime options for causal streaming gesture generation."""

    causal: bool = True
    use_stream_decoder: bool = True
    chunk_size: int = 32
    context_frames: int = 16
    boundary_loss_weight: float = 0.5
    velocity_loss_weight: float = 0.2
    token_temperature: float = 1.0


def enable_streaming(config):
    """Attach streaming options to an existing config object."""
    config.streaming_mode = True
    config.causal_attention = True
    return config
