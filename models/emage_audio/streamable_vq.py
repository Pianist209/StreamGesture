"""Adapters for the four 4x-downsampled StreamableVQVAEs.

The legacy EMAGE VQ-VAEs emit one code per motion frame.  The streamable
models emit one code for four frames, so this adapter owns the conversion
between full SMPL-X motion and four synchronized token streams.
"""

import importlib.util
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn as nn
import yaml

from .processing_emage_audio import axis_angle_to_rotation_6d, recover_from_mask_ts, rotation_6d_to_axis_angle


REPO_ROOT = Path(__file__).resolve().parents[2]
LEGACY_MODELS_DIR = REPO_ROOT.parent / "PantoMatrix-legacy/scripts/EMAGE_2024/models"
LEGACY_PACKAGE_NAME = "_streamable_emage_legacy"


def _streamable_vqvae_class():
    module_name = f"{LEGACY_PACKAGE_NAME}.streamable_motion"
    if module_name not in sys.modules:
        package = types.ModuleType(LEGACY_PACKAGE_NAME)
        package.__path__ = [str(LEGACY_MODELS_DIR)]
        sys.modules[LEGACY_PACKAGE_NAME] = package
        spec = importlib.util.spec_from_file_location(module_name, LEGACY_MODELS_DIR / "streamable_motion.py")
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
    return sys.modules[module_name].StreamableVQVAE


def _load_model_args(config_path):
    config_path = Path(config_path)
    if not config_path.is_absolute():
        config_path = REPO_ROOT / config_path
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    config["use_vq"] = True
    return SimpleNamespace(**config)


def _load_checkpoint(model, checkpoint_path):
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.is_absolute():
        checkpoint_path = REPO_ROOT / checkpoint_path
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state_dict = checkpoint.get("model_state", checkpoint)
    if all(key.startswith("module.") for key in state_dict):
        state_dict = {key[len("module."):]: value for key, value in state_dict.items()}
    model.load_state_dict(state_dict, strict=True)


class StreamableVQModel(nn.Module):
    """Frozen four-part VQ model with one token per four motion frames."""

    joint_mask_upper = [
        False, False, False, True, False, False, True, False, False, True,
        False, False, True, True, True, True, True, True, True, True,
        True, True, False, False, False, False, False, False, False, False,
        False, False, False, False, False, False, False, False, False, False,
        False, False, False, False, False, False, False, False, False, False,
        False, False, False, False, False,
    ]
    joint_mask_lower = [
        True, True, True, False, True, True, False, True, True, False,
        True, True, False, False, False, False, False, False, False, False,
        False, False, False, False, False, False, False, False, False, False,
        False, False, False, False, False, False, False, False, False, False,
        False, False, False, False, False, False, False, False, False, False,
        False, False, False, False, False,
    ]

    def __init__(self, face_model, upper_model, hands_model, lower_model):
        super().__init__()
        self.vq_model_face = face_model
        self.vq_model_upper = upper_model
        self.vq_model_hands = hands_model
        self.vq_model_lower = lower_model
        self.downsample_factor = upper_model.downsample_factor

    @classmethod
    def from_config(cls, config):
        vq_config = config.streamable_vq
        streamable_vqvae = _streamable_vqvae_class()
        models = []
        for part in ("face", "upper", "hands", "lower"):
            model = streamable_vqvae(_load_model_args(getattr(vq_config, f"{part}_config")))
            _load_checkpoint(model, getattr(vq_config, f"{part}_checkpoint"))
            models.append(model)
        return cls(*models)

    def split_inputs(self, smplx_body_rot6d, expression):
        batch_size, frames, channels = smplx_body_rot6d.shape
        body = smplx_body_rot6d.reshape(batch_size, frames, channels // 6, 6)
        return {
            "face": torch.cat((body[:, :, 22:23].reshape(batch_size, frames, 6), expression), dim=-1),
            "upper": body[:, :, self.joint_mask_upper].reshape(batch_size, frames, 78),
            "hands": body[:, :, 25:55].reshape(batch_size, frames, 180),
            "lower": body[:, :, self.joint_mask_lower].reshape(batch_size, frames, 54),
        }

    @torch.no_grad()
    def map2index(self, smplx_body_rot6d, expression):
        inputs = self.split_inputs(smplx_body_rot6d, expression)
        return {
            "face": self.vq_model_face.encode_to_indices(inputs["face"]),
            "upper": self.vq_model_upper.encode_to_indices(inputs["upper"]),
            "hands": self.vq_model_hands.encode_to_indices(inputs["hands"]),
            "lower": self.vq_model_lower.encode_to_indices(inputs["lower"]),
        }

    @torch.no_grad()
    def decode(self, face_index, upper_index, hands_index, lower_index):
        face_mix = self.vq_model_face.decode_indices(face_index)
        upper_6d = self.vq_model_upper.decode_indices(upper_index)
        hands_6d = self.vq_model_hands.decode_indices(hands_index)
        lower_6d = self.vq_model_lower.decode_indices(lower_index)
        return self._assemble(face_mix, upper_6d, hands_6d, lower_6d)

    @torch.no_grad()
    def decode_stream(self, indices, state=None):
        """Stream one token-chunk per body part, threading per-part decoder state.

        Each part's indices_chunk has shape (bs, T_tokens); feeding one token
        yields `downsample_factor` (=4) motion frames.  Returns (assembled dict,
        new_state) so the caller can keep the decoder states across steps.
        """
        if state is None:
            state = {"face": None, "upper": None, "hands": None, "lower": None}
        face_mix, s_face = self.vq_model_face.decode_stream(indices["face"], state["face"])
        upper_6d, s_upper = self.vq_model_upper.decode_stream(indices["upper"], state["upper"])
        hands_6d, s_hands = self.vq_model_hands.decode_stream(indices["hands"], state["hands"])
        lower_6d, s_lower = self.vq_model_lower.decode_stream(indices["lower"], state["lower"])
        new_state = {"face": s_face, "upper": s_upper, "hands": s_hands, "lower": s_lower}
        return self._assemble(face_mix, upper_6d, hands_6d, lower_6d), new_state

    def _assemble(self, face_mix, upper_6d, hands_6d, lower_6d):
        batch_size, frames, _ = upper_6d.shape
        face_jaw = rotation_6d_to_axis_angle(face_mix[:, :, :6].reshape(batch_size, frames, 1, 6)).reshape(batch_size, frames, 3)
        upper = rotation_6d_to_axis_angle(upper_6d.reshape(batch_size, frames, 13, 6)).reshape(batch_size, frames, 39)
        hands = rotation_6d_to_axis_angle(hands_6d.reshape(batch_size, frames, 30, 6)).reshape(batch_size, frames, 90)
        lower = rotation_6d_to_axis_angle(lower_6d.reshape(batch_size, frames, 9, 6)).reshape(batch_size, frames, 27)

        upper_all = recover_from_mask_ts(upper, self.joint_mask_upper)
        hands_all = recover_from_mask_ts(hands, [False] * 25 + [True] * 30)
        lower_all = recover_from_mask_ts(lower, self.joint_mask_lower)
        motion_axis_angle = upper_all + hands_all + lower_all
        motion_axis_angle[:, :, 22 * 3:22 * 3 + 3] = face_jaw
        motion_rot6d = axis_angle_to_rotation_6d(motion_axis_angle.reshape(batch_size, frames, 55, 3)).reshape(batch_size, frames, 330)
        return {
            "expression": face_mix[:, :, 6:],
            "motion_axis_angle": motion_axis_angle,
            "motion_rot6d": motion_rot6d,
            "trans": motion_axis_angle.new_zeros(batch_size, frames, 3),
        }
