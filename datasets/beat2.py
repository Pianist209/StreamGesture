import json
import math
import torch
from torch.utils import data
import numpy as np
import librosa
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from emage_utils.motion_io import beat_format_load, MASK_DICT


def _cfg_get(obj, key, default=None):
    if hasattr(obj, "get"):
        return obj.get(key, default)
    return getattr(obj, key, default)


def rewindow_motion_metadata(items, clip_length, stride):
    """Build fixed-length windows inside continuous manifest coverage.

    The shipped BEAT2 manifest is made from overlapping 64-frame clips. For
    AR training with a 32-token history we need longer examples, so this
    merges only overlapping/touching ranges from the same source file and
    re-emits deterministic windows. It never crosses split, video, audio or
    motion-file boundaries, and it does not invent frames outside the original
    covered ranges.
    """
    if clip_length <= 0 or stride <= 0:
        raise ValueError("clip_length and stride must be positive")

    groups = {}
    for item in items:
        key = (
            item.get("mode"),
            item.get("video_id"),
            item["motion_path"],
            item["audio_path"],
        )
        groups.setdefault(key, []).append(item)

    windows = []
    for key in sorted(groups):
        group = sorted(groups[key], key=lambda x: (x["start_idx"], x["end_idx"]))
        spans = []
        for item in group:
            start, end = int(item["start_idx"]), int(item["end_idx"])
            if not spans or start > spans[-1][1]:
                spans.append([start, end, item])
            else:
                spans[-1][1] = max(spans[-1][1], end)

        for start, end, template in spans:
            last_start = end - clip_length
            current = start
            while current <= last_start:
                new_item = dict(template)
                new_item["start_idx"] = current
                new_item["end_idx"] = current + clip_length
                new_item["video_id"] = f"{template.get('video_id', 'clip')}_{current}_{current + clip_length}"
                windows.append(new_item)
                current += stride
    return windows

class BEAT2Dataset(data.Dataset):
    def __init__(self, cfg, split):
        vid_meta = []
        for data_meta_path in cfg.data.meta_paths:
            vid_meta.extend(json.load(open(data_meta_path, "r")))
        self.vid_meta = [item for item in vid_meta if item.get("mode") == split]
        self.mean = 0
        self.std = 1
        self.joint_mask = MASK_DICT[cfg.model.joint_mask] if cfg.model.joint_mask is not None else None
        self.data_list = self.vid_meta
        self.fps = cfg.model.pose_fps
        self.audio_sr = cfg.model.audio_sr

    def __len__(self):
        return len(self.data_list)
    
    @staticmethod
    def normalize(motion, mean, std):
        return (motion - mean) / (std + 1e-7)
    
    @staticmethod
    def inverse_normalize(motion, mean, std):
        return motion * std + mean

    def __getitem__(self, item):
        data_item = self.data_list[item]
        smplx_data = beat_format_load(data_item["motion_path"], mask=self.joint_mask)
        sdx, edx = data_item["start_idx"], data_item["end_idx"]
        motion = smplx_data["poses"][sdx:edx]
        SMPLX_FPS = 30
        downsample_factor = SMPLX_FPS // self.fps
        motion = motion[::downsample_factor]
        motion = self.normalize(motion, self.mean, self.std)
        
        audio, _ = librosa.load(data_item["audio_path"], sr=self.audio_sr)
        sdx_audio = round(sdx * self.audio_sr / SMPLX_FPS)
        edx_audio = sdx_audio + math.ceil((edx - sdx) * self.audio_sr / SMPLX_FPS)
        audio = audio[sdx_audio:edx_audio]
             
        motion_tensor = torch.from_numpy(motion).float()
        audio_tensor = torch.from_numpy(audio).float()
       
        return dict(
            motion=motion_tensor,
            audio=audio_tensor, 
        )

class BEAT2DatasetEamge(BEAT2Dataset):
    def __init__(self, cfg, split):
        super().__init__(cfg, split)
        rewindow_stride = _cfg_get(cfg.data, "rewindow_stride", None)
        if rewindow_stride is not None:
            smplx_fps = 30
            raw_clip_length = int(_cfg_get(
                cfg.data, "rewindow_length",
                cfg.model.pose_length * (smplx_fps // cfg.model.pose_fps),
            ))
            self.vid_meta = rewindow_motion_metadata(
                self.vid_meta, raw_clip_length, int(rewindow_stride)
            )
            self.data_list = self.vid_meta

    def __getitem__(self, item):
        data_item = self.data_list[item]
        smplx_data = beat_format_load(data_item["motion_path"], mask=None)
        sdx, edx = data_item["start_idx"], data_item["end_idx"]
        motion = smplx_data["poses"][sdx:edx]
        expressions = smplx_data["expressions"][sdx:edx]
        trans = smplx_data["trans"][sdx:edx]
        SMPLX_FPS = 30
        downsample_factor = SMPLX_FPS // self.fps
        motion = motion[::downsample_factor]
        motion = self.normalize(motion, self.mean, self.std)
        
        audio, _ = librosa.load(data_item["audio_path"], sr=self.audio_sr)
        # Round the absolute start, not samples/frame; use a fixed duration so
        # equal-length motion clips still collate into equal-length waveforms.
        sdx_audio = round(sdx * self.audio_sr / SMPLX_FPS)
        edx_audio = sdx_audio + math.ceil((edx - sdx) * self.audio_sr / SMPLX_FPS)
        audio = audio[sdx_audio:edx_audio]
             
        motion_tensor = torch.from_numpy(motion).float()
        audio_tensor = torch.from_numpy(audio).float()
        expressions_tesnor = torch.from_numpy(expressions).float()
        trans_tensor = torch.from_numpy(trans).float()

        return dict(
            motion=motion_tensor,
            audio=audio_tensor, 
            expressions=expressions_tesnor,
            trans=trans_tensor,
        )


class BEAT2DatasetEamgeFootContact(BEAT2Dataset):
    def __init__(self, cfg, split):
        super().__init__(cfg, split)

    def __getitem__(self, item):
        data_item = self.data_list[item]
        smplx_data = beat_format_load(data_item["motion_path"], mask=None)
        sdx, edx = data_item["start_idx"], data_item["end_idx"]
        motion = smplx_data["poses"][sdx:edx]
        expressions = smplx_data["expressions"][sdx:edx]
        trans = smplx_data["trans"][sdx:edx]
        foot_contact = np.load(data_item["motion_path"].replace("smplxflame_30", "footcontact").replace(".npz", ".npy"))[sdx:edx]

        SMPLX_FPS = 30
        downsample_factor = SMPLX_FPS // self.fps
        motion = motion[::downsample_factor]
        motion = self.normalize(motion, self.mean, self.std)
        
        audio, _ = librosa.load(data_item["audio_path"], sr=self.audio_sr)
        sdx_audio = round(sdx * self.audio_sr / SMPLX_FPS)
        edx_audio = sdx_audio + math.ceil((edx - sdx) * self.audio_sr / SMPLX_FPS)
        audio = audio[sdx_audio:edx_audio]
             
        motion_tensor = torch.from_numpy(motion).float()
        audio_tensor = torch.from_numpy(audio).float()
        expressions_tesnor = torch.from_numpy(expressions).float()
        trans_tensor = torch.from_numpy(trans).float()
        foot_contact_tensor = torch.from_numpy(foot_contact).float()
        # print(trans_tensor.shape, foot_contact_tensor.shape)

        return dict(
            motion=motion_tensor,
            audio=audio_tensor, 
            expressions=expressions_tesnor,
            trans=trans_tensor,
            foot_contact=foot_contact_tensor,
        )
