"""Dataset time alignment checks with in-memory audio/motion fixtures."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np


def load_dataset_module(fake_motion=None, fake_audio=None):
    if fake_motion is None:
        fake_motion = {
            "poses": np.zeros((6000, 165), dtype=np.float32),
            "expressions": np.zeros((6000, 100), dtype=np.float32),
            "trans": np.zeros((6000, 3), dtype=np.float32),
        }
    if fake_audio is None:
        fake_audio = np.arange(3000000, dtype=np.float32)
    fake_io = SimpleNamespace(beat_format_load=Mock(return_value=fake_motion), MASK_DICT={})
    fake_librosa = SimpleNamespace(load=Mock(return_value=(fake_audio, 16000)))
    path = Path(__file__).resolve().parents[1] / "datasets/beat2.py"
    spec = importlib.util.spec_from_file_location("beat2_alignment_fixture", path)
    dataset_module = importlib.util.module_from_spec(spec)
    with patch.dict("sys.modules", {"librosa": fake_librosa, "emage_utils.motion_io": fake_io}):
        spec.loader.exec_module(dataset_module)
    return dataset_module


class AudioAlignmentTests(unittest.TestCase):
    def test_long_clip_start_is_aligned_and_batch_lengths_are_constant(self):
        dataset_module = load_dataset_module()
        dataset = dataset_module.BEAT2DatasetEamge.__new__(dataset_module.BEAT2DatasetEamge)
        dataset.fps, dataset.audio_sr = 30, 16000
        dataset.mean, dataset.std = 0, 1
        starts = (0, 1, 2, 5400)
        dataset.data_list = [dict(start_idx=start, end_idx=start + 64,
                                  motion_path="fixture.npz", audio_path="fixture.wav") for start in starts]
        lengths = []
        for i, start in enumerate(starts):
            sample = dataset[i]
            self.assertLessEqual(abs(sample["audio"][0].item() / 16000 - start / 30), 0.5 / 16000)
            self.assertEqual(sample["motion"].shape[0], 64)
            lengths.append(sample["audio"].shape[0])
        self.assertEqual(lengths, [34134] * len(starts))
        self.assertEqual(dataset[3]["audio"][0].item(), 2880000)

    def test_manifest_rewindow_merges_only_continuous_same_source_ranges(self):
        dataset_module = load_dataset_module()
        items = [
            dict(video_id="clip_a", motion_path="a.npz", audio_path="a.wav", mode="train", start_idx=0, end_idx=64),
            dict(video_id="clip_a", motion_path="a.npz", audio_path="a.wav", mode="train", start_idx=20, end_idx=84),
            dict(video_id="clip_a", motion_path="a.npz", audio_path="a.wav", mode="train", start_idx=40, end_idx=104),
            dict(video_id="clip_a", motion_path="a.npz", audio_path="a.wav", mode="train", start_idx=200, end_idx=264),
            dict(video_id="clip_a", motion_path="a.npz", audio_path="a.wav", mode="val", start_idx=0, end_idx=104),
            dict(video_id="clip_b", motion_path="b.npz", audio_path="b.wav", mode="train", start_idx=0, end_idx=104),
        ]
        windows = dataset_module.rewindow_motion_metadata(items, clip_length=100, stride=20)
        spans = [(w["mode"], w["motion_path"], w["start_idx"], w["end_idx"]) for w in windows]
        self.assertEqual(spans, [
            ("train", "a.npz", 0, 100),
            ("train", "b.npz", 0, 100),
            ("val", "a.npz", 0, 100),
        ])

    def test_rewindowed_emage_dataset_returns_256_frame_clips(self):
        dataset_module = load_dataset_module()
        cfg = SimpleNamespace(
            data=SimpleNamespace(rewindow_stride=20),
            model=SimpleNamespace(pose_length=256, pose_fps=30, audio_sr=16000, joint_mask=None),
        )
        dataset = dataset_module.BEAT2DatasetEamge.__new__(dataset_module.BEAT2DatasetEamge)
        dataset.fps, dataset.audio_sr = 30, 16000
        dataset.mean, dataset.std = 0, 1
        dataset.data_list = [dict(start_idx=100, end_idx=356, motion_path="fixture.npz", audio_path="fixture.wav")]
        sample = dataset[0]
        self.assertEqual(sample["motion"].shape[0], 256)
        self.assertEqual(sample["expressions"].shape[0], 256)
        self.assertEqual(sample["audio"].shape[0], 136534)
        self.assertEqual(sample["audio"][0].item(), round(100 * 16000 / 30))


if __name__ == "__main__":
    unittest.main()
