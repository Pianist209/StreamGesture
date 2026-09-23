"""Dataset time alignment checks with in-memory audio/motion fixtures."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np


class AudioAlignmentTests(unittest.TestCase):
    def test_long_clip_start_is_aligned_and_batch_lengths_are_constant(self):
        waveform = np.arange(3000000, dtype=np.float32)
        motion = {
            "poses": np.zeros((6000, 165), dtype=np.float32),
            "expressions": np.zeros((6000, 100), dtype=np.float32),
            "trans": np.zeros((6000, 3), dtype=np.float32),
        }
        fake_io = SimpleNamespace(beat_format_load=Mock(return_value=motion), MASK_DICT={})
        fake_audio = SimpleNamespace(load=Mock(return_value=(waveform, 16000)))
        path = Path(__file__).resolve().parents[1] / "datasets/beat2.py"
        spec = importlib.util.spec_from_file_location("beat2_alignment_fixture", path)
        dataset_module = importlib.util.module_from_spec(spec)
        with patch.dict("sys.modules", {"librosa": fake_audio, "emage_utils.motion_io": fake_io}):
            spec.loader.exec_module(dataset_module)
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


if __name__ == "__main__":
    unittest.main()
