"""CPU checks without datasets, SMPL-X assets or external sVQ checkpoints.

Run: python -m unittest discover -s tests -v
"""
import unittest

import torch

from models.emage_audio import CausalEmageAudioTokenModel, EmageAudioConfig, shift_tokens_with_bos
from models.emage_audio.processing_emage_audio import CausalMelAudioEncoder


class CountingDecoder:
    """Only checks predictor/decoder state plumbing, not the real sVQ network."""

    def decode_stream(self, indices, state=None):
        step = 0 if state is None else state
        frames = indices["face"].unsqueeze(-1).float().repeat_interleave(4, dim=1)
        return {"motion": frames, "decoder_step": torch.full_like(frames, step)}, step + 1


class StreamingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def setUp(self):
        torch.manual_seed(123)
        self.model = CausalEmageAudioTokenModel(EmageAudioConfig(
            hidden_size=16, num_heads=2, audio_f=8, audio_sr=16000,
            token_downsample_factor=4, pose_fps=30, speaker_dims=1,
            vae_codebook_size=16, region_transformer_layers=2, fusion_layers=2,
            history_window_tokens=32,
        )).eval()
        self.speaker = torch.zeros(2, 1, dtype=torch.long)

    @torch.no_grad()
    def test_audio_chunking_matches_full_encoding(self):
        for factor in (3, 4, 8):
            encoder = CausalMelAudioEncoder(16, hidden_f=8, token_downsample_factor=factor).eval()
            waveform = torch.randn(2, 19201)
            n_tokens = waveform.shape[1] * 30 // (factor * 16000)
            expected = encoder(waveform, target_tokens=n_tokens)
            for sizes in ((71, 159, 401, 2203), (2133, 2134), (19201,)):
                state, chunks, start, i = None, [], 0, 0
                while start < waveform.shape[1]:
                    end = start + sizes[i % len(sizes)]
                    chunk, state = encoder.forward_stream(waveform[:, start:end], state)
                    chunks.append(chunk)
                    self.assertLess(state["samples"].shape[1], encoder.n_fft)
                    self.assertLessEqual(state["context"].shape[-1], encoder.mel_context)
                    start, i = end, i + 1
                actual = torch.cat(chunks, dim=1)
                torch.testing.assert_close(actual, expected, atol=3e-6, rtol=3e-5)

    @torch.no_grad()
    def test_training_forward_matches_explicit_32_token_windows(self):
        n = 40
        audio = torch.randn(2, (n * 4 * 16000 + 29) // 30)
        targets = {part: torch.randint(0, 16, (2, n)) for part in self.model.PARTS}
        inputs = shift_tokens_with_bos(targets, self.model.bos_id)
        batched = self.model(audio, self.speaker, inputs)
        audio_features = self.model.audio_encoder(audio, target_tokens=n)

        explicit = {f"cls_{part}": [] for part in self.model.PARTS}
        for j in range(n):
            start = max(0, j - self.model.history_window_tokens + 1)
            logits = self.model._predict_window(
                audio_features[:, start:j + 1], self.speaker,
                {part: tokens[:, start:j + 1] for part, tokens in inputs.items()},
            )
            for part in self.model.PARTS:
                explicit[f"cls_{part}"].append(logits[f"cls_{part}"][:, -1:])
        for key in batched:
            expected = torch.cat(explicit[key], dim=1)
            torch.testing.assert_close(batched[key], expected, atol=3e-6, rtol=3e-5)

    @torch.no_grad()
    def test_tokens_before_32_window_do_not_change_later_prediction(self):
        n = 40
        audio = torch.randn(2, (n * 4 * 16000 + 29) // 30)
        targets = {part: torch.randint(0, 16, (2, n)) for part in self.model.PARTS}
        inputs = shift_tokens_with_bos(targets, self.model.bos_id)
        expected = self.model(audio, self.speaker, inputs)

        changed_inputs = {part: tokens.clone() for part, tokens in inputs.items()}
        last_start = n - self.model.history_window_tokens
        for part in self.model.PARTS:
            changed_inputs[part][:, :last_start] = (changed_inputs[part][:, :last_start] + 7) % 16
        actual = self.model(audio, self.speaker, changed_inputs)

        for key in expected:
            torch.testing.assert_close(actual[key][:, -1:], expected[key][:, -1:], atol=3e-6, rtol=3e-5)

    @torch.no_grad()
    def test_future_audio_and_tokens_do_not_change_prefix(self):
        audio = torch.randn(2, 20000)
        targets = {part: torch.randint(0, 16, (2, 8)) for part in self.model.PARTS}
        inputs = shift_tokens_with_bos(targets, self.model.bos_id)
        expected = self.model(audio, self.speaker, inputs)
        changed_audio = audio.clone()
        changed_audio[:, 6400:] = torch.randn_like(changed_audio[:, 6400:]) * 10
        changed_inputs = {part: tokens.clone() for part, tokens in inputs.items()}
        for tokens in changed_inputs.values():
            tokens[:, 3:] = torch.randint(0, 16, tokens[:, 3:].shape)
        actual = self.model(changed_audio, self.speaker, changed_inputs)
        for key in expected:
            torch.testing.assert_close(actual[key][:, :3], expected[key][:, :3])

    @torch.no_grad()
    def test_free_running_matches_strict_window_and_decodes_each_step(self):
        n = 40
        audio = torch.randn(2, (n * 4 * 16000 + 29) // 30)
        decoder = CountingDecoder()
        outputs = list(self.model.inference_stream(audio, self.speaker, decoder, num_tokens=n))
        self.assertEqual(len(outputs), n)

        audio_features = self.model.audio_encoder(audio, target_tokens=n)
        history = {part: torch.full((2, 1), self.model.bos_id) for part in self.model.PARTS}
        for i, out in enumerate(outputs):
            window_len = min(self.model.history_window_tokens, i + 1)
            start = i + 1 - window_len
            logits = self.model._predict_window(
                audio_features[:, start:i + 1], self.speaker,
                {part: tokens[:, -window_len:] for part, tokens in history.items()},
            )
            for part in self.model.PARTS:
                key = f"cls_{part}"
                torch.testing.assert_close(out["logits"][key], logits[key][:, -1:], atol=3e-6, rtol=3e-5)
                history[part] = torch.cat((history[part], out["indices"][part]), dim=1)
            self.assertEqual(out["start_frame"], (i + 1) * 4)
            self.assertEqual(out["motion"]["motion"].shape[1], 4)
            self.assertTrue((out["motion"]["decoder_step"] == i).all())

        collected = self.model.generate_motion(audio, self.speaker, CountingDecoder(), num_tokens=n)
        self.assertEqual(collected["motion"].shape[1], n * 4)

        state, live = None, []
        for start in range(0, audio.shape[1], 997):
            result, state = self.model.stream_step(audio[:, start:start + 997], self.speaker, state, decoder)
            live.extend(result)
        self.assertEqual(len(live), n)
        for expected, actual in zip(outputs, live):
            for part in self.model.PARTS:
                torch.testing.assert_close(actual["logits"][f"cls_{part}"], expected["logits"][f"cls_{part}"], atol=3e-6, rtol=3e-5)
        self.assertLessEqual(state["tokens"]["face"].shape[1], 32)
        self.assertLessEqual(state["audio_features"].shape[1], 32)
        self.assertNotIn("attention", state)
        self.assertEqual(state["decoder"], n)

    def test_complete_sequence_has_gradients(self):
        self.model.train()
        n = 63
        targets = {part: torch.randint(0, 16, (2, n)) for part in self.model.PARTS}
        inputs = shift_tokens_with_bos(targets, self.model.bos_id)
        outputs = self.model(torch.randn(2, 136534), self.speaker, inputs)
        loss = sum(torch.nn.functional.cross_entropy(outputs[f"cls_{part}"].transpose(1, 2), targets[part])
                   for part in self.model.PARTS)
        loss.backward()
        for name, parameter in self.model.named_parameters():
            self.assertIsNotNone(parameter.grad, name)
            self.assertTrue(torch.isfinite(parameter.grad).all(), name)


if __name__ == "__main__":
    unittest.main()
