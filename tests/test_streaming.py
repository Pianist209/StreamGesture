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
    def test_kv_cache_matches_parallel_teacher_forcing_beyond_32(self):
        n = 40
        audio = torch.randn(2, (n * 4 * 16000 + 29) // 30)
        targets = {part: torch.randint(0, 16, (2, n)) for part in self.model.PARTS}
        inputs = shift_tokens_with_bos(targets, self.model.bos_id)
        expected = self.model(audio, self.speaker, inputs)
        audio_features = self.model.audio_encoder(audio, target_tokens=n)
        cache = None
        outputs = []
        for i in range(n):
            logits, cache = self.model._token_step(
                audio_features[:, i:i+1], self.speaker,
                {part: tokens[:, i:i+1] for part, tokens in inputs.items()}, cache,
            )
            outputs.append(logits)
        for key in expected:
            actual = torch.cat([out[key] for out in outputs], dim=1)
            torch.testing.assert_close(actual, expected[key], atol=3e-6, rtol=3e-5)
        self.assertEqual(cache["regions"]["face"][0][0][0].shape[-2], n)

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
    def test_free_running_matches_full_prefix_and_decodes_each_step(self):
        audio = torch.randn(2, 17067)
        decoder = CountingDecoder()
        outputs = list(self.model.inference_stream(audio, self.speaker, decoder))
        self.assertEqual(len(outputs), 7)
        history = {part: torch.full((2, 1), self.model.bos_id) for part in self.model.PARTS}
        for i, out in enumerate(outputs):
            end = ((i + 1) * 4 * 16000 + 29) // 30
            expected = self.model(audio[:, :end], self.speaker, history)
            for part in self.model.PARTS:
                key = f"cls_{part}"
                torch.testing.assert_close(out["logits"][key], expected[key][:, -1:], atol=3e-6, rtol=3e-5)
                history[part] = torch.cat((history[part], out["indices"][part]), dim=1)
            self.assertEqual(out["start_frame"], (i + 1) * 4)
            self.assertEqual(out["motion"]["motion"].shape[1], 4)
            self.assertTrue((out["motion"]["decoder_step"] == i).all())
        collected = self.model.generate_motion(audio, self.speaker, decoder)
        self.assertEqual(collected["motion"].shape[1], 28)
        # Arbitrary live chunks produce the same first seven forecasts. Feeding
        # the eighth block also forecasts B8, which the file adapter excludes.
        state, live = None, []
        for start in range(0, audio.shape[1], 997):
            result, state = self.model.stream_step(audio[:, start:start+997], self.speaker, state, decoder)
            live.extend(result)
        self.assertEqual(len(live), 8)
        for expected, actual in zip(outputs, live):
            for part in self.model.PARTS:
                torch.testing.assert_close(actual["logits"][f"cls_{part}"], expected["logits"][f"cls_{part}"], atol=3e-6, rtol=3e-5)
        self.assertEqual(state["decoder"], 8)

    def test_complete_sequence_has_gradients(self):
        self.model.train()
        targets = {part: torch.randint(0, 16, (2, 15)) for part in self.model.PARTS}
        inputs = shift_tokens_with_bos(targets, self.model.bos_id)
        outputs = self.model(torch.randn(2, 34134), self.speaker, inputs)
        loss = sum(torch.nn.functional.cross_entropy(outputs[f"cls_{part}"].transpose(1, 2), targets[part])
                   for part in self.model.PARTS)
        loss.backward()
        for name, parameter in self.model.named_parameters():
            self.assertIsNotNone(parameter.grad, name)
            self.assertTrue(torch.isfinite(parameter.grad).all(), name)


if __name__ == "__main__":
    unittest.main()
