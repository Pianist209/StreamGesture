# 32-token 自回归窗口与流式输出

本实现按 LiveGesture 风格的 xAR 方式工作：每一步只预测一个 motion token。一个 token 对应 `token_downsample_factor` 帧动作，默认 4 帧。预测出的 face / upper / hands / lower 四个 token 一起送入冻结的 sVQ decoder，得到下一段动作。

模型现在使用固定的 motion 历史窗口。默认 `history_window_tokens: 32`，所以稳定阶段的语义是：

```text
q_t = f(q_{t-32:t-1}, audio_window, speaker)
q_t -> 4 frames motion
```

序列开头历史不足 32 个 token 时使用 BOS 和已有历史；达到 32 个 token 后不再访问更早的 motion token。训练和流式推理都走同一套窗口语义。

## 训练时间轴

配置里的训练片段改为 256 帧：

```yaml
data:
  train_bs: 2
  rewindow_stride: 20

model:
  pose_length: 256
  token_downsample_factor: 4
  history_window_tokens: 32
```

原始 manifest 仍然是 `beat2_s20_l64_speaker2.json`。`BEAT2DatasetEamge` 会把同一视频、同一 split、同一音频/动作文件中连续覆盖的 64 帧条目合并，再按 256 帧窗口重组样本。它不会跨视频、跨 split、跨文件或跨中间缺口拼接。

256 帧片段包含 B0…B63。训练目标是 B1…B63，所以每个样本有 63 个 token 目标。输入 token 是右移后的 teacher-forcing 历史：

```text
target: q1, q2, ..., q63
input : BOS, q1, ..., q62
```

`forward()` 会先用整段音频算出 causal audio tokens，再为每个目标位置取同样长度的局部窗口。位置 32 之后，每个预测只看最近 32 个输入 token 和对齐的最近 32 个 audio token。没有随机 start 裁剪，也不是一次性生成 63 个 token；只是把 63 个 teacher-forcing 位置并行算 loss。

## 推理时间轴

在线推理从 BOS 开始，每收到一个完整音频 block，就预测下一个 motion token：

```python
model.eval()
motion_vq.eval()
state = None
for audio_chunk in audio_source:  # (batch, samples)
    ready, state = model.stream_step(
        audio_chunk, speaker_id, state=state, motion_vq=motion_vq,
    )
    for block in ready:
        consume(block["motion"], block["start_frame"])
```

`state["tokens"]` 和 `state["audio_features"]` 都最多保留 32 个位置。上一版完整前缀 KV cache 已移除，所以 transformer 推理不会无限增长 motion-token attention state。

音频 encoder 仍然保留自己的流式状态，包括 STFT 未完成样本、mel 卷积上下文和未完成池化区间。这是为了保证流式音频特征和离线 causal 音频特征一致；它不是 motion token 历史，也不会改变“只看最近 32 个 motion token”的语义。

`inference_stream(audio, speaker_id, motion_vq)` 是离线录音的流式适配器。`generate_motion(...)` 收集同一条流式路径的动作输出，用于保存文件和评估。第一次输出仍然从原音频第 4 帧开始，即收到 B0 音频后预测 B1。

## 评估

validation CE 仍然是 teacher forcing，但每个位置用固定 32-token 窗口计算。validation FGD 使用从 BOS 自由生成的 token，并通过同一条 `stream_step` / `decode_stream` 路径解码动作。

拉取这次改动后需要重新训练 predictor；上一版完整前缀 KV cache 语义下训练出的权重不再对应当前推理范式。
