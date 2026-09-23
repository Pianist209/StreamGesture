# 完整前缀自回归与流式输出

本实现按参考图的约定，每一步根据全部过去 token 和已收到的音频，
预测一个时间位置的 face / upper / hands / lower 四个 token；四个 token
一起送入冻结的 sVQ decoder，输出 4 帧动作。没有随机 start 裁剪，也没有
32-token 滑动截断。

## 训练与时间轴

- 30 FPS、4 帧一个 block。64 帧训练片段包含 B0…B15。
- 保留原来的预测目标：将 B1…B15 编码为 q1…q15，输入是
  `[BOS, q1, …, q14]`。完整序列的 15 个位置通过因果 mask 并行计算 CE。
- 推理从 BOS 开始，每步只产生一个新时间位置，并反馈自己的预测。
  teacher forcing 的并行训练不等于推理一次生成 15 个 token。
- 收到 B0 的音频后预测 B1。因此第一次输出从原音频第 4 帧开始，
  即 4/30 秒；这仍然是预测未来动作块的任务，不是同块音频重建。
- NPZ 的 `start_time_seconds` 记录原始时间轴偏移。渲染裁掉相同长度的
  音频，并将 GT / 额外参考动作裁到对应区间。
- 离线文件输出 B1…B(N-1)，不补造 B0，不输出文件末尾之后的预测；
  在线 `stream_step` 每收到完整音频块都会预测下一块。

## 在线调用

```python
model.eval()
motion_vq.eval()
state = None
for audio_chunk in audio_source:  # (batch, samples)，与模型相同设备和采样率
    ready, state = model.stream_step(
        audio_chunk, speaker_id, state=state, motion_vq=motion_vq,
    )
    for block in ready:
        # block['indices']: 每个部位 (batch, 1)
        # block['motion']: 4 帧 motion_axis_angle / expression / trans 等
        # block['start_frame'], block['end_frame']: 原始音频时间轴
        consume(block['motion'], block['start_frame'])
```

`consume` 和 `audio_source` 是调用方接口。一个分块可以产生零个、一个或多个
输出块；不足一个时间区间的音频保留在 state 中，不用未来数据补齐。
每次新会话将 state 重置为 None；同一会话保持 batch、speaker 和 decoder 不变。

音频状态保留 STFT 的未处理样本、有限 mel 卷积上下文和未完成的池化区间。
注意力使用带全局位置偏移的 RoPE 和每层 KV cache。VQ decoder 的状态跨步保存。
完整历史的 KV 内存随时长增长，注意力仍需访问历史；这里不声称恒定内存或已达到
实测实时性能。

`inference_stream(audio, speaker_id, motion_vq)` 可逐块迭代已有录音。
`generate_motion(...)` 收集同一条流式路径的动作输出，供保存文件和评估使用。
`inference(...)` 保留收集 token logits 的接口。

## 修复与评估

- 音频切片先按绝对帧时间换算起点，再取整；固定片段长度向上取整，避免
  16000//30 累积漂移，并保持同长样本可堆叠成 batch。
- STFT 第 k 帧的结束位置为 `k * hop + n_fft`，k 从 0 开始。
  池化的分母使用配置的 token_downsample_factor，而不是写死 4。
- validation CE 仍是 teacher forcing；用于选模的 validation FGD 改为
  从 BOS 自由生成并流式解码后的动作。配置开启整段录音的测试指标，
  可用 `validation.evaluation=False` 关闭耗时的整段评估。

训练命令保持不变，例如：

```bash
torchrun --standalone --nproc_per_node=1 train_emage_audio.py --config configs/emage_streamable_audio.yaml
python -m unittest discover -s tests -v
```

测试覆盖分块不变性、缓存/完整前缀等价性、未来信息隔离、逐步 decoder 状态、
完整训练序列反向传播、长录音的音频切片对齐。decoder 测试使用替身，只验证调用
协议；真实 sVQ 仍依赖配置指定的外部 PantoMatrix-legacy 源码和权重。

音频对齐与输入分布已修正，建议重新训练；旧权重的参数形状兼容并不意味着效果
保持不变。64 帧数据仍只监督最多 15 个历史位置，长历史生成质量需要长录音评估，
必要时使用更长训练片段，而不是恢复随机裁剪。尚未用真实数据、GPU 或 sVQ 权重
验证动作质量和延迟。
