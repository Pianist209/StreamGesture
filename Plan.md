对，那前面的改造路线要修正：**VQ encoder 保留，不动。**如果你是按 LiveGesture 的 SVQ 思路做，这个选择是有依据的。LiveGesture 明确使用 **bidirectional encoder + causal stream decoder**，并且说明推理阶段只使用 causal decoder，bidirectional encoder 只在离线训练/构造 latent token 时使用。([ResearchGate][1])

所以你的重点应该从“把 VQ encoder causal 化”转到 **Stage-2 条件训练、VQ decoder 的启动方式、audio encoder 和真正 streaming inference**。

## 先把你的最终语义定死

设 4 帧一个 block：

$$
B_i=[m_{4i},m_{4i+1},m_{4i+2},m_{4i+3}]
$$

你推理希望：

$$
B_{i-1}+A_{\le t_i}+speaker
\rightarrow q_i
\rightarrow B_i
$$

其中：

$$
t_i=\frac{4i}{30}
$$

也就是：

```text
已知 seed B0

B0 + audio<=t1 → token1 → B1
B1 + audio<=t2 → token2 → B2
B2 + audio<=t3 → token3 → B3
...
```

这就是接下来所有代码的统一标准。

---

## 第一步：VQ-VAE 暂时完全不要改

保留现在：

```text
Bidirectional StreamableEncoder
          ↓
          VQ
          ↓
CausalStreamDecoder
```

这实际上非常接近 LiveGesture 的 asymmetric SVQ。论文就是利用 bidirectional encoder 学更丰富的 latent，但只有 causal decoder 进入在线生成系统。([ResearchGate][1])

你当前 decoder 的：

```python
decode_stream(indices_chunk, state)
```

也已经验证过和 full causal decode 一致，所以 decoder 架构也不用改。

但有一件事需要改：

> **在线 inference 不要调用 bidirectional VQ encoder 去 encode seed pose。**

LiveGesture 也是明确说 encoder 不参与 inference。([ResearchGate][1])

---

# 第二步：Stage-2 target 的构造方式要改，这里很关键

我现在不建议：

```python
latent_index_dict = motion_vq.map2index(
    motion_gt,
    expressions_gt,
)

target = token[:, 1:]
```

原因是这样 `q1` 在 VQ decoder 训练语义中是**第二个 token**，decode 时它原本有 `q0` 的 decoder history。

但你在线推理：

```text
seed B0 是外部 motion
↓
第一个真正生成的 token 是 B1 的 token
↓
VQ decoder state 还是空的
```

会产生 decoder-state mismatch。

### 更干净的训练方式

假设：

```text
motion:
B0 B1 B2 ... B15
```

其中 B0 是 seed。

Stage-2 构造 target 时，直接把 seed 从 VQ target sequence 中去掉：

```python
factor = 4

future_motion = motion_gt[:, factor:]
future_expr = expressions_gt[:, factor:]

latent_index_dict = motion_vq.map2index(
    future_motion,
    future_expr,
)
```

这样 VQ encoder 实际看到：

```text
B1 B2 B3 ... B15
```

得到：

```text
q'_0 q'_1 q'_2 ... q'_14
```

这里：

```text
q'_0 → B1
q'_1 → B2
...
```

更重要的是，`q'_0` 本来就是 **这个 VQ sequence 的第一个 token**。

所以：

```python
decode_stream(q'_0, state=None)
```

正是它应该有的 decoder 状态。

这样你完全不需要用 seed 给 decoder warm-up。

---

# 第三步：past-motion condition 则同时 shift 一块

对应：

```text
target motion:
B1 B2 B3 ... B15

past condition:
B0 B1 B2 ... B14
```

代码就是：

```python
factor = 4

past_motion = motion_gt[:, :-factor]

future_motion = motion_gt[:, factor:]
future_expr = expressions_gt[:, factor:]

target_index = motion_vq.map2index(
    future_motion,
    future_expr,
)

pred = model(
    audio=audio,
    speaker_id=speaker_id,
    past_motion=past_motion,
)
```

于是：

```text
past_motion token 0 = B0 → target q'_0 → B1
past_motion token 1 = B1 → target q'_1 → B2
past_motion token 2 = B2 → target q'_2 → B3
```

这和你的真实 inference **完全一致**。

因此当前训练脚本里的：

```python
masked_motion = torch.zeros_like(motion_gt)
mask = torch.ones_like(motion_gt)
masked_motion[:, :seed_frames] = ...
```

可以全部删掉。现在的训练确实还是原 EMAGE 风格的 seed+mask。

---

# 第四步：你的 `CausalMotionTokenEncoder` 可以继续留

你现在：

```python
Conv1d(
    input_dim,
    hidden_dim,
    kernel_size=4,
    stride=4
)
```

作用正好是：

```text
B0 → h0
B1 → h1
B2 → h2
```

所以输入：

```python
past_motion = motion_gt[:, :-4]
```

输出正好是：

```text
15 blocks → 15 condition tokens
```

target 也是：

```text
future_motion 60 frames
→ VQ /4
→ 15 target tokens
```

长度天然匹配。

这里不需要 mask，也不需要 mask embedding。

---

# 第五步：注意你这里与 LiveGesture 有一个刻意的不同

LiveGesture 的 HAR 在 time \(t\) 使用的是：

$$
x_{t-h},...,x_{t-1}
$$

也就是**过去的 SVQ token history**，而不是 raw past motion。论文的 region-xAR 就是 past motion tokens + aligned causal audio tokens。([researchgate.net][1])

你现在设计的是：

$$
B_{t-1}\rightarrow motion\ encoder\rightarrow h^m_t
$$

即：

```text
past decoded 4-frame motion
      ↓
CausalMotionTokenEncoder
      ↓
motion condition
```

这个可以做，但它属于：

> **借鉴 LiveGesture 的 SVQ，而不是复现 LiveGesture 的 HAR。**

我认为对于你基于 EMAGE 改造的目标，这没有问题。

甚至有一个优势：因为你 inference 本来已经 decode 成连续 motion，所以直接把生成出的 4 帧反馈进去，非常直观。

---

# 第六步：后期要解决 teacher forcing gap

第一阶段训练：

```text
GT B0 → B1
GT B1 → B2
GT B2 → B3
```

推理：

```text
GT seed B0 → pred B1
pred B1 → pred B2
pred B2 → pred B3
```

存在 exposure bias。

先不要一开始处理。

### Phase A

纯 GT past motion：

$$
B^{cond}_{i-1}=B^{GT}_{i-1}
$$

先把模型训稳定。

### Phase B

再增加 rollout：

```text
GT seed B0
    ↓
predict q1
    ↓
decode_stream
    ↓
pred B1
    ↓
predict q2
...
```

可以随机用：

$$
B^{cond}_{i-1}
=
\begin{cases}
B^{GT}_{i-1}&p\\
\hat B_{i-1}&1-p
\end{cases}
$$

LiveGesture 本身也专门用了 autoregressive masking / noise 来提升对错误 motion history 的鲁棒性，因为他们同样面对 streaming prediction drift。([开放获取计算机视觉论坛][2])

你后面可以借这个思想，而不一定非要标准 scheduled sampling。

---

# 第七步：`CausalWavEncoder` 建议直接往 LiveGesture 的方案靠

这一点现在我会修改之前的建议。

LiveGesture 并不是简单地：

```text
raw waveform
→ stride CNN
→ 假装输出率等于 motion FPS
```

它的论文描述是：

```text
raw waveform
    ↓
log-mel feature
    ↓
causal 1D conv projection
    ↓
dilated causal conv pyramid
    ↓
causal audio-motion alignment
    ↓
audio token，与 motion token 同 rate
```

并明确强调 alignment module 将 audio 聚合到 motion token rate，每个 token 只依赖 past/current acoustic evidence。([Muhammad Usama Saleem][3])

这比你现在的：

```text
raw waveform
↓ stride 5
↓ stride 6
↓ stride 6
↓ stride 3

total stride = 540
```

合理得多，因为你现在实际输出：

$$
16000/540=29.63Hz
$$

并不是真正 30Hz。

---

# 第八步：audio 最好直接对齐 7.5 Hz，而不是先对齐 30 Hz

因为 Stage-2 最终根本不是每帧预测：

$$
30Hz
$$

而是：

$$
30/4=7.5Hz
$$

所以我建议：

```text
waveform
  ↓
causal log-mel
例如 100Hz
  ↓
causal dilated CNN
  ↓
high-rate audio feature
  ↓
causal audio-motion aligner
  ↓
7.5Hz
  ↓
a1 a2 a3 ...
```

然后：

```text
audio token ai
motion condition Bi-1
speaker
       ↓
Transformer
       ↓
qi
```

这样最简单。

---

# 第九步：严格 0ms 的 audio alignment 应该这么定义

目标：

```text
q'_0 → B1
```

而 B1 起点是：

$$
t_1=4/30=0.13333s
$$

所以 `audio_token[0]` 只能使用：

$$
audio\le0.13333s
$$

下一个：

$$
t_2=8/30
$$

只允许：

$$
audio\le0.26667s
$$

以此类推。

定义：

$$
t_i=\frac{4(i+1)}{30}
$$

因为你的第 0 个预测 target 是 B1。

如果使用 10ms hop 的 mel：

```text
0 ms
10 ms
20 ms
...
```

则：

```text
q'_0:
只允许 mel frame 的有效终点 <=133.33ms

q'_1:
只允许 <=266.67ms
```

这样不会有 `16000/30` 的整数 stride 问题，也永远不会 drift。

---

# 第十步：RoPE 继续可以用，但功能要分清

LiveGesture 确实在 motion token history 上使用 RoPE。补充材料写的是过去 motion token embedding 投影后加入 rotary positional embeddings，以保证 streaming shift 下的时间位置稳定。([开放获取计算机视觉论坛][4])

因此你可以把：

```python
PeriodicPositionalEncoding
```

换成：

```text
RoPE
```

我赞成。

但正确流程还是：

```text
物理 timestamp
    ↓
先把audio严格聚合成7.5Hz
    ↓
audio_i ↔ motion_i 已经对齐
    ↓
再用RoPE描述token position
```

不是：

```text
29.63Hz audio
+
7.5Hz motion
+
RoPE
→ 自动变对齐
```

RoPE做不到这件事。

---

# 所以你的新修改清单可以精简成 5 项

**VQ-VAE 不动 encoder，也不动 decoder architecture。**

1. **Stage-2 训练改成 shifted future sequence。**

```text
past:
B0 B1 ... B14

VQ target从：
B1 B2 ... B15
单独encode得到
```

而不是：

```text
完整B0...B15 encode
然后drop q0
```

这一点是为了让 VQ decoder 的 stream-start 状态和 inference 一致。

2. **删掉 `mask / mask_embedding / seed+MASK`。**

模型变成：

```python
model(audio, speaker_id, past_motion)
```

3. **改 audio encoder。**

从现在的 raw-wave stride-540：

```text
→ causal mel
→ causal conv pyramid
→ timestamp-aware causal alignment
→ 7.5Hz
```

这也更接近 LiveGesture。([Muhammad Usama Saleem][3])

4. **推理改成真正一 token 一步。**

```text
seed B0
→ q1 → decode_stream → B1
→ q2 → decode_stream → B2
...
```

VQ decoder state 从第一个生成 token开始维护，不需要 bidirectional encoder 参与 inference。

5. **最后加 RoPE + rollout robustness。**

先 teacher forcing，模型稳定以后再加入 generated-past-motion / corruption training。

---

其中我现在最建议你**马上先改的是训练数据构造**：

```python
factor = 4

past_motion = motion_gt[:, :-factor]

future_motion = motion_gt[:, factor:]
future_expr = expressions_gt[:, factor:]

target_index = motion_vq.map2index(
    future_motion,
    future_expr,
)

pred = model(
    audio,
    speaker_id,
    past_motion,
)
```

下一步真正需要仔细设计的是：**`audio` 怎样从原始 waveform 变成与这 15 个 `target_index` 严格一一对应的 15 个 causal audio tokens**。这部分我建议直接按照 LiveGesture 的“mel → causal pyramid → causal audio-motion alignment”思路重写，而不是继续修现在的 stride-540 `CausalWavEncoder`。

[1]: https://www.researchgate.net/publication/403791819_LiveGesture_Streamable_Co-Speech_Gesture_Generation_Model/download?utm_source=chatgpt.com "(PDF) LiveGesture Streamable Co-Speech Gesture Generation Model"
[2]: https://openaccess.thecvf.com/content/CVPR2026/html/Saleem_LiveGesture_Streamable_Co-Speech_Gesture_Generation_Model_CVPR_2026_paper.html?utm_source=chatgpt.com "CVPR 2026 Open Access Repository"
[3]: https://m-usamasaleem.github.io/publication/LiveGesture/LiveGen.pdf?utm_source=chatgpt.com "LiveGesture: Streamable Co-Speech Gesture Generation Model"
[4]: https://openaccess.thecvf.com/content/CVPR2026/supplemental/Saleem_LiveGesture_Streamable_Co-Speech_CVPR_2026_supplemental.pdf?utm_source=chatgpt.com "1. Supplementary Material"

修改：
基于你现在这版代码，我建议**不要推翻结构**，只做必要修改。目标是：

1. **训练：**

$$
(B_0,B_1,...,B_{N-1})+audio_{\le t}
\rightarrow
(q_1,q_2,...,q_N)
$$

2. **推理：**

$$
seed=B_0
$$

逐步：

$$
B_0+audio_{\le133ms}\rightarrow q_1\rightarrow B_1
$$

$$
B_0,B_1+audio_{\le266ms}\rightarrow q_2\rightarrow B_2
$$

严格 0ms lookahead。

---

下面给出具体修改。

---

# 1. 修正 inference 的 audio timestamp（必须改）

你现在：

```python
for t in range(1,total_blocks):

    audio_end = round(
        4*t*self.cfg.audio_sr/self.cfg.pose_fps
    )

    chunk_logits = self(
        audio[:,:audio_end],
        ...
    )
```

这里 t 的定义错位。

因为：

```text
t=1
```

表示预测：

```
B1
```

而 B1 的结束时间：

```
B1 = frames 4~7
```

结束：

$$
8/30s
$$

但是你真正需要的是：

> 预测 B1 时，audio 看到 B0+B1 的时间范围？

这里需要先明确你的训练 alignment。

---

## 推荐统一定义

你的 target：

```python
future_motion = motion_gt[:, factor:]
```

即：

```
target token 0 = B1
```

那么：

模型输入：

```
past_motion = B0
```

预测：

```
q(B1)
```

audio token 应该对应：

```
B1 时间窗口
```

所以：

token i：

对应：

```
Bi+1
```

audio end：

$$
(i+2)\times4/30
$$

但是你想要**0ms latency**，通常不是预测完整 block 后才生成，而是：

```
block起点生成
```

因此应该使用：

$$
(i+1)\times4/30
$$

也就是：

预测 B1：

看到：

```
0~133ms audio
```

预测 B2：

看到：

```
0~266ms audio
```

所以代码：

---

修改：

```python
for i in range(total_blocks-1):

    audio_end = round(
        (i+1)
        * block
        * self.cfg.audio_sr
        / self.cfg.pose_fps
    )

    chunk_logits = self(
        audio[:, :audio_end],
        speaker_id,
        past_motion
    )
```

不要使用 t。

---

# 2. 修正 CausalMelAudioEncoder 输出长度

现在：

```python
N = int(block_idx.max().item()) + 1
```

这是错误来源。

audio决定token数量。

应该：

> motion决定token数量。

改函数：

加入参数：

```python
def forward(self, wav, target_tokens=None):
```

---

然后：

替换：

```python
N = int(block_idx.max().item()) + 1
```

为：

```python
if target_tokens is not None:
    N = target_tokens
else:
    N = int(block_idx.max().item()) + 1
```

---

然后：

过滤非法frame：

加入：

```python
valid = (
    block_idx >=0
) & (
    block_idx < N
)

h_t = h_t[:,valid]
block_idx = block_idx[valid]
```

否则：

最后一个mel frame可能制造额外token。

---

# 3. CausalMelAudioEncoder调用修改

现在：

```python
audio_face = self.audio_encoder_face(audio)
```

改：

```python
token_length = body_hint.shape[1]

audio_face = self.audio_encoder_face(
    audio,
    target_tokens=token_length
)

audio_body = self.audio_encoder_body(
    audio,
    target_tokens=token_length
)
```

不要再：

```python
audio_face = audio_face[:,:token_length]
```

因为那是补救，不是严格align。

---

# 4. 加 seed motion inference

现在：

训练支持：

```python
seed_motion
```

但是测试没有传。

修改：

`train_emage_audio.py`

这里：

```python
token_logits = actual_model.inference(
    audio,
    speaker_id,
    motion_vq
)
```

改：

```python
seed_motion = motion_gt[:, :cfg.model.token_downsample_factor]
```

但是 inference_fn 没有 motion_gt。

所以需要 dataset 返回：

例如：

```python
batch["motion"]
```

测试读取：

```python
test_file["motion_path"]
```

加载前4帧。

伪代码：

```python
gt = beat_format_load(
    test_file["motion_path"],
    [True]*55
)

seed = gt["poses"][:4]
```

转换：

axis-angle:

```
4,55,3
```

↓

rot6d:

```
4,330
```

然后：

```python
seed_motion = torch.from_numpy(seed)
seed_motion = seed_motion.unsqueeze(0).to(device)
```

调用：

```python
token_logits = actual_model.inference(
    audio,
    speaker_id,
    motion_vq,
    seed_motion
)
```

---

# 5. 修正 inference 输出长度

现在：

```python
return {
 cls_face:
 torch.cat(values,dim=1)
}
```

输出：

```
B1...BN
```

没有 B0。

这是正确的。

但是后面：

```python
motion_vq.decode(
 all tokens
)
```

会少4帧。

如果评价需要完整序列：

需要：

```text
seed
+
generated
```

拼接。

修改：

```python
motion_pred = torch.cat(
[
 seed_motion,
 generated_motion
],
dim=1
)
```

---

# 6. 验证 decode_stream 正确性

增加测试：

在 `streamable_vq.py`

写：

```python
@torch.no_grad()
def test_stream_decode(self, indices):

    full = self.decode(
        **indices
    )


    state=None
    outs=[]

    T=indices["upper"].shape[1]

    for i in range(T):

        chunk={
          k:v[:,i:i+1]
          for k,v in indices.items()
        }

        out,state=self.decode_stream(
            chunk,
            state
        )

        outs.append(
            out["motion_rot6d"]
        )


    stream=torch.cat(
        outs,
        dim=1
    )


    print(
        torch.max(
          torch.abs(
             full["motion_rot6d"]-stream
          )
        )
    )
```

必须：

```
<1e-5
```

否则 decoder streaming有bug。

---

# 7. 关于 Transformer causal mask

你的：

```python
memory_mask=causal_mask
```

保留。

但是这里：

```python
face_motion_decoder
```

也需要：

```python
memory_mask=causal_mask
```

你已经加了。

正确。

---

# 8. 训练部分只需要小修改

现在：

```python
past_motion = motion_gt[:,:-factor]

future_motion = motion_gt[:,factor:]
```

保持。

但是建议加入长度检查：

```python
assert past_motion.shape[1] % factor ==0
assert future_motion.shape[1] % factor ==0
```

否则 VQ token长度可能错。

---

# 9. RoPE 暂时不要加

你现在：

```python
PeriodicPositionalEncoding
```

可以继续。

原因：

现在最大问题不是位置编码。

而是：

```
audio token
motion token
target token
```

三者时间轴。

先闭环。

---

# 最终修改列表

## 必改

### modeling_emage_audio.py

修改：

1.

```python
audio_encoder(audio,target_tokens)
```

2.

inference:

```python
audio_end=(i+1)*block*sr/fps
```

3.

加入 seed_motion 流程

---

### processing_emage_audio.py

修改：

`CausalMelAudioEncoder.forward`

增加：

```python
target_tokens
```

并过滤：

```python
block_idx>=0
block_idx<N
```

---

### train_emage_audio.py

修改：

test inference:

传：

```python
seed_motion
```

---

## 不改

* VQ encoder ✅
* VQ decoder结构 ✅
* CausalMotionTokenEncoder ✅
* shifted future training ✅
* cross attention结构 ✅

---

改完以后，你的模型才真正满足：

```
audio (causal)
+
speaker
+
past generated motion

↓

future VQ token

↓

causal VQ decoder

↓

4-frame motion block
```

这才是严格 streaming EMAGE → LiveGesture 风格的版本。你现在距离这个版本主要差的是**时间轴闭环**，不是模型结构。


**针对“历史窗口滑动后，音频还停在开头”这个错误，只需修改推理相关逻辑，不需要重新训练。** 但它位于 `modeling_emage_audio(4).py`，不是外层的 `train_emage_audio(3).py`。

建议先做下面这个最小修复，用现有 checkpoint 验证。训练依赖真实动作历史导致的退化，可能仍然存在。

**1. 给 `forward()` 增加音频窗口起点**

找到 `CausalEmageAudioTokenModel.forward()`，将开头改成下面这样；从 `causal_mask` 开始，后面的代码保持原样：

```python
def forward(self, audio, speaker_id, past_motion, audio_token_start=0):
    body_hint = self.motion_encoder(past_motion)
    body_hint_body = self.bodyhints_body(body_hint)
    body_hint_face = self.bodyhints_face(body_hint)
    token_length = body_hint.shape[1]

    # past_motion 对应全局 token 区间 [start, end)
    start = int(audio_token_start)
    end = start + token_length
    if start < 0:
        raise ValueError("audio_token_start must be non-negative")

    # 先从完整音频前缀提取特征，保留原有 STFT 时间网格和卷积上下文。
    audio_face_all = self.audio_encoder_face(
        audio, target_tokens=end
    )
    audio_body_all = self.audio_encoder_body(
        audio, target_tokens=end
    )

    # 再选择与动作历史对应的音频 token。
    if (
        audio_face_all.shape[1] < end
        or audio_body_all.shape[1] < end
    ):
        raise RuntimeError(
            f"Audio tokens insufficient: need {end}, "
            f"face={audio_face_all.shape[1]}, "
            f"body={audio_body_all.shape[1]}"
        )

    audio_face = audio_face_all[:, start:end]
    audio_body = audio_body_all[:, start:end]

    causal_mask = self._causal_mask(token_length, audio.device)

    # 以下继续使用原来的 speaker、attention 和分类头代码。
```

这里新增的是普通参数，没有增加模型权重，**旧 checkpoint 可以继续加载**。训练调用不传这个参数时，默认 `start=0`，保持原来的行为。

**2. 在 `inference()` 中计算当前窗口的全局位置**

找到循环里的：

```python
audio_end = round(
    (i + 1) * block * self.cfg.audio_sr / self.cfg.pose_fps
)
chunk_logits = self(audio[:, :audio_end], speaker_id, past_motion)
```

替换为：

```python
# 当前已有动作 B0 ... Bi，即全局 i + 1 个历史块。
history_tokens = past_motion.shape[1] // block
audio_token_end = i + 1
audio_token_start = audio_token_end - history_tokens

audio_end = round(
    audio_token_end * block
    * self.cfg.audio_sr / self.cfg.pose_fps
)

chunk_logits = self(
    audio[:, :audio_end],
    speaker_id,
    past_motion,
    audio_token_start=audio_token_start,
)
```

其余的 `argmax → decode_stream → 回填动作 → 截断历史` 暂时保持原样。

这个计算沿用你当前的约定：**seed 恰好是一个 block，也就是 4 帧**。可以在初始化 `past_motion` 后加一个检查：

```python
if past_motion.shape[1] != block:
    raise ValueError(
        f"Expected a {block}-frame seed, "
        f"got {past_motion.shape[1]} frames"
    )
```

修复后，窗口对应关系如下：

| 当前历史范围   | 动作条件     | 音频条件     |
| -------- | -------- | -------- |
| `B0…B31` | 第 0～31 块 | 第 0～31 块 |
| `B1…B32` | 第 1～32 块 | 第 1～32 块 |
| `B2…B33` | 第 2～33 块 | 第 2～33 块 |

**关键是先编码完整音频前缀，再截取特征窗口。** 这样无需立即改造音频编码器的缓存，也避免直接裁剪原始音频带来的卷积上下文丢失和 STFT 网格变化。代价是仍然会重复编码越来越长的音频，适合作为第一步正确性修复。

**3. 先用原权重验证，不要同时改其他行为**

先保持 `argmax`、history window 和音频时间戳实现不变，重跑同一段音频。重点看：

* 超过约 4.27 秒后，动作是否继续随音频变化。
* 上身、手、下身是否仍长时间重复同一个 token。
* 输出是否仍然出现严格的 `motion[t+4] == motion[t]`。

**如果修复后仍然很快进入循环，就不能只靠改推理解决了**，下一步要处理训练只使用真实历史、推理使用生成历史的差异。现在先改上面两处，可以直接检验这个明确的音频错位错误造成了多大影响。
