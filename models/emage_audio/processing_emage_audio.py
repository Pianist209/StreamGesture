import math
import torch
import torch.nn as nn
import torch.nn.functional as F

def _copysign(a, b):
    signs_differ = (a < 0) != (b < 0)
    return torch.where(signs_differ, -a, a)

def _sqrt_positive_part(x):
    ret = torch.zeros_like(x)
    positive_mask = x > 0
    ret[positive_mask] = torch.sqrt(x[positive_mask])
    return ret

def matrix_to_quaternion(matrix):
    if matrix.size(-1) != 3 or matrix.size(-2) != 3:
        raise ValueError
    m00 = matrix[..., 0, 0]
    m11 = matrix[..., 1, 1]
    m22 = matrix[..., 2, 2]
    o0 = 0.5 * _sqrt_positive_part(1 + m00 + m11 + m22)
    x = 0.5 * _sqrt_positive_part(1 + m00 - m11 - m22)
    y = 0.5 * _sqrt_positive_part(1 - m00 + m11 - m22)
    z = 0.5 * _sqrt_positive_part(1 - m00 - m11 + m22)
    o1 = _copysign(x, matrix[..., 2, 1] - matrix[..., 1, 2])
    o2 = _copysign(y, matrix[..., 0, 2] - matrix[..., 2, 0])
    o3 = _copysign(z, matrix[..., 1, 0] - matrix[..., 0, 1])
    return torch.stack((o0, o1, o2, o3), -1)

def quaternion_to_axis_angle(quaternions):
    norms = torch.norm(quaternions[..., 1:], p=2, dim=-1, keepdim=True)
    half_angles = torch.atan2(norms, quaternions[..., :1])
    angles = 2 * half_angles
    eps = 1e-6
    small_angles = angles.abs() < eps
    sin_half_angles_over_angles = torch.empty_like(angles)
    sin_half_angles_over_angles[~small_angles] = (
        torch.sin(half_angles[~small_angles]) / angles[~small_angles]
    )
    sin_half_angles_over_angles[small_angles] = (
        0.5 - (angles[small_angles]*angles[small_angles])/48
    )
    return quaternions[..., 1:] / sin_half_angles_over_angles

def matrix_to_axis_angle(matrix):
    return quaternion_to_axis_angle(matrix_to_quaternion(matrix))

def rotation_6d_to_matrix(d6: torch.Tensor) -> torch.Tensor:
    a1, a2 = d6[..., :3], d6[..., 3:]
    b1 = F.normalize(a1, dim=-1)
    b2 = a2 - (b1 * a2).sum(-1, keepdim=True) * b1
    b2 = F.normalize(b2, dim=-1)
    b3 = torch.cross(b1, b2, dim=-1)
    return torch.stack((b1, b2, b3), dim=-2)

def rotation_6d_to_axis_angle(rot6d):
    return matrix_to_axis_angle(rotation_6d_to_matrix(rot6d))

def matrix_to_rotation_6d(matrix: torch.Tensor) -> torch.Tensor:
    return matrix[..., :2, :].clone().reshape(*matrix.size()[:-2], 6)

def axis_angle_to_quaternion(axis_angle):
    angles = torch.norm(axis_angle, p=2, dim=-1, keepdim=True)
    half_angles = 0.5 * angles
    eps = 1e-6
    small_angles = angles.abs() < eps
    sin_half_angles_over_angles = torch.empty_like(angles)
    sin_half_angles_over_angles[~small_angles] = (
        torch.sin(half_angles[~small_angles]) / angles[~small_angles]
    )
    sin_half_angles_over_angles[small_angles] = (
        0.5 - (angles[small_angles] * angles[small_angles]) / 48
    )
    quaternions = torch.cat(
        [torch.cos(half_angles), axis_angle * sin_half_angles_over_angles], dim=-1
    )
    return quaternions

def quaternion_to_matrix(quaternions):
    r, i, j, k = torch.unbind(quaternions, -1)
    two_s = 2.0 / (quaternions * quaternions).sum(-1)

    o = torch.stack(
        (
            1 - two_s * (j * j + k * k),
            two_s * (i * j - k * r),
            two_s * (i * k + j * r),
            two_s * (i * j + k * r),
            1 - two_s * (i * i + k * k),
            two_s * (j * k - i * r),
            two_s * (i * k - j * r),
            two_s * (j * k + i * r),
            1 - two_s * (i * i + j * j),
        ),
        -1,
    )
    return o.reshape(quaternions.shape[:-1] + (3, 3))

def axis_angle_to_matrix(axis_angle):
    return quaternion_to_matrix(axis_angle_to_quaternion(axis_angle))

def axis_angle_to_rotation_6d(axis_angle):
    return matrix_to_rotation_6d(axis_angle_to_matrix(axis_angle))


def velocity2position(data_seq, dt, init_pos):
    res_trans = []
    for i in range(data_seq.shape[1]):
        if i == 0:
            res_trans.append(init_pos.unsqueeze(1))
        else:
            res = data_seq[:, i-1:i] * dt + res_trans[-1]
            res_trans.append(res)
    return torch.cat(res_trans, dim=1)


def recover_from_mask_ts(selected_motion: torch.Tensor, mask: list) -> torch.Tensor:
    device = selected_motion.device
    dtype = selected_motion.dtype
    mask_arr = torch.tensor(mask, dtype=torch.bool, device=device)
    j = len(mask_arr)
    sum_mask = mask_arr.sum().item()
    c_channels = selected_motion.shape[-1] // sum_mask
    new_shape = selected_motion.shape[:-1] + (sum_mask, c_channels)
    selected_motion = selected_motion.reshape(new_shape)
    out_shape = list(selected_motion.shape[:-2]) + [j, c_channels]
    recovered = torch.zeros(out_shape, dtype=dtype, device=device)
    recovered[..., mask_arr, :] = selected_motion
    final_shape = list(recovered.shape[:-2]) + [j*c_channels]
    recovered = recovered.reshape(final_shape)
    return recovered


class Quantizer(nn.Module):
    def __init__(self, n_e, e_dim, beta):
        super().__init__()
        self.e_dim = e_dim
        self.n_e = n_e
        self.beta = beta
        self.embedding = nn.Embedding(self.n_e, self.e_dim)
        self.embedding.weight.data.uniform_(-1.0/self.n_e, 1.0/self.n_e)

    def forward(self, z):
        assert z.shape[-1] == self.e_dim
        z_flattened = z.contiguous().view(-1, self.e_dim)
        d = torch.sum(z_flattened**2, dim=1, keepdim=True) + \
            torch.sum(self.embedding.weight**2, dim=1) - 2*torch.matmul(z_flattened, self.embedding.weight.t())
        min_encoding_indices = torch.argmin(d, dim=1)
        z_q = self.embedding(min_encoding_indices).view(z.shape)
        loss = torch.mean((z_q - z.detach())**2) + self.beta*torch.mean((z_q.detach() - z)**2)
        z_q = z + (z_q - z).detach()
        min_encodings = F.one_hot(min_encoding_indices, self.n_e).type(z.dtype)
        e_mean = torch.mean(min_encodings, dim=0)
        perplexity = torch.exp(-torch.sum(e_mean*torch.log(e_mean+1e-10)))
        return loss, z_q, min_encoding_indices, perplexity

    def map2index(self, z):
        assert z.shape[-1] == self.e_dim
        z_flattened = z.contiguous().view(-1, self.e_dim)
        d = torch.sum(z_flattened**2, dim=1, keepdim=True) + \
            torch.sum(self.embedding.weight**2, dim=1) - 2*torch.matmul(z_flattened, self.embedding.weight.t())
        min_encoding_indices = torch.argmin(d, dim=1)
        return min_encoding_indices.reshape(z.shape[0], -1)

    def get_codebook_entry(self, indices):
        index_flattened = indices.view(-1)
        z_q = self.embedding(index_flattened)
        z_q = z_q.view(indices.shape+(self.e_dim,)).contiguous()
        return z_q

def init_weight(m):
    if isinstance(m, nn.Conv1d) or isinstance(m, nn.Linear) or isinstance(m, nn.ConvTranspose1d):
        nn.init.xavier_normal_(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)

class ResBlock(nn.Module):
    def __init__(self, channel):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv1d(channel, channel, 3, 1, 1),
            nn.LeakyReLU(0.2, True),
            nn.Conv1d(channel, channel, 3, 1, 1),
        )
    def forward(self, x):
        return self.model(x)+x

class VQEncoderV5(nn.Module):
    def __init__(self, args):
        super().__init__()
        n_down = args.vae_layer
        channels = [args.vae_length]*(n_down)
        input_size = args.vae_test_dim
        layers = [
            nn.Conv1d(input_size, channels[0], 3, 1, 1),
            nn.LeakyReLU(0.2, True),
            ResBlock(channels[0]),
        ]
        for i in range(1, n_down):
            layers += [
                nn.Conv1d(channels[i-1], channels[i], 3, 1, 1),
                nn.LeakyReLU(0.2, True),
                ResBlock(channels[i]),
            ]
        self.main = nn.Sequential(*layers)
        self.main.apply(init_weight)
    def forward(self, inputs):
        inputs = inputs.permute(0,2,1)
        outputs = self.main(inputs).permute(0,2,1)
        return outputs

class VQEncoderV6(nn.Module):
    def __init__(self, args):
        super().__init__()
        n_down = args.vae_layer
        channels = [args.vae_length]*(n_down)
        input_size = args.vae_test_dim
        layers = [
            nn.Conv1d(input_size, channels[0], 3, 1, 1),
            nn.LeakyReLU(0.2, True),
            ResBlock(channels[0]),
        ]
        for i in range(1, n_down):
            layers += [
                nn.Conv1d(channels[i-1], channels[i], 3, 1, 1),
                nn.LeakyReLU(0.2, True),
                ResBlock(channels[i]),
            ]
        self.main = nn.Sequential(*layers)
        self.main.apply(init_weight)
    def forward(self, inputs):
        inputs = inputs.permute(0,2,1)
        outputs = self.main(inputs).permute(0,2,1)
        return outputs

class VQDecoderV5(nn.Module):
    def __init__(self, args):
        super().__init__()
        n_up = args.vae_layer
        channels = [args.vae_length]*(n_up)+[args.vae_test_dim]
        input_size = args.vae_length
        n_resblk = 2
        if input_size == channels[0]:
            layers = []
        else:
            layers = [nn.Conv1d(input_size, channels[0], 3, 1, 1)]
        for i in range(n_resblk):
            layers += [ResBlock(channels[0])]
        for i in range(n_up):
            layers += [
                nn.Conv1d(channels[i], channels[i+1], 3, 1, 1),
                nn.LeakyReLU(0.2, True)
            ]
        layers += [nn.Conv1d(channels[-1], channels[-1], 3, 1, 1)]
        self.main = nn.Sequential(*layers)
        self.main.apply(init_weight)
    def forward(self, inputs):
        inputs = inputs.permute(0,2,1)
        outputs = self.main(inputs).permute(0,2,1)
        return outputs

class BasicBlock(nn.Module):
    """ based on timm: https://github.com/rwightman/pytorch-image-models """
    def __init__(self, inplanes, planes, ker_size, stride=1, downsample=None, dilation=1, first_dilation=None, act_layer=nn.LeakyReLU, norm_layer=nn.BatchNorm1d):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv1d(
            inplanes, planes, kernel_size=ker_size, stride=stride, padding=first_dilation,
            dilation=dilation, bias=True)
        self.bn1 = norm_layer(planes)
        self.act1 = act_layer(inplace=True)
        self.conv2 = nn.Conv1d(
            planes, planes, kernel_size=ker_size, padding=ker_size//2, dilation=dilation, bias=True)
        self.bn2 = norm_layer(planes)
        self.act2 = act_layer(inplace=True)
        if downsample is not None:
            self.downsample = nn.Sequential(
                nn.Conv1d(inplanes, planes,  stride=stride, kernel_size=ker_size, padding=first_dilation, dilation=dilation, bias=True),
                norm_layer(planes), 
            )
        else: self.downsample=None

    def forward(self, x):
        shortcut = x
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        if self.downsample is not None:
            shortcut = self.downsample(shortcut)
        x += shortcut
        x = self.act2(x)
        return x

class WavEncoder(nn.Module):
    def __init__(self, out_dim, audio_in=1):
        super().__init__() 
        self.out_dim = out_dim
        self.feat_extractor = nn.Sequential( 
                BasicBlock(audio_in, out_dim//4, 15, 5, first_dilation=1600, downsample=True),
                BasicBlock(out_dim//4, out_dim//4, 15, 6, first_dilation=0, downsample=True),
                BasicBlock(out_dim//4, out_dim//4, 15, 1, first_dilation=7, ),
                BasicBlock(out_dim//4, out_dim//2, 15, 6, first_dilation=0, downsample=True),
                BasicBlock(out_dim//2, out_dim//2, 15, 1, first_dilation=7),
                BasicBlock(out_dim//2, out_dim, 15, 3,  first_dilation=0,downsample=True),     
            )
    def forward(self, wav_data):
        if wav_data.dim() == 2:
            wav_data = wav_data.unsqueeze(1) 
        else:
            wav_data = wav_data.transpose(1, 2)
        out = self.feat_extractor(wav_data)
        return out.transpose(1, 2)


class CausalBasicBlock(nn.Module):
    """Residual temporal block whose output at t only uses samples up to t."""

    def __init__(self, inplanes, planes, kernel_size, stride=1):
        super().__init__()
        self.kernel_size = kernel_size
        self.conv1 = nn.Conv1d(inplanes, planes, kernel_size, stride=stride)
        self.conv2 = nn.Conv1d(planes, planes, kernel_size)
        self.norm1 = nn.LayerNorm(planes)
        self.norm2 = nn.LayerNorm(planes)
        self.act = nn.LeakyReLU(inplace=True)
        self.downsample = nn.Conv1d(inplanes, planes, kernel_size=1, stride=stride) if stride != 1 or inplanes != planes else None

    @staticmethod
    def _normalize(x, norm):
        return norm(x.transpose(1, 2)).transpose(1, 2)

    def forward(self, x):
        shortcut = x if self.downsample is None else self.downsample(x)
        x = self.conv1(F.pad(x, (self.kernel_size - 1, 0)))
        x = self.act(self._normalize(x, self.norm1))
        x = self.conv2(F.pad(x, (self.kernel_size - 1, 0)))
        x = self._normalize(x, self.norm2)
        return self.act(x + shortcut)


class CausalWavEncoder(nn.Module):
    """30 FPS waveform encoder with no future-audio dependency."""

    def __init__(self, out_dim, audio_in=1):
        super().__init__()
        self.feat_extractor = nn.Sequential(
            CausalBasicBlock(audio_in, out_dim // 4, 15, 5),
            CausalBasicBlock(out_dim // 4, out_dim // 4, 15, 6),
            CausalBasicBlock(out_dim // 4, out_dim // 4, 15, 1),
            CausalBasicBlock(out_dim // 4, out_dim // 2, 15, 6),
            CausalBasicBlock(out_dim // 2, out_dim // 2, 15, 1),
            CausalBasicBlock(out_dim // 2, out_dim, 15, 3),
        )

    def forward(self, wav_data):
        if wav_data.dim() == 2:
            wav_data = wav_data.unsqueeze(1)
        else:
            wav_data = wav_data.transpose(1, 2)
        return self.feat_extractor(wav_data).transpose(1, 2)

def _hann_window(n_fft):
    return 0.5 - 0.5 * torch.cos(2 * math.pi * torch.arange(n_fft).float() / (n_fft - 1))


def _mel_filters(sr, n_fft, n_mels, fmin=0.0, fmax=None):
    """Slaney-free triangular mel filterbank in pure torch (matches librosa)."""
    fmax = fmax if fmax is not None else sr / 2.0
    n_freqs = n_fft // 2 + 1
    freqs = torch.linspace(0.0, float(sr / 2.0), n_freqs)

    def hz_to_mel(f):
        return 2595.0 * torch.log10(f + 700.0) - 2595.0 * math.log10(700.0)

    def mel_to_hz(m):
        return 700.0 * (10.0 ** (m / 2595.0) - 1.0)

    mel_min, mel_max = hz_to_mel(torch.as_tensor(fmin, dtype=torch.float64)), hz_to_mel(torch.as_tensor(fmax, dtype=torch.float64))
    mels = torch.linspace(mel_min.item(), mel_max.item(), n_mels + 2)
    mel_freqs = mel_to_hz(mels)
    filters = torch.zeros(n_mels, n_freqs)
    for i in range(n_mels):
        f_lo, f_c, f_hi = mel_freqs[i], mel_freqs[i + 1], mel_freqs[i + 2]
        idx = (freqs >= f_lo) & (freqs < f_c)
        filters[i, idx] = (freqs[idx] - f_lo) / (f_c - f_lo if f_c != f_lo else 1.0)
        idx = (freqs >= f_c) & (freqs <= f_hi)
        filters[i, idx] = (f_hi - freqs[idx]) / (f_hi - f_c if f_hi != f_c else 1.0)
    return filters


class CausalLogMel(nn.Module):
    """Causal log-mel spectrogram.

    Uses `center=False` STFT with NO padding: STFT frame k covers samples
    [k*hop, k*hop+n_fft), so it only depends on audio already received.
    """

    def __init__(self, sr=16000, n_fft=400, hop_length=160, n_mels=80, eps=1e-6):
        super().__init__()
        self.sr = sr
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels
        self.eps = eps
        self.register_buffer("window", _hann_window(n_fft).float())
        self.register_buffer("mel_basis", _mel_filters(sr, n_fft, n_mels).float())

    def forward(self, wav):
        # wav: (bs, T) mono
        spec = torch.stft(
            wav,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.n_fft,
            window=self.window,
            center=False,
            pad_mode="constant",
            return_complex=True,
        )
        power = spec.abs() ** 2
        mel = torch.matmul(self.mel_basis, power)  # (bs, n_mels, T)
        return torch.log(mel.clamp_min(self.eps) + 1.0)


class CausalMelAudioEncoder(nn.Module):
    """causal log-mel -> causal conv pyramid -> timestamp-aware 7.5Hz aligner.

    Motion token rate: one audio token per `token_downsample_factor` motion
    frames (7.5Hz at 30fps / 4x). Each audio token i pools only the mel frames
    whose *window end* is <= the physical end time of motion block i
    (4*(i+1)/30 s). This gives strict 0ms lookahead and never drifts because
    membership is decided on exact sample timestamps, not integer frame counts.
    """

    def __init__(self, out_dim, sr=16000, n_mels=80, n_fft=400, hop_length=160,
                 hidden_f=256, token_downsample_factor=4, pose_fps=30, n_pyramid=3):
        super().__init__()
        self.sr = sr
        self.hop_length = hop_length
        self.n_fft = n_fft
        self.token_downsample_factor = token_downsample_factor
        self.pose_fps = pose_fps
        self.hidden_f = hidden_f
        self.mel = CausalLogMel(sr, n_fft, hop_length, n_mels)
        self.proj = nn.Conv1d(n_mels, hidden_f, kernel_size=3, padding=0)
        self.pyramid = nn.Sequential(
            *[CausalBasicBlock(hidden_f, hidden_f, 5, 1) for _ in range(n_pyramid)]
        )
        self.post = nn.Linear(hidden_f, out_dim)

    def forward(self, wav, target_tokens=None):
        # wav: (bs, T_audio) or (bs, 1, T_audio)
        if wav.dim() == 3:
            wav = wav[:, 0] if wav.shape[1] == 1 else wav.mean(dim=1)
        logmel = self.mel(wav)                          # (bs, n_mels, T)
        h = self.proj(F.pad(logmel, (2, 0)))            # causal pad kernel-1=2
        h = self.pyramid(h)                             # (bs, hidden_f, T)
        bs_, _, T = h.shape
        if T == 0:
            return torch.empty(bs_, 0, self.post.out_features, device=h.device)

        # exact integer timestamp of each mel frame's window end (in samples)
        frame_ends = (torch.arange(1, T + 1, device=h.device) * self.hop_length
                      + self.n_fft)                                    # (T,)
        # block end boundary for block i is 4*(i+1)/30 s; membership is
        # decided by strict <= via exact integer arithmetic (no float drift).
        block_denom = 4 * self.sr
        block_idx = ((frame_ends * self.pose_fps + block_denom - 1) // block_denom - 1).long()  # (T,)
        # The number of audio tokens is driven by the motion side (target_tokens),
        # not by how much audio was passed: keep only mel frames that fall inside
        # a valid token slot and drop any that would create an extra trailing token.
        if target_tokens is not None and target_tokens <= block_idx.max().item():
            N = int(target_tokens)
        else:
            N = int(block_idx.max().item()) + 1
        valid = (block_idx >= 0) & (block_idx < N)
        block_idx = block_idx[valid]
        h_t = h.transpose(1, 2)[:, valid]               # (bs, T_valid, hidden_f)
        T_valid = int(valid.sum().item())
        summed = h_t.new_zeros(bs_, N, self.hidden_f).scatter_add_(
            1, block_idx.view(1, T_valid, 1).expand(bs_, T_valid, self.hidden_f), h_t)
        counts = h_t.new_zeros(N).scatter_add(0, block_idx, h_t.new_ones(T_valid)).clamp_min(1)  # (N,)
        pooled = summed / counts.view(1, N, 1)          # (bs, N, hidden_f)
        return self.post(pooled)                        # (bs, N, out_dim)


class MLP(nn.Module):
    def __init__(self, in_dim, middle_dim, out_dim):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, middle_dim)
        self.fc2 = nn.Linear(middle_dim, out_dim)
        self.act = nn.LeakyReLU(0.1, True)
    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return x

class PeriodicPositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, period=15, max_seq_len=60): 
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(period, d_model)
        position = torch.arange(0, period, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float()*(-math.log(10000.0)/d_model))
        pe[:,0::2] = torch.sin(position*div_term)
        pe[:,1::2] = torch.cos(position*div_term)
        pe = pe.unsqueeze(0)
        repeat_num = (max_seq_len//period)+1
        pe = pe.repeat(1, repeat_num, 1)
        self.register_buffer('pe', pe)
    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)
