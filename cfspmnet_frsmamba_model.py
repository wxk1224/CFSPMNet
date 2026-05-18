
import math

import torch
import torch.nn.functional as F
from einops import einsum, rearrange, repeat
from einops.layers.torch import Rearrange
from torch import Tensor, nn

class DepthwiseSeparableTokenFusionConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding):
        super().__init__()
        self.depthwise = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size=kernel_size,
            groups=in_channels,
            padding=padding,
            bias=False,
        )
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=(1, 1), bias=False)

    def forward(self, x):
        return self.pointwise(self.depthwise(x))

class MultiScalePhysiologicalTokenizer(nn.Module):

    def __init__(
        self,
        f1=10,
        kernel_size=64,
        D=3,
        pooling_size1=8,
        pooling_size2=8,
        dropout_rate=0.3,
        number_channel=22,
        emb_size=40,
        temporal_kernel_sizes=(36, 24, 18),
        fusion_kernel_size=16,
    ):
        super().__init__()
        del kernel_size

        if len(temporal_kernel_sizes) != 3:
            raise ValueError("temporal_kernel_sizes must contain exactly 3 kernel sizes.")

        temporal_channels = f1 * len(temporal_kernel_sizes)
        fusion_channels = temporal_channels * D

        self.temporal_branches = nn.ModuleList(
            [nn.Conv2d(1, f1, kernel_size=(1, kernel), padding="same", bias=False) for kernel in temporal_kernel_sizes]
        )
        self.activation = nn.ELU()
        self.temporal_bn = nn.BatchNorm2d(temporal_channels)

        self.spatial_conv = nn.Conv2d(
            temporal_channels,
            fusion_channels,
            kernel_size=(number_channel, 1),
            groups=temporal_channels,
            padding="valid",
            bias=False,
        )
        self.spatial_bn = nn.BatchNorm2d(fusion_channels)
        self.pool1 = nn.MaxPool2d(kernel_size=(1, pooling_size1), stride=(1, pooling_size1))
        self.dropout1 = nn.Dropout(dropout_rate)

        self.fusion_conv = DepthwiseSeparableTokenFusionConv(
            fusion_channels,
            fusion_channels,
            kernel_size=(1, fusion_kernel_size),
            padding="same",
        )
        self.fusion_bn = nn.BatchNorm2d(fusion_channels)
        self.pool2 = nn.MaxPool2d(kernel_size=(1, pooling_size2), stride=(1, pooling_size2))
        self.dropout2 = nn.Dropout(dropout_rate)

        self.sequence_projection = Rearrange("b e h w -> b (h w) e")
        self.embedding_projection = (
            nn.Identity() if fusion_channels == emb_size else nn.Linear(fusion_channels, emb_size)
        )

    def forward(self, x: Tensor) -> Tensor:
        x = torch.cat([branch(x) for branch in self.temporal_branches], dim=1)
        x = self.temporal_bn(self.activation(x))

        x = self.spatial_conv(x) # depthwise spatial conv 
        x = self.spatial_bn(x)  # ELU + BN
        x = self.activation(x)  
        x = self.pool1(x) # MaxPooling(1x8)
        x = self.dropout1(x) # Dropout

        x = self.fusion_conv(x)
        x = self.fusion_bn(x)
        x = self.activation(x)
        x = self.pool2(x)
        x = self.dropout2(x)

        x = self.sequence_projection(x)
        x = self.embedding_projection(x)
        return x

class LearnablePositionEncoding(nn.Module):
    def __init__(self, embedding, length=100, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.encoding = nn.Parameter(torch.randn(1, length, embedding))

    def forward(self, x):
        x = x + self.encoding[:, : x.shape[1], :].to(x.device)
        return self.dropout(x)

def _init_avg_lowpass(conv: nn.Conv1d):
    kernel_size = conv.kernel_size[0]
    with torch.no_grad():
        conv.weight.zero_()
        conv.weight[:, :, :] = 1.0 / kernel_size


class RhythmicTokenStateDecomposition(nn.Module):
    def __init__(self, emb_size, low_kernel=9):
        super().__init__()
        self.low_pass = nn.Conv1d(
            emb_size,
            emb_size,
            kernel_size=low_kernel,
            padding=low_kernel // 2,
            groups=emb_size,
            bias=False,
        )
        _init_avg_lowpass(self.low_pass)

    def forward(self, x):
        x_seq = x.transpose(1, 2)
        low = self.low_pass(x_seq)
        high = x_seq - low
        return low, high


class TimeDomainRhythmicStateModeling(nn.Module):

    def __init__(self, emb_size, low_kernels=(5, 9, 13), high_kernels=(3, 5, 7), rhythm_branch_mode="full"):
        super().__init__()
        self.rhythm_branch_mode = rhythm_branch_mode
        self.decompose = RhythmicTokenStateDecomposition(emb_size, low_kernel=9)
        self.low_convs = nn.ModuleList(
            [
                nn.Conv1d(emb_size, emb_size, kernel_size=k, padding=k // 2, groups=emb_size, bias=False)
                for k in low_kernels
            ]
        )
        self.high_convs = nn.ModuleList(
            [
                nn.Conv1d(emb_size, emb_size, kernel_size=k, padding=k // 2, groups=emb_size, bias=False)
                for k in high_kernels
            ]
        )
        self.low_fuse = nn.Sequential(
            nn.Conv1d(emb_size * len(low_kernels), emb_size, kernel_size=1, bias=False),
            nn.BatchNorm1d(emb_size),
            nn.SiLU(),
        )
        self.high_fuse = nn.Sequential(
            nn.Conv1d(emb_size * len(high_kernels), emb_size, kernel_size=1, bias=False),
            nn.BatchNorm1d(emb_size),
            nn.SiLU(),
        )
        self.context_proj = nn.Sequential(
            nn.Conv1d(emb_size * 2, emb_size * 2, kernel_size=1, bias=False),
            nn.BatchNorm1d(emb_size * 2),
            nn.SiLU(),
            nn.Conv1d(emb_size * 2, emb_size, kernel_size=1, bias=False),
        )
        self.context_norm = nn.LayerNorm(emb_size)

    def forward(self, x):
        low, high = self.decompose(x)
        low_ms = torch.cat([conv(low) for conv in self.low_convs], dim=1)
        high_ms = torch.cat([conv(high) for conv in self.high_convs], dim=1)
        low_fused = self.low_fuse(low_ms) + low
        high_fused = self.high_fuse(high_ms) + high

        if self.rhythm_branch_mode == "low_only":
            low_used = low_fused
            high_used = torch.zeros_like(high_fused)
            residual = low_fused
        elif self.rhythm_branch_mode == "high_only":
            low_used = torch.zeros_like(low_fused)
            high_used = high_fused
            residual = high_fused
        else:
            low_used = low_fused
            high_used = high_fused
            residual = 0.5 * (low_fused + high_fused)

        context = self.context_proj(torch.cat([low_used, high_used], dim=1))
        context = context + residual
        context = self.context_norm(context.transpose(1, 2))
        return low_fused.transpose(1, 2), high_fused.transpose(1, 2), context


class FourierRhythmicStateModeling(nn.Module):
    def __init__(
        self,
        emb_size,
        low_kernels=(5, 9, 13),
        high_kernels=(3, 5, 7),
        rhythm_branch_mode="full",
        num_blocks=None,
        sparsity_threshold=0.01,
        rhythm_low_ratio=0.5,
    ):
        super().__init__()
        del low_kernels, high_kernels
        self.rhythm_branch_mode = rhythm_branch_mode
        self.emb_size = emb_size
        self.sparsity_threshold = sparsity_threshold
        self.rhythm_low_ratio = rhythm_low_ratio
        self.num_blocks = self._resolve_num_blocks(emb_size, preferred=num_blocks)
        self.block_size = emb_size // self.num_blocks
        self.scale = 0.02

        self.w = nn.Parameter(self.scale * torch.randn(self.num_blocks, self.block_size, self.block_size, 2))
        self.w1 = nn.Parameter(self.scale * torch.randn(2, self.num_blocks, self.block_size, 1))
        self.w2 = nn.Parameter(self.scale * torch.randn(2, self.num_blocks, self.block_size, 1))
        self.b = nn.Parameter(self.scale * torch.randn(2, self.num_blocks, self.block_size))
        self.rhythm_filter = nn.Parameter(torch.ones(1, emb_size, 1))
        self.context_proj = nn.Sequential(
            nn.Conv1d(emb_size * 2, emb_size * 2, kernel_size=1, bias=False),
            nn.BatchNorm1d(emb_size * 2),
            nn.SiLU(),
            nn.Conv1d(emb_size * 2, emb_size, kernel_size=1, bias=False),
        )
        self.context_norm = nn.LayerNorm(emb_size)

    @staticmethod
    def _resolve_num_blocks(channels: int, preferred=None) -> int:
        if preferred is not None:
            candidates = [preferred]
        else:
            candidates = [8, 6, 5, 4, 3, 2, 1]
        for candidate in candidates:
            if channels % candidate == 0:
                return candidate
        return 1

    def _fourier_mix(self, x_seq: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        dtype = x_seq.dtype
        x_spectrum = torch.fft.rfft(x_seq.float(), dim=-1, norm="ortho")
        origin_spectrum = x_spectrum
        batch_size, channels, spectrum_bins = x_spectrum.shape
        x_spectrum = x_spectrum.reshape(batch_size, self.num_blocks, self.block_size, spectrum_bins)

        weight = torch.view_as_complex(self.w.contiguous())
        mixed = torch.einsum("bkif,kio->bkof", x_spectrum, weight)
        o_real = F.silu(
            mixed.real * self.w1[0].unsqueeze(0)
            - mixed.imag * self.w1[1].unsqueeze(0)
            + self.b[0, :, :, None]
        )
        o_imag = F.silu(
            mixed.imag * self.w2[0].unsqueeze(0)
            + mixed.real * self.w2[1].unsqueeze(0)
            + self.b[1, :, :, None]
        )
        mixed = torch.stack([o_real, o_imag], dim=-1)
        mixed = F.softshrink(mixed, lambd=self.sparsity_threshold)
        mixed = torch.view_as_complex(mixed).reshape(batch_size, channels, spectrum_bins)
        mixed = mixed * self.rhythm_filter + origin_spectrum
        enhanced = torch.fft.irfft(mixed, n=x_seq.shape[-1], dim=-1, norm="ortho").type(dtype)
        return enhanced + x_seq, mixed

    def _split_low_high(self, mixed_spectrum: torch.Tensor, sequence_length: int) -> tuple[torch.Tensor, torch.Tensor]:
        spectrum_bins = mixed_spectrum.shape[-1]
        low_bins = max(1, min(spectrum_bins - 1, int(round(spectrum_bins * self.rhythm_low_ratio))))
        low_mask = torch.zeros_like(mixed_spectrum)
        low_mask[..., :low_bins] = 1.0
        high_mask = 1.0 - low_mask
        low = torch.fft.irfft(mixed_spectrum * low_mask, n=sequence_length, dim=-1, norm="ortho")
        high = torch.fft.irfft(mixed_spectrum * high_mask, n=sequence_length, dim=-1, norm="ortho")
        return low, high

    def forward(self, x):
        x_seq = x.transpose(1, 2)
        enhanced, mixed_spectrum = self._fourier_mix(x_seq)
        low_fused, high_fused = self._split_low_high(mixed_spectrum, x_seq.shape[-1])
        low_fused = low_fused + x_seq
        high_fused = high_fused + x_seq

        if self.rhythm_branch_mode == "low_only":
            low_used = low_fused
            high_used = torch.zeros_like(high_fused)
            residual = low_fused
        elif self.rhythm_branch_mode == "high_only":
            low_used = torch.zeros_like(low_fused)
            high_used = high_fused
            residual = high_fused
        else:
            low_used = low_fused
            high_used = high_fused
            residual = enhanced

        context = self.context_proj(torch.cat([low_used, high_used], dim=1))
        context = context + residual
        context = self.context_norm(context.transpose(1, 2))
        return low_fused.transpose(1, 2), high_fused.transpose(1, 2), context


class RhythmicStateModelingSwitch(nn.Module):

    def __init__(
        self,
        emb_size,
        low_kernels=(5, 9, 13),
        high_kernels=(3, 5, 7),
        rhythm_branch_mode="full",
        rhythm_impl="fourier",
        num_blocks=None,
        sparsity_threshold=0.01,
        rhythm_low_ratio=0.5,
    ):
        super().__init__()
        self.rhythm_impl = rhythm_impl
        if rhythm_impl == "fourier":
            self.impl = FourierRhythmicStateModeling(
                emb_size=emb_size,
                low_kernels=low_kernels,
                high_kernels=high_kernels,
                rhythm_branch_mode=rhythm_branch_mode,
                num_blocks=num_blocks,
                sparsity_threshold=sparsity_threshold,
                rhythm_low_ratio=rhythm_low_ratio,
            )
        elif rhythm_impl == "conv":
            self.impl = TimeDomainRhythmicStateModeling(
                emb_size=emb_size,
                low_kernels=low_kernels,
                high_kernels=high_kernels,
                rhythm_branch_mode=rhythm_branch_mode,
            )
        else:
            raise ValueError(f"Unsupported rhythm_impl: {rhythm_impl}")

    def forward(self, x):
        return self.impl(x)


class FourierRhythmicStateModulator(nn.Module):

    def __init__(self, d_model, d_inner):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.token_modulator = nn.Linear(d_model, d_inner)
        self.residual_modulator = nn.Linear(d_model, d_inner)

    def forward(self, rhythmic_context):
        context = self.norm(rhythmic_context)
        token_scale = torch.sigmoid(self.token_modulator(context))
        residual_bias = self.residual_modulator(context)
        return token_scale, residual_bias

class FRSMambaStateSpaceMixer(nn.Module):

    def __init__(self, input_channels, use_fourier_rhythmic_modeling=True, rhythm_branch_mode="full"):
        super().__init__()
        self.d_model = input_channels
        self.d_inner = self.d_model * 2
        self.dt_rank = math.ceil(self.d_model / 16)
        self.d_state = 16
        self.use_fourier_rhythmic_modeling = use_fourier_rhythmic_modeling
        self.rhythm_branch_mode = rhythm_branch_mode

        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2)
        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            kernel_size=3,
            groups=self.d_inner,
            padding=2,
        )
        self.x_proj = nn.Linear(self.d_inner, self.dt_rank + self.d_state * 2, bias=False)
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True)

        a = repeat(torch.arange(1, self.d_state + 1), "n -> d n", d=self.d_inner)
        self.A_log = nn.Parameter(torch.log(a))
        self.D = nn.Parameter(torch.ones(self.d_inner))
        self.out_proj = nn.Linear(self.d_inner, self.d_model)
        self.fourier_rhythmic_modulator = FourierRhythmicStateModulator(self.d_model, self.d_inner) if use_fourier_rhythmic_modeling else None

    

    def ssm(self, x):
        _, n = self.A_log.shape
        a = -torch.exp(self.A_log.float())
        d = self.D.float()
        x_dbl = self.x_proj(x)
        delta, b, c = x_dbl.split(split_size=[self.dt_rank, n, n], dim=-1)
        delta = F.softplus(self.dt_proj(delta))
        return self.selective_scan(x, delta, a, b, c, d)

    def selective_scan(self, u, delta, a, b, c, d):
        batch_size, sequence_length, inner_dim = u.shape
        state_dim = a.shape[1]
        delta_a = torch.exp(einsum(delta, a, "b l d, d n -> b l d n"))
        delta_b_u = einsum(delta, b, u, "b l d, b l n, b l d -> b l d n")

        state = torch.zeros((batch_size, inner_dim, state_dim), device=delta_a.device, dtype=delta_a.dtype)
        outputs = []
        for index in range(sequence_length):
            state = delta_a[:, index] * state + delta_b_u[:, index]
            y = einsum(state, c[:, index, :], "b d n, b n -> b d")
            outputs.append(y)
        y = torch.stack(outputs, dim=1)
        return y + u * d
    
    def forward(self, x, rhythmic_context=None, return_details=False):
        _, sequence_length, _ = x.shape
        x_and_res = self.in_proj(x)
        x, res = x_and_res.split(split_size=[self.d_inner, self.d_inner], dim=-1)

        token_scale = None
        residual_bias = None
        if self.use_fourier_rhythmic_modeling and self.rhythm_branch_mode != "none" and rhythmic_context is not None:
            token_scale, residual_bias = self.fourier_rhythmic_modulator(rhythmic_context)
            x = x * (1.0 + token_scale)
            res = res + residual_bias
 
        x = rearrange(x, "b l d -> b d l")
        x = self.conv1d(x)[:, :, :sequence_length]
        x = rearrange(x, "b d l -> b l d")

        x = F.silu(x)
        y = self.ssm(x)
        y = y * F.silu(res)
        out = self.out_proj(y)
        if not return_details:
            return out
        return out, {
            "token_scale": token_scale,
            "residual_bias": residual_bias,
        }


class FRSMambaBlock(nn.Module):

    def __init__(
        self,
        emb_size,
        drop_p=0.3,
        use_fourier_rhythmic_modeling=True,
        rhythm_branch_mode="full",
        rhythm_impl="fourier",
        rhythm_num_blocks=None,
        rhythm_sparsity_threshold=0.01,
        rhythm_low_ratio=0.5,
    ):
        super().__init__()
        self.norm = nn.LayerNorm(emb_size)
        self.use_fourier_rhythmic_modeling = use_fourier_rhythmic_modeling
        self.rhythm_branch_mode = rhythm_branch_mode
        self.rhythm_impl = rhythm_impl
        self.rhythmic_state_modeling = (
            RhythmicStateModelingSwitch(
                emb_size,
                rhythm_branch_mode=rhythm_branch_mode,
                rhythm_impl=rhythm_impl,
                num_blocks=rhythm_num_blocks,
                sparsity_threshold=rhythm_sparsity_threshold,
                rhythm_low_ratio=rhythm_low_ratio,
            )
            if use_fourier_rhythmic_modeling
            else None
        )
        self.temporal_mixer = FRSMambaStateSpaceMixer(input_channels=emb_size, use_fourier_rhythmic_modeling=use_fourier_rhythmic_modeling, rhythm_branch_mode=rhythm_branch_mode)
        self.drop = nn.Dropout(drop_p)

    @property
    def fourier_rhythmic_state_modeling(self):
        return self.rhythmic_state_modeling

    def forward(self, x, return_details=False):
        x_norm = self.norm(x)
        rhythmic_context = None
        low_fused = None
        high_fused = None
        
        if self.use_fourier_rhythmic_modeling and self.rhythm_branch_mode != "none":
            low_fused, high_fused, rhythmic_context = self.rhythmic_state_modeling(x_norm)
        if return_details:
            out, mixer_details = self.temporal_mixer(x_norm, rhythmic_context, return_details=True)
        else:
            out = self.temporal_mixer(x_norm, rhythmic_context)
            mixer_details = None
        residual_out = x + self.drop(out)
        if not return_details:
            return residual_out
        return residual_out, {
            "block_input": x,
            "normalized_tokens": x_norm,
            "low_fused": low_fused,
            "high_fused": high_fused,
            "rhythmic_context": rhythmic_context,
            "token_scale": None if mixer_details is None else mixer_details["token_scale"],
            "residual_bias": None if mixer_details is None else mixer_details["residual_bias"],
            "block_output": residual_out,
        }

class PredictionHead(nn.Sequential):
    """Paper module: final decision head."""

    def __init__(self, flatten_number, n_classes):
        super().__init__()
        self.fc = nn.Sequential(nn.Dropout(0.5), nn.Linear(flatten_number, n_classes))

    def forward(self, x):
        return self.fc(x)

class CFSPMNet(nn.Module):
   
    def __init__(
        self,
        emb_size=64,
        depth=2,
        eeg_f1=10,
        eeg_kernel_size=64,
        eeg_D=3,
        eeg_pooling_size1=8,
        eeg_pooling_size2=8,
        eeg_dropout_rate=0.3,
        eeg_number_channel=22,
        number_class=2,
        flatten_eeg=600,
        use_fourier_rhythmic_modeling=True,
        use_temporal_encoder=True,
        rhythm_branch_mode="full",
        rhythm_impl="fourier",
        rhythm_num_blocks=None,
        rhythm_sparsity_threshold=0.01,
        rhythm_low_ratio=0.5,
        **kwargs,
    ):
        super().__init__()
        self.emb_size = emb_size
        self.use_temporal_encoder = use_temporal_encoder
        self.rhythm_branch_mode = rhythm_branch_mode

        self.physiological_tokenizer = MultiScalePhysiologicalTokenizer(
            f1=eeg_f1,
            kernel_size=eeg_kernel_size,
            D=eeg_D,
            pooling_size1=eeg_pooling_size1,
            pooling_size2=eeg_pooling_size2,
            dropout_rate=eeg_dropout_rate,
            number_channel=eeg_number_channel,
            emb_size=emb_size,
        )
        self.position_encoding = LearnablePositionEncoding(emb_size, dropout=0.1) if use_temporal_encoder else nn.Identity()
        
        self.temporal_encoder = (
            nn.Sequential(
                *[
                    FRSMambaBlock(
                        emb_size,
                        drop_p=eeg_dropout_rate,
                        use_fourier_rhythmic_modeling=use_fourier_rhythmic_modeling,
                        rhythm_branch_mode=rhythm_branch_mode,
                        rhythm_impl=rhythm_impl,
                        rhythm_num_blocks=rhythm_num_blocks,
                        rhythm_sparsity_threshold=rhythm_sparsity_threshold,
                        rhythm_low_ratio=rhythm_low_ratio,
                    )
                    for _ in range(depth)
                ]
            )
            if use_temporal_encoder
            else nn.Identity()
        )
        self.final_norm = nn.LayerNorm(emb_size)
        self.flatten = nn.Flatten()
        self.prediction_head = PredictionHead(flatten_eeg, number_class)

    def forward_features(self, x, return_block_details=False):
        if x.dim() == 3:
            x = x.unsqueeze(1)

        physiological_tokens = self.physiological_tokenizer(x)
        export = {
            "physiological_tokens": physiological_tokens,
        }
        if not self.use_temporal_encoder:
            features = self.final_norm(physiological_tokens)
            flatten_features = self.flatten(features)
            logits = self.prediction_head(flatten_features)
            export.update(
                {
                    "positioned_tokens": physiological_tokens,
                    "encoder_features": features,
                    "flatten_features": flatten_features,
                    "logits": logits,
                }
            )
            return export

        scaled_tokens = physiological_tokens * math.sqrt(self.emb_size)
        positioned_tokens = self.position_encoding(scaled_tokens)
        hidden = positioned_tokens
        
        block_details = []
        for block_idx, block in enumerate(self.temporal_encoder):
            if return_block_details:
                hidden, details = block(hidden, return_details=True)
                details["block_index"] = block_idx
                block_details.append(details)
            else:
                hidden = block(hidden)
        features = self.final_norm(hidden)
        flatten_features = self.flatten(features)
        logits = self.prediction_head(flatten_features)

        export.update(
            {
                "positioned_tokens": positioned_tokens,
                "encoder_features": features,
                "flatten_features": flatten_features,
                "logits": logits,
            }
        )

        if return_block_details and block_details:
            export["encoder_block_details"] = block_details
            export.update(
                {
                    "low_fused": block_details[-1]["low_fused"],
                    "high_fused": block_details[-1]["high_fused"],
                    "rhythmic_context": block_details[-1]["rhythmic_context"],
                    "token_scale": block_details[-1]["token_scale"],
                    "residual_bias": block_details[-1]["residual_bias"],
                }
            )
        return export

    def forward(self, x, return_features=False, return_block_details=False):
        export = self.forward_features(x, return_block_details=return_block_details)
        if return_features:
            return export
        return export["logits"]


def build_cfspmnet_from_args(args):
    return CFSPMNet(
        emb_size=args.emb_size,
        depth=args.depth,
        eeg_f1=args.f1,
        eeg_kernel_size=args.kernel_size,
        eeg_D=args.D,
        eeg_pooling_size1=args.pooling_size1,
        eeg_pooling_size2=args.pooling_size2,
        eeg_dropout_rate=args.dropout,
        eeg_number_channel=args.n_channels,
        number_class=args.n_classes,
        flatten_eeg=args.flatten,
        use_fourier_rhythmic_modeling=getattr(args, "use_fourier_rhythmic_modeling", True),
        use_temporal_encoder=getattr(args, "use_temporal_encoder", True),
        rhythm_branch_mode=getattr(args, "rhythm_branch_mode", "full"),
        rhythm_impl=getattr(args, "rhythm_impl", "fourier"),
        rhythm_num_blocks=getattr(args, "rhythm_num_blocks", None),
        rhythm_sparsity_threshold=getattr(args, "rhythm_sparsity_threshold", 0.01),
        rhythm_low_ratio=getattr(args, "rhythm_low_ratio", 0.5),
    )

