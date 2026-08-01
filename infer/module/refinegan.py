import functools
import math

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils import weight_norm, remove_weight_norm
from torch.utils.checkpoint import checkpoint
from torchaudio.functional.functional import (
    _apply_sinc_resample_kernel,
    _get_sinc_resample_kernel,
)

from infer.module.commons import init_weights, get_padding


@functools.lru_cache(maxsize=64)
def _cached_sinc_resample_kernel(
    orig_freq, new_freq, gcd, lowpass_filter_width, rolloff, resampling_method, beta, device, dtype
):
    # torchaudio.functional.resample() builds its filter kernel by creating a fresh
    # CPU tensor and .to(device)-ing it, which is not legal to record inside a CUDA
    # graph capture region (raises "operation not permitted when stream is
    # capturing"). realtime_gui.py wraps this generator's forward pass in
    # tools/cuda_graph.py's run_cuda_graph, so we memoize the kernel-building step:
    # it only ever actually runs during the graph's warmup replay (plain eager
    # execution, not real capture), and the capture itself hits this cache, which is
    # a pure dict lookup with no device-transfer op to trip the capture.
    return _get_sinc_resample_kernel(
        orig_freq, new_freq, gcd, lowpass_filter_width, rolloff, resampling_method, beta, device, dtype
    )


def _resample_cuda_graph_safe(x, orig_freq, new_freq, lowpass_filter_width, rolloff, resampling_method, beta):
    gcd = math.gcd(orig_freq, new_freq)
    # Always build the filter kernel in fp32, regardless of x's dtype: the sinc /
    # Kaiser / Hann window math involves large orig_freq*new_freq magnitudes and
    # transcendental functions that lose too much precision - and can overflow to
    # NaN - in fp16. Only the conv1d application below needs to run in x's own
    # dtype; the kernel is cast down to match right before that.
    kernel, width = _cached_sinc_resample_kernel(
        orig_freq, new_freq, gcd, lowpass_filter_width, rolloff, resampling_method, beta, x.device, torch.float32
    )
    kernel = kernel.to(dtype=x.dtype)
    return _apply_sinc_resample_kernel(x, orig_freq, new_freq, gcd, kernel, width)


class ResBlock(nn.Module):
    """
    Residual block with multiple dilated convolutions.

    This block applies a sequence of dilated convolutional layers with Leaky ReLU activation.
    It's designed to capture information at different scales due to the varying dilation rates.
    """

    def __init__(
        self,
        channels: int,
        kernel_size: int = 7,
        dilation: tuple[int] = (1, 3, 5),
        leaky_relu_slope: float = 0.2,
    ):
        super().__init__()

        self.leaky_relu_slope = leaky_relu_slope

        self.convs1 = nn.ModuleList(
            [
                weight_norm(
                    nn.Conv1d(
                        channels,
                        channels,
                        kernel_size,
                        stride=1,
                        dilation=d,
                        padding=get_padding(kernel_size, d),
                    )
                )
                for d in dilation
            ]
        )
        self.convs1.apply(init_weights)

        self.convs2 = nn.ModuleList(
            [
                weight_norm(
                    nn.Conv1d(
                        channels,
                        channels,
                        kernel_size,
                        stride=1,
                        dilation=1,
                        padding=get_padding(kernel_size, 1),
                    )
                )
                for d in dilation
            ]
        )
        self.convs2.apply(init_weights)

    def forward(self, x: torch.Tensor):
        for c1, c2 in zip(self.convs1, self.convs2):
            xt = F.leaky_relu(x, self.leaky_relu_slope)
            xt = c1(xt)
            xt = F.leaky_relu(xt, self.leaky_relu_slope)
            xt = c2(xt)
            x = xt + x

        return x

    def remove_weight_norm(self):
        for c1, c2 in zip(self.convs1, self.convs2):
            remove_weight_norm(c1)
            remove_weight_norm(c2)


class AdaIN(nn.Module):
    """
    Adaptive Instance Normalization layer.

    This layer applies a scaling factor to the input based on a learnable weight.
    """

    def __init__(
        self,
        *,
        channels: int,
        leaky_relu_slope: float = 0.2,
    ):
        super().__init__()

        self.weight = nn.Parameter(torch.ones(channels) * 1e-4)
        # safe to use in-place as it is used on a new x+gaussian tensor
        self.activation = nn.LeakyReLU(leaky_relu_slope)

    def forward(self, x: torch.Tensor):
        gaussian = torch.randn_like(x) * self.weight[None, :, None]

        return self.activation(x + gaussian)


class ParallelResBlock(nn.Module):
    """
    Parallel residual block that applies multiple residual blocks with different kernel sizes in parallel.
    """

    def __init__(
        self,
        *,
        in_channels: int,
        out_channels: int,
        kernel_sizes: tuple[int] = (3, 7, 11),
        dilation: tuple[int] = (1, 3, 5),
        leaky_relu_slope: float = 0.2,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.input_conv = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=7,
            stride=1,
            padding=3,
        )

        self.input_conv.apply(init_weights)

        self.blocks = nn.ModuleList(
            [
                nn.Sequential(
                    AdaIN(channels=out_channels),
                    ResBlock(
                        out_channels,
                        kernel_size=kernel_size,
                        dilation=dilation,
                        leaky_relu_slope=leaky_relu_slope,
                    ),
                    AdaIN(channels=out_channels),
                )
                for kernel_size in kernel_sizes
            ]
        )

    def forward(self, x: torch.Tensor):
        x = self.input_conv(x)
        return torch.stack([block(x) for block in self.blocks], dim=0).mean(dim=0)

    def remove_weight_norm(self):
        # input_conv is a plain Conv1d (never weight-normed); only the inner
        # ResBlock convs need it removed.
        for block in self.blocks:
            block[1].remove_weight_norm()


class SineGenerator(nn.Module):
    """
    Definition of sine generator

    Generates sine waveforms with optional harmonics and additive noise.
    Can be used to create harmonic noise source for neural vocoders.
    """

    def __init__(
        self,
        samp_rate,
        harmonic_num=0,
        sine_amp=0.1,
        noise_std=0.003,
        voiced_threshold=0,
    ):
        super(SineGenerator, self).__init__()
        self.sine_amp = sine_amp
        self.noise_std = noise_std
        self.harmonic_num = harmonic_num
        self.dim = self.harmonic_num + 1
        self.sampling_rate = samp_rate
        self.voiced_threshold = voiced_threshold

        self.merge = nn.Sequential(
            nn.Linear(self.dim, 1, bias=False),
            nn.Tanh(),
        )

    def _f02uv(self, f0):
        # generate uv signal
        uv = torch.ones_like(f0)
        uv = uv * (f0 > self.voiced_threshold)
        return uv

    def _f02sine(self, f0_values):
        """f0_values: (batchsize, length, dim)
        where dim indicates fundamental tone and overtones
        """
        # convert to F0 in rad. The integer part n can be ignored
        # because 2 * np.pi * n doesn't affect phase
        rad_values = (f0_values / self.sampling_rate) % 1

        # initial phase noise (no noise for fundamental component)
        rand_ini = torch.rand(
            f0_values.shape[0], f0_values.shape[2], device=f0_values.device
        )
        rand_ini[:, 0] = 0
        rad_values[:, 0, :] = rad_values[:, 0, :] + rand_ini

        # instantanouse phase sine[t] = sin(2*pi \sum_i=1 ^{t} rad)
        tmp_over_one = torch.cumsum(rad_values, 1) % 1
        tmp_over_one_idx = (tmp_over_one[:, 1:, :] - tmp_over_one[:, :-1, :]) < 0
        cumsum_shift = torch.zeros_like(rad_values)
        cumsum_shift[:, 1:, :] = tmp_over_one_idx * -1.0

        sines = torch.sin(torch.cumsum(rad_values + cumsum_shift, dim=1) * 2 * np.pi)

        return sines

    def forward(self, f0):
        with torch.no_grad():
            f0_buf = torch.zeros(f0.shape[0], f0.shape[1], self.dim, device=f0.device)
            # fundamental component
            f0_buf[:, :, 0] = f0[:, :, 0]
            for idx in range(self.harmonic_num):
                f0_buf[:, :, idx + 1] = f0_buf[:, :, 0] * (idx + 2)

            sine_waves = self._f02sine(f0_buf) * self.sine_amp

            uv = self._f02uv(f0)

            noise_amp = uv * self.noise_std + (1 - uv) * self.sine_amp / 3
            noise = noise_amp * torch.randn_like(sine_waves)

            sine_waves = sine_waves * uv + noise

        # merge with grad
        # f0_buf/sine_waves are computed in fp32 above regardless of model precision
        # (torch.zeros default dtype); cast to the merge layer's dtype so this works
        # under half-precision inference too (mirrors GeneratorNSF's SourceModuleHnNSF
        # doing the same: `sine_wavs.to(dtype=self.l_linear.weight.dtype)`).
        sine_waves = sine_waves.to(dtype=self.merge[0].weight.dtype)
        return self.merge(sine_waves)


class RefineGANGenerator(nn.Module):
    """
    RefineGAN generator for audio synthesis.

    This generator uses a combination of downsampling, residual blocks, and parallel residual blocks
    to refine an input mel-spectrogram (in this port: the model's latent `z`) and fundamental
    frequency (F0) into an audio waveform. It can also incorporate global conditioning.

    Ported from Applio (rvc/lib/algorithm/generators/refinegan.py).
    """

    def __init__(
        self,
        *,
        sample_rate: int = 44100,
        downsample_rates: tuple[int] = (2, 2, 8, 8),  # unused
        upsample_rates: tuple[int] = (8, 8, 2, 2),
        leaky_relu_slope: float = 0.2,
        num_mels: int = 128,
        start_channels: int = 16,  # unused
        gin_channels: int = 256,
        checkpointing: bool = False,
        upsample_initial_channel=512,
        # "sinc_interp_kaiser" matches Applio/upstream training exactly, but its Kaiser
        # window uses torch.i0 (Bessel function). Kept configurable in case a future
        # export/inference backend here can't support it (mirrors the other app's port).
        resampling_method: str = "sinc_interp_kaiser",
    ):
        super().__init__()
        self.upsample_rates = upsample_rates
        self.leaky_relu_slope = leaky_relu_slope
        self.checkpointing = checkpointing
        self.resampling_method = resampling_method

        self.upp = int(np.prod(upsample_rates))
        self.m_source = SineGenerator(sample_rate)

        # expanded f0 sinegen -> match mel_conv
        self.pre_conv = weight_norm(
            nn.Conv1d(
                1,
                16,
                7,
                1,
                padding=3,
            )
        )

        # f0 downsampling and upchanneling
        channels = start_channels
        size = self.upp
        self.downsample_blocks = nn.ModuleList([])
        self.df0 = []
        for i, u in enumerate(upsample_rates):

            new_size = int(size / upsample_rates[-i - 1])
            # T dimension factors for torchaudio.functional.resample
            self.df0.append([size, new_size])
            size = new_size

            new_channels = channels * 2
            self.downsample_blocks.append(
                weight_norm(nn.Conv1d(channels, new_channels, 7, 1, padding=3))
            )
            channels = new_channels

        # mel handling
        channels = upsample_initial_channel

        self.mel_conv = weight_norm(
            nn.Conv1d(
                num_mels,
                channels // 2,
                7,
                1,
                padding=3,
            )
        )

        self.mel_conv.apply(init_weights)

        if gin_channels != 0:
            self.cond = nn.Conv1d(256, channels // 2, 1)

        self.upsample_blocks = nn.ModuleList([])
        self.upsample_conv_blocks = nn.ModuleList([])

        for rate in upsample_rates:
            new_channels = channels // 2

            self.upsample_blocks.append(nn.Upsample(scale_factor=rate, mode="linear"))

            self.upsample_conv_blocks.append(
                ParallelResBlock(
                    in_channels=channels + channels // 4,
                    out_channels=new_channels,
                    kernel_sizes=(3, 7, 11),
                    dilation=(1, 3, 5),
                    leaky_relu_slope=leaky_relu_slope,
                )
            )

            channels = new_channels

        self.conv_post = weight_norm(
            nn.Conv1d(channels, 1, 7, 1, padding=3, bias=False)
        )
        self.conv_post.apply(init_weights)

    def forward(self, mel: torch.Tensor, f0: torch.Tensor, g: torch.Tensor = None):
        f0_size = mel.shape[-1]
        # change f0 helper to full size
        f0 = F.interpolate(f0.unsqueeze(1), size=f0_size * self.upp, mode="linear")
        # get f0 turned into sines harmonics
        har_source = self.m_source(f0.transpose(1, 2)).transpose(1, 2)
        # prepare for fusion to mel
        x = self.pre_conv(har_source)
        # downsampled/upchanneled versions for each upscale
        downs = []
        for block, (old_size, new_size) in zip(self.downsample_blocks, self.df0):
            x = F.leaky_relu(x, self.leaky_relu_slope)
            downs.append(x)
            # attempt to cancel spectral aliasing
            x = _resample_cuda_graph_safe(
                x.contiguous(),
                orig_freq=int(f0_size * old_size),
                new_freq=int(f0_size * new_size),
                lowpass_filter_width=64,
                rolloff=0.9475937167399596,
                resampling_method=self.resampling_method,
                beta=14.769656459379492,
            )
            x = block(x)

        # expanding spectrogram from 192 to 256 channels
        mel = self.mel_conv(mel)
        if g is not None:
            # adding expanded speaker embedding
            mel = mel + self.cond(g)

        x = torch.cat([mel, x], dim=1)

        for ups, res, down in zip(
            self.upsample_blocks,
            self.upsample_conv_blocks,
            reversed(downs),
        ):
            x = F.leaky_relu(x, self.leaky_relu_slope)

            if self.training and self.checkpointing:
                x = checkpoint(ups, x, use_reentrant=False)
                x = torch.cat([x, down], dim=1)
                x = checkpoint(res, x, use_reentrant=False)
            else:
                x = ups(x)
                x = torch.cat([x, down], dim=1)
                x = res(x)

        x = F.leaky_relu(x, self.leaky_relu_slope)
        x = self.conv_post(x)
        x = torch.tanh(x)

        return x

    def remove_weight_norm(self):
        remove_weight_norm(self.pre_conv)
        remove_weight_norm(self.mel_conv)
        remove_weight_norm(self.conv_post)

        # downsample_blocks are plain weight-normed Conv1d layers (no custom
        # remove_weight_norm method of their own), unlike upsample_conv_blocks.
        for block in self.downsample_blocks:
            remove_weight_norm(block)

        for block in self.upsample_conv_blocks:
            block.remove_weight_norm()
