# Copyright 2025 Reo Yoneyama (Nagoya University)
#  MIT License (https://opensource.org/licenses/MIT)

"""Modules for signal decomposition."""

from typing import List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from scipy import __version__ as scipy_version
from torch import Tensor, nn

# Validate PyTorch version for Kaiser window compatibility
if scipy_version >= "1.0.1":
    from scipy.signal.windows import kaiser
else:
    from scipy.signal import kaiser


def is_power_of_two(n: int):
    return n > 0 and (n & (n - 1)) == 0


def design_prototype_filter(taps=62, cutoff_ratio=0.142, beta=9.0):
    """
    Design a prototype filter for Pseudo Quadrature Mirror Filter (PQMF) banks.

    Reference:
        - "A Kaiser window approach for the design of prototype filters of cosine modulated filterbanks"
        - https://ieeexplore.ieee.org/abstract/document/681427

    Args:
        taps (int): Number of filter taps. Must be an even number.
        cutoff_ratio (float): Cut-off frequency as a ratio of Nyquist frequency. Must be in the range (0.0, 1.0).
        beta (float): Beta parameter for the Kaiser window.

    Returns:
        ndarray: Impulse response of the designed prototype filter with shape (taps + 1,).
    """
    # check the arguments are valid
    assert taps % 2 == 0, "The number of taps mush be even number."
    assert 0.0 < cutoff_ratio < 1.0, "Cutoff ratio must be > 0.0 and < 1.0."

    # make initial filter
    omega_c = np.pi * cutoff_ratio
    with np.errstate(invalid="ignore"):
        h_i = np.sin(omega_c * (np.arange(taps + 1) - 0.5 * taps)) / (
            np.pi * (np.arange(taps + 1) - 0.5 * taps)
        )
    h_i[taps // 2] = np.cos(0) * cutoff_ratio  # fix nan due to indeterminate form

    # apply kaiser window
    w = kaiser(taps + 1, beta)
    h = h_i * w

    return h


class PQMF(nn.Module):
    """Pseudo Quadrature Mirror Filter (PQMF) module.

    This implementation is based on the PQMF implementation used in the ParallelWaveGAN repository, which in turn follows
    the design described in: "Near-perfect-reconstruction pseudo-QMF banks" (https://ieeexplore.ieee.org/document/258122).

    Reference implementations:
        - ParallelWaveGAN (https://github.com/kan-bayashi/ParallelWaveGAN)
    """

    def __init__(
        self,
        num_split: int = 4,
        taps: int = 62,
        cutoff_ratio: float = 0.142,
        beta: float = 9.0,
    ):
        """Initilize PQMF module.

        The cutoff_ratio and beta parameters are optimized for #num_split = 4.
        See dicussion in https://github.com/kan-bayashi/ParallelWaveGAN/issues/195.

        Args:
            num_split (int): The number of num_split.
            taps (int): The number of filter taps.
            cutoff_ratio (float): Cut-off frequency ratio.
            beta (float): Beta coefficient for kaiser window.
        """
        super().__init__()
        self.taps = taps
        # filter for downsampling & upsampling
        updown_filter = torch.zeros((num_split, num_split, num_split)).float()
        for k in range(num_split):
            updown_filter[k, k, 0] = 1.0
        self.register_buffer("updown_filter", updown_filter)
        self.num_split = num_split

        # build analysis & synthesis filter coefficients
        h_proto = design_prototype_filter(taps, cutoff_ratio, beta)
        h_analysis = np.zeros((num_split, len(h_proto)))
        h_synthesis = np.zeros((num_split, len(h_proto)))
        for k in range(num_split):
            h_analysis[k] = (
                2
                * h_proto
                * np.cos(
                    (2 * k + 1)
                    * (np.pi / (2 * num_split))
                    * (np.arange(taps + 1) - (taps / 2))
                    + (-1) ** k * np.pi / 4
                )
            )
            h_synthesis[k] = (
                2
                * h_proto
                * np.cos(
                    (2 * k + 1)
                    * (np.pi / (2 * num_split))
                    * (np.arange(taps + 1) - (taps / 2))
                    - (-1) ** k * np.pi / 4
                )
            )

        # Save filter weights
        analysis_filter = torch.from_numpy(h_analysis).float().unsqueeze(1)
        synthesis_filter = torch.from_numpy(h_synthesis).float().unsqueeze(0)
        self.register_buffer("analysis_filter", analysis_filter)
        self.register_buffer("synthesis_filter", synthesis_filter)
        self.pad = nn.ConstantPad1d((taps // 2, taps // 2), value=0)

    def analysis(self, x: Tensor) -> List[Tensor]:
        """
        Decompose the input signal into num_split subscale signals.

        Args:
            x (Tensor): Input signal with shape (B, 1, T).

        Returns:
            List[Tensor]: List of subscale signals, each with shape (B, 1, T_sub).
        """
        # (B, 1, T) -> (B, num_split, T')
        x = F.conv1d(x, self.analysis_filter)
        x = F.conv1d(self.pad(x), self.updown_filter, stride=self.num_split)
        # Split along channel dimension into a list of mono subscale signals
        return list(x.chunk(self.num_split, dim=1))

    def synthesis(self, xs: List[Tensor]) -> Tensor:
        """
        Reconstruct the full-scale signal from subscale signals.

        Args:
            xs (List[Tensor]): List of subscale signals, each with shape (B, 1, T_sub).

        Returns:
            Tensor: Reconstructed signal of shape (B, 1, T).
        """
        assert (
            len(xs) == self.num_split
        ), f"Expected {self.num_split} subscales, but got {len(xs)}."
        x = torch.cat(xs, dim=1)
        x = F.conv_transpose1d(
            x, self.updown_filter * self.num_split, stride=self.num_split
        )
        x = F.conv1d(self.pad(x), self.synthesis_filter)
        return x


class MultiStream1d(nn.Module):
    """
    A module for subscale analysis and synthesis with learnable convolutions.

    Reference:
        - Multi-stream HiFi-GAN with data-driven waveform decomposition
        (https://ast-astrec.nict.go.jp/release/preprints/preprint_asru_2021_okamoto.pdf)
    """

    def __init__(self, num_split: int = 4, taps: int = 62) -> None:
        """Initilize MultiStream1d module.

        Args:
            num_split (int): Number of num_split to decompose the input signal into.
            taps (int): Number of filter taps for the analysis and synthesis filter.
        """
        super().__init__()
        self.num_split = num_split

        # Initialize a filter for downsampling and upsampling.
        updown_filter = torch.zeros((num_split, num_split, num_split)).float()
        for k in range(num_split):
            updown_filter[k, k, 0] = 1.0
        self.register_buffer("updown_filter", updown_filter)

        # Define analysis and synthesis convolutions based on the causal mode
        self.conv_analysis = nn.Conv1d(
            1,
            num_split,
            taps + 1,
            padding=taps // 2,
            padding_mode="reflect",
            bias=False,
        )
        self.conv_synthesis = nn.Conv1d(
            num_split,
            1,
            taps + 1,
            padding=taps // 2,
            padding_mode="reflect",
            bias=False,
        )

    def analysis(self, x: Tensor) -> List[Tensor]:
        """
        Decompose the input signal into num_split subscale signals.

        Args:
            x (Tensor): Input signal with shape (B, 1, T).

        Returns:
            List[Tensor]: List of subscale signals, each with shape (B, 1, T_sub).
        """
        # (B, 1, T) -> (B, num_split, T')
        x = self.conv_analysis(x)
        x = F.conv1d(x, self.updown_filter, stride=self.num_split)
        return list(x.chunk(self.num_split, dim=1))

    def synthesis(self, xs: List[Tensor]) -> Tensor:
        """
        Reconstruct the full-scale signal from subscale signals.

        Args:
            xs (List[Tensor]): List of subscale signals, each with shape (B, 1, T_sub).

        Returns:
            Tensor: Reconstructed signal of shape (B, 1, T).
        """
        assert (
            len(xs) == self.num_split
        ), f"Expected {self.num_split} subscales, but got {len(xs)}."
        x = torch.cat(xs, dim=1)
        x = F.conv_transpose1d(
            x, self.updown_filter * self.num_split, stride=self.num_split
        )
        x = self.conv_synthesis(x)
        return x


class DWT1d(nn.Module):
    """
    Discrete Wavelet Transform (DWT) using Haar (Daubechies 1) wavelet for 1D signals.

    This class performs a single-level decomposition and reconstruction of by splitting
    the input signal into its approximation and detail components.
    """

    def __init__(self, num_split: int) -> None:
        """
        Initialize the DWT1d module.

        Args:
            num_split (int): Number of subscales (must be a power of two).
        """
        super().__init__()
        assert is_power_of_two(num_split)
        self.num_split = num_split

        lpf = np.sqrt(0.5) * torch.FloatTensor([1.0, 1.0])
        hpf = np.sqrt(0.5) * torch.FloatTensor([1.0, -1.0])
        self.register_buffer("lpf", lpf.reshape(1, 1, -1))
        self.register_buffer("hpf", hpf.reshape(1, 1, -1))

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Single-level forward DWT using Haar wavelet.

        Args:
            x (Tensor): Input tensor of shape (B, C, T).

        Returns:
            Tuple[Tensor, Tensor]: (low, high) components.
        """
        _, channels, length = x.shape
        x = F.pad(x, (0, 1), mode="replicate") if length % 2 else x

        # Decompose into low and high frequency components
        low = F.conv1d(x, self.lpf, stride=2, groups=channels)
        high = F.conv1d(x, self.hpf, stride=2, groups=channels)
        return low, high

    def inverse(self, low: Tensor, high: Tensor) -> Tensor:
        """
        Applies the inverse DWT to reconstruct a signal from its components.

        Args:
            low (Tensor): Low-frequency component.
            high (Tensor): High-frequency component.

        Returns:
            Tensor: Reconstructed signal of shape (B, C, T).
        """
        channels = low.shape[1]
        # Recomstruct from low and high frequency components
        low_up = F.conv_transpose1d(low, self.lpf, stride=2, groups=channels)
        high_up = F.conv_transpose1d(high, self.hpf, stride=2, groups=channels)
        return low_up + high_up

    def analysis(self, x: Tensor) -> List[Tensor]:
        """
        Decompose the input signal into num_split subscale signals.

        Args:
            x (Tensor): Input signal with shape (B, 1, T).

        Returns:
            List[Tensor]: List of subscale signals, each with shape (B, 1, T_sub).
        """
        exponent = self.num_split.bit_length() - 1
        xs = [x]
        new_xs: List[Tensor] = []
        for _ in range(exponent):
            for x_ in xs:
                scale, detail = self.forward(x_)
                new_xs += [scale, detail]
            xs = new_xs
            new_xs = []
        return xs

    def synthesis(self, xs: List[Tensor]) -> Tensor:
        """
        Reconstruct the full-scale signal from subscale signals.

        Args:
            xs (List[Tensor]): List of subscale signals, each with shape (B, 1, T_sub).

        Returns:
            Tensor: Reconstructed signal of shape (B, 1, T).
        """
        assert (
            len(xs) == self.num_split
        ), f"Expected {self.num_split} subscales, but got {len(xs)}."
        exponent = self.num_split.bit_length() - 1
        new_xs: List[Tensor] = []
        for _ in range(exponent):
            for scale, detail in zip(xs[::2], xs[1::2]):
                x = self.inverse(scale, detail)
                new_xs += [x]
            xs = new_xs
            new_xs = []
        return xs[0]
