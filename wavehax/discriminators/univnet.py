# Copyright 2024 Reo Yoneyama (Nagoya University)
#  MIT License (https://opensource.org/licenses/MIT)

"""
Discriminator modules for GAN-based vocoders.

This code contains the implementation of HiFi-GAN's multi-period discriminator and UnivNet's multi-resolution spectral discriminator.

References:
    - https://github.com/jik876/hifi-gan
    - https://www.isca-speech.org/archive/interspeech_2021/jang21_interspeech.html
    - https://github.com/kan-bayashi/ParallelWaveGAN
"""

import copy
from logging import getLogger
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from wavehax.modules import spectral_norm, weight_norm

# A logger for this file
logger = getLogger(__name__)


class PeriodDiscriminator(nn.Module):
    """
    HiFiGAN's period discriminator module.

    This discriminator operates over periodic patterns in the waveform by splitting the input into
    chunks based on a given period and applying convolutional layers with different downsampling scales.
    """

    def __init__(
        self,
        period: int,
        channels: int,
        kernel_sizes: Tuple[int, int],
        downsample_scales: List[int],
        max_downsample_channels: Optional[int] = 1024,
        use_weight_norm: Optional[bool] = True,
        use_spectral_norm: Optional[bool] = False,
    ) -> None:
        """
        Initialize the PeriodDiscriminator module.

        Args:
            period (int): Period to split the waveform for 2D convolution processing.
            channels (int): Number of initial channels in the convolution layers.
            kernel_sizes (Tuple[int, int]): Kernel sizes for the first and last convolution layers.
            downsample_scales (List[int]): List of downsampling factors for each layer.
            max_downsample_channels (int, optional): Maximum number of channels after downsampling (default: 1024).
            use_weight_norm (bool, optional): Whether to apply weight normalization to the convolution layers (default: True).
            use_spectral_norm (bool, optional): Whether to apply spectral normalization to the convolution layers (default: False).
        """
        super().__init__()
        assert len(kernel_sizes) == 2
        assert kernel_sizes[0] % 2 == 1, "Kernel size must be odd number."
        assert kernel_sizes[1] % 2 == 1, "Kernel size must be odd number."

        self.period = period
        self.convs = nn.ModuleList()
        in_channels = 1
        out_channels = channels
        for downsample_scale in downsample_scales:
            self.convs += [
                nn.Sequential(
                    nn.Conv2d(
                        in_channels,
                        out_channels,
                        (kernel_sizes[0], 1),
                        (downsample_scale, 1),
                        padding=((kernel_sizes[0] - 1) // 2, 0),
                    ),
                    nn.LeakyReLU(negative_slope=0.1),
                )
            ]
            in_channels = out_channels
            out_channels = min(out_channels * 4, max_downsample_channels)
        self.output_conv = nn.Conv2d(
            out_channels,
            1,
            (kernel_sizes[1], 1),
            padding=((kernel_sizes[1] - 1) // 2, 0),
        )

        if use_weight_norm and use_spectral_norm:
            raise ValueError("Either use use_weight_norm or use_spectral_norm.")

        # Apply weight norm
        if use_weight_norm:
            self.apply(weight_norm)

        # Apply spectral norm
        if use_spectral_norm:
            self.apply(spectral_norm)

    def forward(self, x: Tensor) -> Tuple[Tensor, List[Tensor]]:
        """
        Calculate forward propagation.

        Args:
            x (Tensor): Input waveforms with shape (batch, 1, length).

        Returns:
            Tensor: Discriminator output tensor.
            List[Tensor]: List of intermediate feature maps at each layer.
        """
        # Reshape the input from 1D to 2D (B, 1, T) -> (B, C, T/P, P)
        b, c, t = x.shape
        if t % self.period != 0:
            n_pad = self.period - (t % self.period)
            x = F.pad(x, (0, n_pad), "reflect")
            t += n_pad
        x = x.view(b, c, t // self.period, self.period)

        # Apply convolutional layers
        fmap = []
        for f in self.convs:
            x = f(x)
            fmap.append(x)
        x = self.output_conv(x)
        out = torch.flatten(x, 1, -1)

        return out, fmap


class MultiPeriodDiscriminator(nn.Module):
    """
    HiFi-GAN's multi-period discriminator module.

    This module contains multiple PeriodDiscriminators, each operating on a different period, to capture
    various periodic patterns in the waveform.
    """

    def __init__(self, periods: List[int], discriminator_params: Dict) -> None:
        """
        Initialize the HiFiGANMultiPeriodDiscriminator module.

        Args:
            periods (List[int]): List of periods for each period discriminator.
            discriminator_params (Dict): Common parameters for initializing the period discriminators.
        """
        super().__init__()
        self.discriminators = nn.ModuleList()
        for period in periods:
            params = copy.deepcopy(discriminator_params)
            params["period"] = period
            self.discriminators += [PeriodDiscriminator(**params)]

    def forward(self, x: Tensor) -> Tuple[List[Tensor], List[Tensor]]:
        """
        Calculate forward propagation.

        Args:
            x (Tensor): Input waveforms with shape (batch, 1, length).

        Returns:
            List[Tensor]: List of outputs from each period discriminator.
            List[Tensor]: List of feature maps from all discriminators.
        """
        outs, fmaps = [], []
        for f in self.discriminators:
            out, fmap = f(x)
            outs.append(out)
            fmaps.extend(fmap)

        return outs, fmaps


class SpectralDiscriminator(nn.Module):
    """
    UnivNet's spectral discriminator module.

    This module extracts features from the input waveform by converting it into the frequency domain via STFT
    and applies convolutional layers for feature extraction.
    """

    def __init__(
        self,
        fft_size: int,
        hop_size: int,
        win_length: int,
        window: str,
        channels: int,
        kernel_sizes: List[Tuple[int, int]],
        strides: List[Tuple[int, int]],
        use_weight_norm: Optional[bool] = True,
    ) -> None:
        """
        Initilize the SpectralDiscriminator module.

        Args:
            fft_size (int): Number of Fourier transform points for STFT.
            hop_size (int): Hop length (frameshift) in samples for STFT.
            win_length (int): Window length for STFT.
            window (str): Name of the window function for STFT.
            channels (int): Number of hidden channels.
            kernel_sizes (List[Tuple[int, int]]): List of kernel sizes for each convolutional layer.
            strides (List[Tuple[int, int]]): List of stride values for each convolutional layer.
            use_weight_norm (bool, optional): Whether to apply weight normalization to the convolutional layers (default: True).
        """
        super().__init__()
        self.fft_size = fft_size
        self.hop_size = hop_size
        self.win_length = win_length
        self.register_buffer("window", getattr(torch, window)(win_length))

        # Define convolutional layers
        self.input_conv = nn.Sequential(
            nn.Conv2d(1, channels, 1), nn.LeakyReLU(negative_slope=0.1)
        )
        self.convs = nn.ModuleList()
        for i in range(len(kernel_sizes)):
            self.convs += [
                nn.Sequential(
                    nn.Conv2d(
                        channels,
                        channels,
                        kernel_sizes[i],
                        padding=(kernel_sizes[i][0] // 2, kernel_sizes[i][1] // 2),
                        stride=strides[i],
                    ),
                    nn.LeakyReLU(negative_slope=0.1),
                )
            ]
        self.output_conv = nn.Conv2d(channels, 1, 1)

        # Apply weight norm
        if use_weight_norm:
            self.apply(weight_norm)

    def forward(self, x: Tensor) -> Tuple[Tensor, List[Tensor]]:
        """
        Calculate forward propagation.

        Args:
            x (Tensor): Input waveforms with shape (batch, 1, length).

        Returns:
            Tensor: Discriminator output tensor.
            List[Tensor]: List of intermediate feature maps at each layer.
        """
        x = torch.stft(
            x.squeeze(1),
            n_fft=self.fft_size,
            hop_length=self.hop_size,
            win_length=self.win_length,
            window=self.window,
            center=True,
            return_complex=True,
        ).abs()
        x = x.unsqueeze(1)

        x = self.input_conv(x)
        fmap = []
        for f in self.convs:
            x = f(x)
            fmap.append(x)
        x = self.output_conv(x)

        return x, fmap


class MultiResolutionDiscriminator(nn.Module):
    """
    UnivNet's multi-resolution spectral discriminator module.

    This module contains multiple spectral discriminators, each analyzing the input waveform at different resolutions.
    """

    def __init__(
        self,
        fft_sizes: List[int],
        hop_sizes: List[int],
        win_lengths: List[int],
        discriminator_params: Dict,
    ) -> None:
        """
        Initilize the UnivNetMultiResolutionDiscriminator module.

        Args:
            fft_sizes (List[int]): List of FFT sizes for each spectral discriminator.
            hop_sizes (List[int]): List of hop sizes for each spectral discriminator.
            win_lengths (List[int]): List of window lengths for each spectral discriminator.
            discriminator_params (Dict): Common parameters for initializing the spectral discriminators.
        """
        super().__init__()
        assert len(fft_sizes) == len(hop_sizes) == len(win_lengths)
        self.discriminators = nn.ModuleList()

        # Add discriminators
        for i in range(len(fft_sizes)):
            params = copy.deepcopy(discriminator_params)
            self.discriminators += [
                SpectralDiscriminator(
                    fft_size=fft_sizes[i],
                    hop_size=hop_sizes[i],
                    win_length=win_lengths[i],
                    **params,
                )
            ]

    def forward(self, x: Tensor) -> Tuple[List[Tensor], List[Tensor]]:
        """
        Calculate forward propagation.

        Args:
            x (Tensor): Input waveforms with shape (batch, 1, length).

        Returns:
            List[Tensor]: List of outputs from each period discriminator.
            List[Tensor]: List of feature maps from all discriminators.
        """
        outs, fmaps = [], []
        for f in self.discriminators:
            out, fmap = f(x)
            outs.append(out)
            fmaps.extend(fmap)

        return outs, fmaps


class MultiResolutionMultiPeriodDiscriminator(nn.Module):
    """
    UnivNet's combined discriminator module.

    This module combines the multi-resolution spectral discriminator and the multi-period discriminator,
    providing a comprehensive analysis of input waveforms by considering both time-domain and frequency-domain features.
    """

    def __init__(
        self,
        # Multi-period discriminator related
        periods: List[int],
        period_discriminator_params: Dict,
        # Multi-resolution discriminator related
        fft_sizes: List[int],
        hop_sizes: List[int],
        win_lengths: List[int],
        spectral_discriminator_params: Dict,
    ) -> None:
        """
        Initilize the MultiResolutionMultiPeriodDiscriminator module.

        Args:
            periods (List[int]): List of periods for the HiFi-GAN period discriminators.
            period_discriminator_params (Dict): Common parameters for initializing the period discriminators.
            fft_sizes (List[int]): List of FFT sizes for the spectral discriminators.
            hop_sizes (List[int]): List of hop sizes for the spectral discriminators.
            win_lengths (List[int]): List of window lengths for the spectral discriminators.
            window (str): Name of the window function.
            spectral_discriminator_params (Dict): Common parameters for initializing the spectral discriminators.
        """
        super().__init__()
        self.mpd = MultiPeriodDiscriminator(
            periods=periods,
            discriminator_params=period_discriminator_params,
        )
        self.mrd = MultiResolutionDiscriminator(
            fft_sizes=fft_sizes,
            hop_sizes=hop_sizes,
            win_lengths=win_lengths,
            discriminator_params=spectral_discriminator_params,
        )

    def forward(self, x: Tensor) -> Tuple[List[Tensor], List[Tensor]]:
        """
        Calculate forward propagation.

         Args:
            x (Tensor): Input waveforms with shape (batch, 1, length).

        Returns:
            List[Tensor]: List of outputs from each discriminator.
            List[Tensor]: List of feature maps from all discriminators.
        """
        mpd_outs, mpd_fmaps = self.mpd(x)
        mrd_outs, mrd_fmaps = self.mrd(x)
        outs = mpd_outs + mrd_outs
        fmaps = mpd_fmaps + mrd_fmaps

        return outs, fmaps
