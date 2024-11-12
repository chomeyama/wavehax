"""Spectral loss modules."""

from typing import Optional

import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from wavehax.modules import MelSpectrogram


class MelSpectralLoss(nn.Module):
    """Module for calculating L1 loss on Mel-spectrograms."""

    def __init__(
        self,
        sample_rate: int,
        hop_length: int,
        n_fft: int,
        n_mels: int,
        window: Optional[str] = "hann_window",
        fmin: Optional[float] = 0,
        fmax: Optional[float] = None,
    ) -> None:
        """
        Initialize the MelSpectralLoss module.

        Args:
            sample_rate (int): Sampling frequency of input waveforms.
            hop_length (int): Hop length (frameshift) in samples.
            n_fft (int): Number of Fourier transform points (FFT size).
            n_mels (int): Number of mel basis.
            window (str, optional): Name of the window function (default: "hann_window).
            fmin (float, optional): Minimum frequency for mel-filter bank (default: 0).
            fmax (float, optional): Maximum frequency for mel-filter bank (default: None).
        """
        super().__init__()
        self.mel_extractor = MelSpectrogram(
            sample_rate=sample_rate,
            hop_length=hop_length,
            n_fft=n_fft,
            n_mels=n_mels,
            window=window,
            fmin=fmin,
            fmax=fmax,
        )

    def forward(self, x: Tensor, y: Tensor) -> Tensor:
        """
        Calculate the L1 loss between mel-spectrograms of the generated and target waveforms.

        Args:
            x (Tensor): Generated audio waveform with shape (batch, samples) or (batch, 1, samples).
            y (Tensor): Targetaudio waveform with shape (batch, samples) or (batch, 1, samples).

        Returns:
            Tensor: Mel-spectral L1 loss value.
        """
        x_log_mel = self.mel_extractor(x)
        y_log_mel = self.mel_extractor(y)
        loss = F.l1_loss(x_log_mel, y_log_mel)

        return loss
