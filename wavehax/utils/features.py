# Copyright 2024 Reo Yoneyama (Nagoya University)
#  MIT License (https://opensource.org/licenses/MIT)

"""
Functions for audio and feature processing.

References:
    - https://github.com/bigpon/QPPWG
"""

from logging import getLogger
from typing import Optional

import numpy as np
import pyloudnorm as pyln
from numpy import ndarray
from scipy.interpolate import interp1d

# A logger for this file
logger = getLogger(__name__)


def normalize_loudness(
    audio: ndarray, sample_rate: int, target_db: Optional[float] = -24.0
):
    """
    Normalizes the loudness of an input monaural audio signal.

    Args:
        audio (ndarray): Input audio waveform.
        sample_rate (int): Sampling frequency of the audio.
        target_db (float, optional): Target loudness in decibels (default: -24.0).

    Returns:
        ndarray: Loudness-normalized audio waveform.
    """
    meter = pyln.Meter(sample_rate)
    loudness = meter.integrated_loudness(audio)
    normed_audio = pyln.normalize.loudness(audio, loudness, target_db)
    return normed_audio


def fill_zeros_with_neighbors(arr: ndarray) -> ndarray:
    """
    Replaces zero values in the input array with the nearest non-zero values from neighboring indices.

    Args:
        arr (ndarray): Input array.

    Returns:
        ndarray: Array with zero values replaced by neighboring non-zero values.
    """
    new_arr = arr.copy()
    for i in range(1, len(arr)):
        if new_arr[i] == 0:
            new_arr[i] = new_arr[i - 1]
    for i in range(len(arr) - 1, 0, -1):
        if new_arr[i - 1] == 0:
            new_arr[i - 1] = new_arr[i]
    return new_arr


def convert_to_continuous_f0(f0: ndarray) -> ndarray:
    """
    Converts an F0 sequence with intermittent zero values into a continuous F0 array
    by linearly interpolating over non-zero values.

    Args:
        f0 (ndarray): Input F0 array with zero and non-zero values.

    Returns:
        ndarray: Continuous F0 array.
    """
    if f0.sum() == 0:
        return f0

    # Get start and end of f0
    start_f0 = f0[f0 != -1][0]
    end_f0 = f0[f0 != -1][-1]

    # Padding start and end of f0 sequence
    start_f0 = f0[f0 != 0][0]
    end_f0 = f0[f0 != 0][-1]
    start_idx = np.where(f0 == start_f0)[0][0]
    end_idx = np.where(f0 == end_f0)[0][-1]
    cf0 = f0.copy()
    cf0[:start_idx] = start_f0
    cf0[end_idx:] = end_f0

    # Get non-zero frame index
    nonzero_idxs = np.where(cf0 != 0)[0]

    # Perform linear interpolation
    interp_fn = interp1d(nonzero_idxs, cf0[nonzero_idxs])
    cf0 = interp_fn(np.arange(0, cf0.shape[0]))

    return cf0
