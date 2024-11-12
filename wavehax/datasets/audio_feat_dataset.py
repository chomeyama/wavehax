# Copyright 2024 Reo Yoneyama (Nagoya University)
#  MIT License (https://opensource.org/licenses/MIT)

"""
Audio and feature dataset modules.

These modules provide classes to handle datasets of audio and acoustic features.

References:
    - https://github.com/kan-bayashi/ParallelWaveGAN
"""

from logging import getLogger
from multiprocessing import Manager
from typing import Any, List, Optional

import numpy as np
from hydra.utils import to_absolute_path
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset

from wavehax.utils import (
    check_filename,
    read_audio,
    read_hdf5,
    read_txt,
    validate_length,
)

# A logger for this file
logger = getLogger(__name__)


class AudioFeatDataset(Dataset):
    """PyTorch compatible dataset for paired audio and acoustic features."""

    def __init__(
        self,
        sample_rate: int,
        hop_length: int,
        audio_list: str,
        feat_list: str,
        feat_names: List[str],
        use_continuous_f0: bool,
        scaler: StandardScaler,
        audio_length_threshold: Optional[int] = None,
        feat_length_threshold: Optional[int] = None,
        return_filename: Optional[bool] = False,
        allow_cache: Optional[bool] = False,
    ) -> None:
        """
        Initialize the AudioFeatDataset.

        Args:
            sample_rate (int): Sampling frequency of the audio.
            hop_length (int): Hop size for acoustic features.
            audio_list (str): Filepath to a list of audio files.
            feat_list (str): Filepath to a list of feature files.
            feat_names (List[str]): Names of auxiliary features to load.
            use_continuous_f0 (bool): Whether to use continuous F0 values.
            scaler (StandardScaler): A fitted scaler for feature normalization.
            audio_length_threshold (int, optional): Minimum audio length to include in the dataset (default: None).
            feat_length_threshold (int, optional): Minimum feature length to include in the dataset (default: None).
            return_filename (bool, optional): Whether to return filenames with the data (default: False).
            allow_cache (bool, optional): Whether to cache loaded data in memory for faster access (default: False).
        """

        # Load audio and feature files & check filename
        audio_files = read_txt(to_absolute_path(audio_list))
        feat_files = read_txt(to_absolute_path(feat_list))
        assert check_filename(audio_files, feat_files)

        # Filter by threshold
        if audio_length_threshold is not None:
            audio_lengths = [
                read_audio(to_absolute_path(f), sample_rate).shape[0]
                for f in audio_files
            ]
            idxs = [
                idx
                for idx in range(len(audio_files))
                if audio_lengths[idx] > audio_length_threshold
            ]
            if len(audio_files) != len(idxs):
                logger.warning(
                    f"Some files are filtered by audio length threshold "
                    f"({len(audio_files)} -> {len(idxs)})."
                )
            audio_files = [audio_files[idx] for idx in idxs]
            feat_files = [feat_files[idx] for idx in idxs]
        if feat_length_threshold is not None:
            f0_lengths = [
                read_hdf5(to_absolute_path(f), feat_names[0]).shape[0]
                for f in feat_files
            ]
            idxs = [
                idx
                for idx in range(len(feat_files))
                if f0_lengths[idx] > feat_length_threshold
            ]
            if len(feat_files) != len(idxs):
                logger.warning(
                    f"Some files are filtered by mel length threshold "
                    f"({len(feat_files)} -> {len(idxs)})."
                )
            audio_files = [audio_files[idx] for idx in idxs]
            feat_files = [feat_files[idx] for idx in idxs]

        # Check the number of files
        assert len(audio_files) != 0, f"${audio_list} is empty."
        assert (
            len(audio_files) == len(feat_files)
        ), f"Number of audio and features files are different ({len(audio_files)} vs {len(feat_files)})."

        self.sample_rate = sample_rate
        self.hop_length = hop_length
        self.audio_files = audio_files
        self.feat_files = feat_files
        self.feat_names = feat_names
        self.f0_type = "cf0" if use_continuous_f0 else "f0"
        self.scaler = scaler
        self.return_filename = return_filename
        self.allow_cache = allow_cache

        if allow_cache:
            # NOTE(kan-bayashi): Manager is need to share memory in dataloader with num_workers > 0
            self.manager = Manager()
            self.caches = self.manager.list()
            self.caches += [() for _ in range(len(audio_files))]

    def __getitem__(self, idx: int) -> List[Any]:
        """
        Get the specified index data.

        Args:
            idx (int): Index of the item.

        Returns:
            list: [filename (optional), audio waveform, auxiliary features, F0 sequence]
        """
        if self.allow_cache and len(self.caches[idx]) != 0:
            return self.caches[idx]

        # Load audio waveform
        audio = read_audio(to_absolute_path(self.audio_files[idx]), self.sample_rate)

        # Get auxiliary features
        feats = []
        for feat_name in self.feat_names:
            feat = read_hdf5(to_absolute_path(self.feat_files[idx]), feat_name)
            feat = self.scaler[feat_name].transform(feat)
            feats += [feat]
        feat = np.concatenate(feats, axis=1)

        # Get f0 sequence
        f0 = read_hdf5(to_absolute_path(self.feat_files[idx]), self.f0_type)
        feat, f0 = validate_length([feat, f0])

        items = [audio, feat, f0]
        if self.return_filename:
            items = [self.feat_files[idx]] + items

        if self.allow_cache:
            self.caches[idx] = items

        return items

    def __len__(self) -> int:
        """
        Return the number of items in the dataset.

        Returns:
            int: Dataset size.
        """
        return len(self.audio_files)


class FeatDataset(Dataset):
    """PyTorch compatible dataset for acoustic features."""

    def __init__(
        self,
        sample_rate: int,
        hop_length: int,
        feat_list: str,
        feat_names: List[str],
        use_continuous_f0: bool,
        scaler: StandardScaler,
        f0_factor: Optional[float] = 1.0,
        return_filename: Optional[bool] = False,
    ) -> None:
        """
        Initialize the FeatDataset.

        Args:
            sample_rate (int): Sampling frequency of the audio.
            hop_length (int): Hop size for acoustic features.
            feat_list (str): Filepath to a list of feature files.
            feat_names (List[str]): Names of auxiliary features to load.
            use_continuous_f0 (bool): Whether to use continuous F0 values.
            scaler (StandardScaler): A fitted scaler for feature normalization.
            f0_factor (float, optional): Scaling factor for the F0 values (default: [1.0]).
            return_filename (bool, optional): Whether to return filenames with the data (default: False).
        """
        self.sample_rate = sample_rate
        self.hop_length = hop_length
        self.feat_files = read_txt(to_absolute_path(feat_list))
        self.feat_names = feat_names
        self.f0_type = "cf0" if use_continuous_f0 else "f0"
        self.scaler = scaler
        self.f0_factor = f0_factor
        self.return_filename = return_filename

    def __getitem__(self, idx: int) -> List[Any]:
        """
        Get the specified index data.

        Args:
            idx (int): Index of the item.

        Returns:
            list: [filename (optional), auxiliary features, F0 sequence]
        """
        # Get auxiliary features
        feats = []
        for feat_name in self.feat_names:
            feat = read_hdf5(to_absolute_path(self.feat_files[idx]), feat_name)
            if feat_name in ["f0", "cf0"]:
                feat *= self.f0_factor
            feat = self.scaler[feat_name].transform(feat)
            feats += [feat]
        feat = np.concatenate(feats, axis=1)

        # Get F0 sequences
        f0 = read_hdf5(to_absolute_path(self.feat_files[idx]), self.f0_type)
        f0 = f0 * self.f0_factor
        feat, f0 = validate_length([feat, f0])

        items = [feat, f0]
        if self.return_filename:
            items = [self.feat_files[idx]] + items

        return items

    def __len__(self) -> int:
        """
        Return the number of items in the dataset.

        Returns:
            int: Dataset size.
        """
        return len(self.feat_files)
