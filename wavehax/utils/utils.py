# Copyright 2024 Reo Yoneyama (Nagoya University)
#  MIT License (https://opensource.org/licenses/MIT)

"""
Utility functions for file I/O, dynamic module importing, etc.

References:
    - https://github.com/bigpon/QPPWG
"""

import os
import sys
from logging import getLogger
from typing import Any, Dict, List, Optional

import h5py
import numpy as np
import soundfile as sf
import yaml
from librosa import resample
from numpy import ndarray

logger = getLogger(__name__)


def dynamic_import(module_class: str) -> Any:
    """
    Dynamically imports a Python class from a module using its full module path.

    Args:
        module_class (str): Full module path in the format 'module.submodule.ClassName'.

    Returns:
        Any: The imported class object.
    """
    module_path, class_name = module_class.rsplit(".", 1)
    module = __import__(module_path, fromlist=[class_name])
    return getattr(module, class_name)


def read_yaml(file_path: str) -> Dict:
    """
    Reads a YAML configuration file and returns its contents as a dictionary.

    Args:
        file_path (str): Path to the YAML file.

    Returns:
        Dict: Parsed contents of the YAML file.
    """
    with open(file_path) as file:
        config = yaml.safe_load(file)
    return config


def read_audio(file_path: str, sample_rate: int) -> ndarray:
    """
    Reads an audio file, resamples it to the target sampling frequency if necessary,
    and returns the audio waveform as a numpy array.

    Args:
        file_path (str): Path to the audio file.
        sample_rate (int): Desired sampling frequency for resampling.

    Returns:
        ndarray: Audio waveform array.
    """
    audio, sr = sf.read(file_path, dtype="float32")

    assert (
        np.abs(audio).max() <= 1.0
    ), f"{file_path} seems to be different from 16 bit PCM."

    if len(audio.shape) != 1:
        logger.warning(f"{file_path} seems to be multi-channel signal {audio.shape}.")
        audio = audio.mean(axis=-1)

    if sr != sample_rate:
        logger.warning(f"Resample {file_path} from {sr} Hz to {sample_rate} Hz.")
        audio = resample(audio, orig_sr=sr, target_sr=sample_rate)

    return audio


def read_hdf5(hdf5_name: str, hdf5_path: str) -> Any:
    """
    Reads a dataset from an HDF5 file.

    Args:
        hdf5_name (str): Path to the HDF5 file.
        hdf5_path (str): Dataset path within the HDF5 file.

    Returns:
        Any: Dataset values from the HDF5 file.
    """
    if not os.path.exists(hdf5_name):
        logger.error(f"There is no such a hdf5 file ({hdf5_name}).")
        sys.exit(1)

    hdf5_file = h5py.File(hdf5_name, "r")

    if hdf5_path not in hdf5_file:
        logger.error(f"There is no data named {hdf5_path} in {hdf5_name}.")
        sys.exit(1)

    hdf5_data = hdf5_file[hdf5_path][()]
    hdf5_file.close()

    return hdf5_data


def write_hdf5(
    hdf5_name: str,
    hdf5_path: str,
    write_data: ndarray,
    is_overwrite: Optional[bool] = True,
) -> None:
    """
    Writes a dataset to an HDF5 file, optionally overwriting existing datasets.

    Args:
        hdf5_name (str): HDF5 file path.
        hdf5_path (str): Dataset path within the HDF5 file.
        write_data (ndarray): Data to write into the HDF5 file.
        is_overwrite (bool, optional): Whether to overwrite existing datasets (default: True).
    """
    # Convert to numpy array
    write_data = np.array(write_data)

    # Check folder existence
    folder_name, _ = os.path.split(hdf5_name)
    if not os.path.exists(folder_name) and len(folder_name) != 0:
        os.makedirs(folder_name)

    # Check hdf5 existence
    if os.path.exists(hdf5_name):
        # If already exists, open with r+ mode
        hdf5_file = h5py.File(hdf5_name, "r+")
        # Check dataset existence
        if hdf5_path in hdf5_file:
            if is_overwrite:
                hdf5_file.__delitem__(hdf5_path)
            else:
                logger.error(
                    "Dataset in hdf5 file already exists. "
                    "if you want to overwrite, please set is_overwrite = True."
                )
                hdf5_file.close()
                sys.exit(1)
    else:
        # If not exists, open with w mode
        hdf5_file = h5py.File(hdf5_name, "w")

    # Write data to hdf5
    hdf5_file.create_dataset(hdf5_path, data=write_data)
    hdf5_file.flush()
    hdf5_file.close()


def check_hdf5(hdf5_name: str, hdf5_path: str) -> bool:
    """
    Checks if a specified dataset exists in an HDF5 file.

    Args:
        hdf5_name (str): HDF5 file path.
        hdf5_path (str): Dataset path within the HDF5 file.

    Returns:
        bool: True if the dataset exists, False otherwise.
    """
    if not os.path.exists(hdf5_name):
        return False

    with h5py.File(hdf5_name, "r") as hdf5_file:
        return hdf5_path in hdf5_file


def read_txt(file_list: str) -> List[str]:
    """
    Read lines from a text file, removing newline characters.

    Args:
        file_list (str): Path to the text file containing filenames.

    Returns:
        List[str]: A list of filenames, with newline characters removed.
    """
    with open(file_list) as f:
        filenames = f.readlines()
    return [filename.replace("\n", "") for filename in filenames]


def check_filename(list1: List[str], list2: List[str]) -> bool:
    """
    Check if the filenames in two lists (without extensions) are identical.

    Args:
        list1 (List[str]): First list of file paths or names.
        list2 (List[str]): Second list of file paths or names.

    Returns:
        bool: True if the filenames (without extensions) in both lists match, False otherwise.
    """

    def _filename(x):
        return os.path.basename(x).split(".")[0]

    list1 = list(map(_filename, list1))
    list2 = list(map(_filename, list2))

    return list1 == list2


def validate_length(
    xs: List, ys: Optional[List] = None, hop_size: Optional[int] = None
) -> List:
    """
    Validates and adjusts the lengths of feature arrays and corresponding audio data
    for alignment during audio processing. If audio data is provided, their lengths
    are adjusted relative to the hop size.

    Args:
        xs (List): List of feature arrays in ndarray.
        ys (List, optional): List of audio arrays in ndarray (default: None).
        hop_size (int, optional): Frame shift in samples (default: None).

    Returns:
        List: A list of length-adjusted features and optionally audios if provided.
    """
    # Get minimum length for features and audios
    min_len_x = min([x.shape[0] for x in xs])
    if ys is not None:
        min_len_y = min([y.shape[0] for y in ys])
        if min_len_y < min_len_x * hop_size:
            min_len_x = min_len_y // hop_size
        if min_len_y > min_len_x * hop_size:
            min_len_y = min_len_x * hop_size
        ys = [y[:min_len_y] for y in ys]
    xs = [x[:min_len_x] for x in xs]

    return xs + ys if ys is not None else xs
