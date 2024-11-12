# Copyright 2024 Reo Yoneyama (Nagoya University)
#  MIT License (https://opensource.org/licenses/MIT)

"""
Acoustice feature extraction script.

This script extracts various acoustic features such as F0 (fundamental frequency), mel-spectrograms,
spectral envelopes, aperiodicities, and mel-generalized cepstra from a list of audio files.
The extracted features are saved in HDF5 format for further use in neural vocoder training or other tasks.


References:
    - https://github.com/kan-bayashi/ParallelWaveGAN
    - https://github.com/bigpon/QPPWG
    - https://github.com/k2kobayashi/sprocket
"""

import copy
import multiprocessing as mp
import os
from logging import getLogger
from typing import Dict, List, Optional, Tuple

import hydra
import numpy as np
import pysptk
import pyworld
import torch
from hydra.utils import to_absolute_path
from omegaconf import DictConfig, OmegaConf

from wavehax.modules import MelSpectrogram
from wavehax.utils import (
    convert_to_continuous_f0,
    read_audio,
    read_txt,
    read_yaml,
    write_hdf5,
)

# A logger for this file
logger = getLogger(__name__)


# All-pass-filter coefficients {key -> Sampling frequency : value -> coefficient}
ALPHA = {
    8000: 0.312,
    12000: 0.369,
    16000: 0.410,
    22050: 0.455,
    24000: 0.466,
    32000: 0.504,
    44100: 0.544,
    48000: 0.554,
}


def path_create(
    audio_paths: List[str], in_dir: str, out_dir: str, extname: str
) -> List[str]:
    """
    Create directories and prepare paths for feature files.

    Args:
        audio_paths (List[str]): List of input audio file paths.
        in_dir (str): Directory containing the input audio files.
        out_dir (str): Directory where the extracted features will be saved.
        extname (str): File extension for the output files (e.g., "h5" for HDF5 files).

    Returns:
        List[str]: List of paths where the extracted features will be saved.
    """
    for audio_path in audio_paths:
        path_replace(audio_path, in_dir, out_dir, extname=extname)


def path_replace(
    file_path: str, in_dir: str, out_dir: str, extname: Optional[str] = None
) -> str:
    """
    Modify the file path by replacing the input directory with the output directory,
    and optionally changing the file extension.

    Args:
        file_path (str): Original file path.
        in_dir (str): Input directory to be replaced in the path.
        out_dir (str): Output directory to be inserted in the path.
        extname (str, optional): New file extension without a dot (default: None).

    Returns:
        str: New file path with the updated directory and file extension.
    """
    if extname is not None:
        file_path = f"{os.path.splitext(file_path)[0]}.{extname}"
    file_path = file_path.replace(in_dir, out_dir)
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    return file_path


def spk_division(
    file_list: List[str], cfg: Dict, spk_info: Dict, split: Optional[str] = "/"
) -> Tuple[List[List[str]], Dict]:
    """
    Divide a list of audio files based on the speaker and prepare speaker-specific configurations.

    Args:
        file_list (List[str]): List of audio file paths.
        cfg (Dict): Configuration dictionary with default parameters.
        spk_info (Dict): Dictionary containing speaker-specific information such as F0 range.
        split (str, optional): Delimiter used to split file paths and extract speaker information (default: "/").

    Returns:
        Tuple[List[List[str]], Dict]:
            - List of file lists grouped by speaker.
            - List of speaker-specific configurations (modified copies of the original cfg).
    """
    file_lists, cfgs, tempf = [], [], []
    prespk = None
    for file in file_list:
        spk = file.split(split)[cfg.spkidx]
        if spk != prespk:
            if tempf:
                file_lists.append(tempf)
            tempf = []
            prespk = spk
            tempc = copy.deepcopy(cfg)
            if spk in spk_info:
                tempc["f0_min"] = spk_info[spk]["f0_min"]
                tempc["f0_max"] = spk_info[spk]["f0_max"]
            else:
                msg = f"Since {spk} is not in spk_info dict, "
                msg += "default f0 range and power threshold are used."
                logger.info(msg)
                tempc["f0_min"] = 70
                tempc["f0_max"] = 300
            cfgs.append(tempc)
        tempf.append(file)
    file_lists.append(tempf)

    return file_lists, cfgs


def feature_list_create(audio_scp: str, cfg: Dict) -> None:
    """
    Create a list file containing paths to feature files.

    Args:
        audio_scp (str): Path to the SCP file that lists input audio files.
        cfg (Dict): Configuration dictionary with input and output directory information.
    """
    feature_list_file = audio_scp.replace("scp/", "list/").replace(".scp", ".list")
    audio_paths = read_txt(audio_scp)
    with open(feature_list_file, "w") as f:
        for audio_path in audio_paths:
            feat_name = path_replace(
                audio_path,
                cfg.in_dir,
                cfg.out_dir,
                extname=cfg.feature_format,
            )
            f.write(f"{feat_name}\n")


def extract_acoustic_features(
    queue: mp.Queue, audio_paths: List[str], cfg: Dict
) -> None:
    """
    Extract various acoustic features (F0, mel-spectrogram, spectral envelope, etc.) from a list of WAV files.

    Args:
        queue (multiprocessing.Queue): Queue to communicate the status of the process.
        wav_paths (List[str]): List of paths to the WAV files for feature extraction.
        cfg (Dict): Configuration dictionary for feature extraction parameters.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Define mel-spectrogram extractor
    mel_extractor = MelSpectrogram(
        sample_rate=cfg.sample_rate,
        hop_length=cfg.hop_length,
        n_fft=cfg.n_fft,
        n_mels=cfg.n_mels,
        fmin=cfg.fmin,
        fmax=cfg.fmax,
    ).to(device)

    # Read speaker information
    spk_info = None
    if cfg.spk_info and os.path.exists(to_absolute_path(cfg.spk_info)):
        spk_info = read_yaml(to_absolute_path(cfg.spk_info))
    logger.info(f"Speaker information: {spk_info}.")

    # Feature extraction loop
    for i, audio_path in enumerate(audio_paths):
        # Check the existence of speaker and style
        f0_min, f0_max = cfg["f0_min"], cfg["f0_max"]
        if spk_info is not None:
            spk, style = audio_path.split("/")[-4:-2]
            if spk not in spk_info:
                logger.warning(f"{spk} of {audio_path} is not in {spk_info}.")
            elif style not in spk_info[spk]:
                logger.warning(f"{spk}/{style} of {audio_path} is not in {spk_info}.")
            else:
                f0_min = spk_info[spk][style]["f0_min"]
                f0_max = spk_info[spk][style]["f0_max"]

        # Load audio file (WORLD analyzer requires float64)
        x = read_audio(audio_path, cfg.sample_rate).astype(np.float64)

        # Extract F0
        f0, t = pyworld.harvest(
            x,
            fs=cfg.sample_rate,
            f0_floor=f0_min if f0_min > 0 else cfg["f0_min"],
            f0_ceil=f0_max if f0_max > 0 else cfg["f0_max"],
            frame_period=1000 * cfg.hop_length / cfg.sample_rate,
        )
        if f0_min <= 0 or f0_max <= 0:
            f0 *= 0.0

        # Extract spectral envelope and aperiodicity
        env = pyworld.cheaptrick(x, f0, t, fs=cfg.sample_rate, fft_size=cfg.fft_size)
        ap = pyworld.d4c(x, f0, t, fs=cfg.sample_rate, fft_size=cfg.fft_size)

        # Convert F0 to continuous F0 and voiced/unvoiced flags
        cf0 = convert_to_continuous_f0(np.copy(f0))
        lf0 = np.log(cf0 + 1.0)
        vuv = f0 != 0

        # Convert spectral envelope to mel-generalized cepstrum (MGC)
        mgc = pysptk.sp2mc(env, order=cfg.mgc_dim - 1, alpha=ALPHA[cfg.sample_rate])

        # Convert aperiodicity to mel-generalized cepstra and coded aperiodicity (MAP and CAP)
        map = pysptk.sp2mc(ap, order=cfg.map_dim - 1, alpha=ALPHA[cfg.sample_rate])
        bap = pyworld.code_aperiodicity(ap, cfg.sample_rate)

        # Extract mel-spectrogram (MEL)
        x = torch.tensor(x, dtype=torch.float, device=device).view(1, -1)
        mel = mel_extractor(x)[0].cpu().numpy().T

        # Prepare output dictionary
        features = {
            "f0": f0.astype(np.float32).reshape(-1, 1),
            "cf0": cf0.astype(np.float32).reshape(-1, 1),
            "lf0": lf0.astype(np.float32).reshape(-1, 1),
            "vuv": vuv.astype(np.float32).reshape(-1, 1),
            "mgc": mgc.astype(np.float32),
            "map": map.astype(np.float32),
            "bap": bap.astype(np.float32),
            "mel": mel.astype(np.float32),
        }

        # Save features to HDF5
        feat_path = to_absolute_path(
            path_replace(audio_path, cfg.in_dir, cfg.out_dir, extname="h5")
        )
        for key, value in features.items():
            write_hdf5(feat_path, key, value)
        logger.info(
            f"Processed {audio_path} and saved features to {feat_path} ({i + 1}/{len(audio_paths)})."
        )

    # Update queue progress
    queue.put("Finish")


@hydra.main(config_path="config", config_name="extract_features")
def main(cfg: DictConfig) -> None:
    logger.info(OmegaConf.to_yaml(cfg))

    # Read audio file list
    file_list = read_txt(to_absolute_path(cfg.audio_scp))
    logger.info(f"Number of utterances = {len(file_list)}")

    # Divide audio file list
    file_lists = np.array_split(file_list, 4)
    file_lists = [f_list.tolist() for f_list in file_lists]

    # Create feature list file
    feature_list_create(to_absolute_path(cfg.audio_scp), cfg)

    # Create folder
    path_create(file_list, cfg.in_dir, cfg.out_dir, cfg.feature_format)

    # Multi processing
    processes = []
    queue = mp.Queue()
    for file_list in file_lists:
        p = mp.Process(target=extract_acoustic_features, args=(queue, file_list, cfg))
        p.start()
        processes.append(p)

    # Wait for all process
    for p in processes:
        p.join()


if __name__ == "__main__":
    main()
