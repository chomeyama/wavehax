# Copyright 2024 Reo Yoneyama (Nagoya University)
#  MIT License (https://opensource.org/licenses/MIT)

"""
Statistics computing script for feature normalization.

This script computes the mean and variance of various acoustic features for normalization
purposes, commonly used in neural vocoder training. It updates or creates feature-specific
scalers based on the provided list of feature files.

References:
    - https://github.com/kan-bayashi/ParallelWaveGAN
    - https://github.com/bigpon/QPPWG
"""

import os
from logging import getLogger

import hydra
import numpy as np
from hydra.utils import to_absolute_path
from joblib import dump, load
from omegaconf import DictConfig, OmegaConf
from sklearn.preprocessing import StandardScaler

from wavehax.utils import read_hdf5, read_txt

logger = getLogger(__name__)


def compute_statistics(cfg: DictConfig):
    """
    Compute and save statistics (mean and variance) for normalization of acoustic features.

    This function processes each feature specified in the configuration, calculating
    statistics in an online fashion using `partial_fit` to handle large datasets.

    Args:
        cfg (DictConfig): Configuration containing feature names, file paths, and save directory.

    Workflow:
        - Load or initialize scalers for each feature.
        - Read feature data from the provided HDF5 files.
        - Update the scalers with the new data.
        - Save the updated scalers to a specified path.

    Raises:
        Warning: If a feature array has length 0, a warning is logged and the feature is skipped.
    """
    # Define scalers
    scaler = load(cfg.save_path) if os.path.isfile(cfg.save_path) else {}
    for feat_name in cfg.feat_names:
        scaler[feat_name] = StandardScaler()

    # Get feature paths
    feat_paths = read_txt(to_absolute_path(cfg.filepath_list))
    logger.info(f"Number of utterances = {len(feat_paths)}.")

    # Perform online calculation
    for file_path in feat_paths:
        for feat_name in cfg.feat_names:
            feat = read_hdf5(to_absolute_path(file_path), feat_name)
            if feat_name == "f0":
                feat = np.expand_dims(feat[feat > 0], axis=-1)
            if feat.shape[0] == 0:
                logger.warning(f"Feature length is 0 {file_path}, {feat_name}")
                continue
            scaler[feat_name].partial_fit(feat)

    # Save the computed statistics
    save_path = to_absolute_path(cfg.save_path)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    dump(scaler, save_path)
    logger.info(f"Successfully saved statistics to {cfg.save_path}.")


@hydra.main(version_base=None, config_path="config", config_name="compute_statistics")
def main(cfg: DictConfig):
    logger.info(OmegaConf.to_yaml(cfg))
    compute_statistics(cfg)


if __name__ == "__main__":
    main()
