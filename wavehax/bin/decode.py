# Copyright 2024 Reo Yoneyama (Nagoya University)
#  MIT License (https://opensource.org/licenses/MIT)

"""
Decoding script for GAN-based vocoders.

This script performs inference for GAN-based vocoders, loading model checkpoints and feature data,
and generating audio waveforms based on the provided acoustic features.

References:
    - https://github.com/kan-bayashi/ParallelWaveGAN
"""

import os
from logging import getLogger
from time import time

import hydra
import numpy as np
import soundfile as sf
import torch
from hydra.utils import instantiate, to_absolute_path
from joblib import load
from omegaconf import DictConfig
from tqdm import tqdm

from wavehax.datasets import FeatDataset
from wavehax.modules import remove_weight_norm

# A logger for this file
logger = getLogger(__name__)


@hydra.main(version_base=None, config_path="config", config_name="decode")
def main(cfg: DictConfig) -> None:
    """
    Run the decoding process to generate audio waveforms from acoustic features.

    This function:
    - Loads a pre-trained GAN-based vocoder model.
    - Loads and scales feature data using a StandardScaler.
    - Decodes the features to generate corresponding audio waveforms.
    - Saves the generated audio as PCM 16-bit WAV files.

    Args:
        cfg (DictConfig): Configuration object loaded via Hydra.
    """
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    torch.cuda.manual_seed(cfg.seed)
    os.environ["PYTHONHASHSEED"] = str(cfg.seed)
    torch.set_num_threads(cfg.num_threads)
    logger.info(f"Number of threads: {cfg.num_threads}.")

    # Set device for computation (CPU or GPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Decode on {device}.")

    # Load scaler for normalizing features
    scaler = load(to_absolute_path(cfg.data.stats))

    # Load model parameters from checkpoint
    if cfg.ckpt_path is None:
        ckpt_path = os.path.join(
            cfg.out_dir,
            "checkpoints",
            f"checkpoint-{cfg.ckpt_steps}steps.pkl",
        )
    else:
        ckpt_path = cfg.ckpt_path
    assert os.path.exists(ckpt_path), f"Checkpoint file {ckpt_path} does not exist!"
    logger.info(f"Load model parameters from {ckpt_path}.")
    state_dict = torch.load(to_absolute_path(ckpt_path), map_location="cpu")
    state_dict = state_dict["model"]["generator"]

    # Instantiate and prepare the generator model
    model = instantiate(cfg.generator)
    model.load_state_dict(state_dict)
    model.apply(remove_weight_norm)
    model.eval().to(device)

    # Prepare output directory for saving generated waveforms
    out_dir = to_absolute_path(os.path.join(cfg.out_dir, cfg.tag, str(cfg.ckpt_steps)))
    logger.info(f"Save output waveforms to {out_dir}.")
    os.makedirs(out_dir, exist_ok=True)

    # Get hop length from the model
    if hasattr(model, "hop_length"):
        hop_length = model.hop_length
    elif hasattr(model, "upsample_scales"):
        hop_length = np.prod(model.upsample_scales)

    total_rtf = 0.0  # Real-time factor tracker

    # Perform inference for each F0 scaling factor
    for f0_factor in cfg.f0_factors:
        # Prepare the dataset
        dataset = FeatDataset(
            scaler=scaler,
            feat_list=cfg.data.eval_feat,
            sample_rate=model.sample_rate,
            hop_length=hop_length,
            feat_names=cfg.data.feat_names,
            use_continuous_f0=cfg.data.use_continuous_f0,
            f0_factor=f0_factor,
            return_filename=True,
        )
        logger.info(f"The number of features to be decoded = {len(dataset)}.")

        with torch.no_grad(), tqdm(dataset, desc="[decode]") as pbar:
            for feat_path, c, f0 in pbar:
                c = torch.FloatTensor(c.T).unsqueeze(0).to(device)
                f0 = torch.FloatTensor(f0).view(1, 1, -1).to(device)

                # Perform waveform generation
                start = time()
                y = model.inference(c, f0)
                rtf = (time() - start) / (y.size(-1) / model.sample_rate)
                pbar.set_postfix({"RTF": rtf})
                total_rtf += rtf

                y = y.view(-1).cpu().numpy()

                # Save generated waveform as a WAV file
                utt_id = os.path.splitext(os.path.basename(feat_path))[0]
                if "jvs" in feat_path:
                    spk_id, style_id = feat_path.split("/")[-4:-2]
                    save_path = os.path.join(
                        out_dir, f"{spk_id}_{style_id}_{utt_id}.wav"
                    )
                else:
                    save_path = os.path.join(out_dir, utt_id + ".wav")
                y = np.clip(y, -1, 1)
                sf.write(save_path, y, model.sample_rate, "PCM_16")

            # Report average real-time factor
            average_rtf = total_rtf / len(dataset)
            logger.info(
                f"Finished generation of {len(dataset)} utterances (RTF = {average_rtf:.6f})."
            )


if __name__ == "__main__":
    main()
