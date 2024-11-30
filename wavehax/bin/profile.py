# Copyright 2024 Reo Yoneyama (Nagoya University)
#  MIT License (https://opensource.org/licenses/MIT)

"""
Profiling script for GAN-based vocoders.

This script profiles a GAN-based vocoder model by calculating its MACs (multiply-accumulate operations)
and counting the number of learnable parameters.
"""

import hydra
import torch
from omegaconf import DictConfig
from torchprofile import profile_macs

from wavehax.modules import remove_weight_norm


@hydra.main(version_base=None, config_path="config", config_name="profile")
def main(cfg: DictConfig) -> None:
    """Profile model parameters and MACs."""

    # Instantiate model
    model = hydra.utils.instantiate(cfg.generator)
    model.apply(remove_weight_norm)
    model.eval()

    # Generated waveform duration in seconds
    dur_in_sec = 1.0

    # Prepare dummy inputs
    num_frames = int(model.sample_rate / model.hop_length * dur_in_sec)
    cond = torch.randn(1, model.in_channels, num_frames)
    f0 = torch.ones(1, 1, num_frames)

    # Calculate MACs
    macs = profile_macs(model, (cond, f0))

    # Calculate learnable model parameters
    params = 0
    for p in model.parameters():
        if p.requires_grad:
            params += p.numel()

    print(f"Model class: {model.__class__.__name__}")
    print(f"Duration: {dur_in_sec} [sec]")
    print(f"MAC counts: {macs}")
    print(f"Parameters: {params}")


if __name__ == "__main__":
    main()
