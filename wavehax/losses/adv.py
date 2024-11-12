"""
Adversarial loss modules.

References:
    - https://github.com/kan-bayashi/ParallelWaveGAN
"""

from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class AdversarialLoss(nn.Module):
    """Module for calculating adversarial loss in GANs."""

    def __init__(
        self,
        average_by_discriminators: Optional[bool] = False,
        loss_type: Optional[str] = "mse",
    ) -> None:
        """
        Initialize the AdversarialLoss module.

        Args:
            average_by_discriminators (bool, optional): If True, the loss is averaged over the number of discriminators (default: False).
            loss_type (str, optional): Type of GAN loss to use, either "mse" or "hinge" (default: "mse").
        """
        super().__init__()
        assert loss_type.lower() in ["mse", "hinge"], f"{loss_type} is not supported."
        self.average_by_discriminators = average_by_discriminators

        if loss_type == "mse":
            self.adv_criterion = self._mse_adv_loss
            self.fake_criterion = self._mse_fake_loss
            self.real_criterion = self._mse_real_loss
        else:
            self.adv_criterion = self._hinge_adv_loss
            self.fake_criterion = self._hinge_fake_loss
            self.real_criterion = self._hinge_real_loss

    def forward(
        self, p_fakes: List[Tensor], p_reals: Optional[List[Tensor]] = None
    ) -> Union[Tensor, Tuple[Tensor, Tensor]]:
        """
        Calculate adversarial loss for both generator and discriminator.

        Args:
            p_fakes (List[Tensor]): List of discriminator outputs from the generated data.
            p_reals (List[Tensor], optional): List of discriminator outputs from real data.
                If not provided, only generator loss is computed (default: None).

        Returns:
            Tensor: Generator adversarial loss.
            If p_reals is provided:
                Tuple[Tensor, Tensor]: Fake and real discriminator loss values.
        """
        # Generator adversarial loss
        if p_reals is None:
            adv_loss = 0.0
            for p_fake in p_fakes:
                adv_loss += self.adv_criterion(p_fake)

            if self.average_by_discriminators:
                adv_loss /= len(p_fakes)

            return adv_loss

        # Discriminator adversarial loss
        else:
            fake_loss, real_loss = 0.0, 0.0
            for p_fake, p_real in zip(p_fakes, p_reals):
                fake_loss += self.fake_criterion(p_fake)
                real_loss += self.real_criterion(p_real)

            if self.average_by_discriminators:
                fake_loss /= len(p_fakes)
                real_loss /= len(p_reals)

            return fake_loss, real_loss

    def _mse_adv_loss(self, x: Tensor) -> Tensor:
        """Calculate MSE loss for generator."""
        return F.mse_loss(x, x.new_ones(x.size()))

    def _mse_real_loss(self, x: Tensor) -> Tensor:
        """Calculate MSE loss for real samples."""
        return F.mse_loss(x, x.new_ones(x.size()))

    def _mse_fake_loss(self, x: Tensor) -> Tensor:
        """Calculate MSE loss for fake samples."""
        return F.mse_loss(x, x.new_zeros(x.size()))

    def _hinge_adv_loss(self, x: Tensor) -> Tensor:
        """Calculate hinge loss for generator."""
        return -x.mean()

    def _hinge_real_loss(self, x: Tensor) -> Tensor:
        """Calculate hinge loss for real samples."""
        return -torch.mean(torch.min(x - 1, x.new_zeros(x.size())))

    def _hinge_fake_loss(self, x: Tensor) -> Tensor:
        """Calculate hinge loss for fake samples."""
        return -torch.mean(torch.min(-x - 1, x.new_zeros(x.size())))


class FeatureMatchingLoss(nn.Module):
    """Module for feature matching loss in GANs, comparing latent features."""

    def __init__(self, average_by_layers: Optional[bool] = False) -> None:
        """
        Initialize the FeatureMatchingLoss module.

        Args:
            average_by_layers (bool, optional): If True, the loss is averaged over the number of layers (default: False).
        """
        super().__init__()
        self.average_by_layers = average_by_layers

    def forward(self, fmaps_fake: List[Tensor], fmaps_real: List[Tensor]) -> Tensor:
        """
        Calculate feature matching loss.

        Args:
            fmaps_fake (List[Tensor]): List of discriminator's latent features from generated data.
            fmaps_real (List[Tensor]): List of discriminator's latent features from real data.

        Returns:
            Tensor: The feature matching loss value.
        """
        assert len(fmaps_fake) == len(fmaps_real)

        fm_loss = 0.0
        for feat_fake, feat_real in zip(fmaps_fake, fmaps_real):
            fm_loss += F.l1_loss(feat_fake, feat_real.detach())

        if self.average_by_layers:
            fm_loss /= len(fmaps_fake)

        return fm_loss
