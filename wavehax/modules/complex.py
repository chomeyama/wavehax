# Copyright 2024 Reo Yoneyama (Nagoya University)
#  MIT License (https://opensource.org/licenses/MIT)

"""
Modules for complex neural networks.

This module includes components and utility functions for implementing complex-valued neural networks,
such as complex convolutions and initialization methods for complex weights.
"""

import math
from logging import getLogger
from typing import Optional, Tuple, Type, Union

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor

# A logger for this file
logger = getLogger(__name__)

# Define a class derived from nn.Module
ModuleType = Type[nn.Module]


def complex_weight_init(weight: Tensor) -> Tensor:
    """
    Initialize weights for a complex-valued neural network layer.

    This function creates a complex-valued weight matrix with orthogonal initialization based on
    singular value decomposition (SVD) as described in the paper "Deep Complex Networks" (Trabelsi et al., 2017).

    Args:
        weight (Tensor): A tensor representing the weight matrix to be initialized.

    Returns:
        Tensor: A complex-valued tensor with the same shape as the input weight, with orthogonal real and imaginary components.

    Reference:
        - https://arxiv.org/abs/1705.09792
    """
    weight_shape = weight.size()
    nin, nout = weight_shape[:2]

    if len(weight_shape) > 2:
        bsz = np.prod(weight_shape[2:])
    else:
        bsz = 1  # for linear layer

    # Create two real-valued matrices for real and imaginary components
    real = torch.rand(bsz, nout, nin)
    imag = torch.rand(bsz, nout, nin)

    # Form a complex matrix using the real and imaginary parts
    weight_new = torch.complex(real, imag)

    # Apply Singular Value Decomposition (SVD)
    u, s, vh = torch.linalg.svd(weight_new, full_matrices=False)

    # Replace the singular values with an identity matrix to create orthogonal components
    diag_matrix = torch.eye(min(nin, nout), dtype=torch.cfloat, device=weight.device)
    weight_new = torch.matmul(torch.matmul(u, diag_matrix), vh)

    # Scale the weight matrix to ensure variance according to Equation 13 in the reference paper
    scale = torch.sqrt(torch.tensor(2.0 / (nin + nout), dtype=torch.cfloat))
    weight_new = scale * weight_new

    return weight_new.reshape(*weight.shape).to(torch.cfloat)


def complex_conv(
    real: Tensor, imag: Tensor, fn_real: ModuleType, fn_imag: ModuleType
) -> Tuple[Tensor, Tensor]:
    """
    Perform complex convolution by combining the real and imaginary components.

    The complex convolution is defined as:
    - real_output = real_weight * real_input - imag_weight * imag_input
    - imag_output = real_weight * imag_input + imag_weight * real_input

    Args:
        real (Tensor): Real part of the input tensor.
        imag (Tensor): Imaginary part of the input tensor.
        fn_real (ModuleType): Convolutional layer for the real component.
        fn_imag (ModuleType): Convolutional layer for the imaginary component.

    Returns:
        Tuple[Tensor, Tensor]: The real and imaginary parts of the output after the complex convolution.
    """
    out_real = fn_real(real) - fn_imag(imag)
    out_imag = fn_real(imag) + fn_imag(real)
    return out_real, out_imag


class ComplexConv1d(nn.Module):
    """
    A 1D complex convolutional layer module.

    This module performs a 1D convolution on complex-valued inputs by applying separate convolutions to
    the real and imaginary parts, followed by a combination to compute the resulting complex-valued output.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: Optional[int] = 1,
        padding: Optional[int] = 0,
        dilation: Optional[int] = 1,
        groups: Optional[int] = 1,
        bias: Optional[bool] = True,
        padding_mode: Optional[str] = "reflect",
    ) -> None:
        """
        Initialize the ComplexConv1d module.

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            kernel_size (int): Size of the convolving kernel.
            stride (int, optional): Stride of the convolution (default: 1).
            padding (int, optional): Padding added to both sides of the input (default: 0).
            dilation (int, optional): Spacing between kernel elements (default: 1).
            groups (int, optional): Number of blocked connections from input to output channels (default: 1).
            bias (bool, optional): If True, adds a learnable bias to the output (default: True).
            padding_mode (str, optional): Padding mode for convolution (default: 'reflect').
        """
        super().__init__()
        self.kernel_size = (kernel_size,)

        # Define real and imaginary convolution layers
        self.real_conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            dilation=dilation,
            stride=stride,
            padding=padding,
            padding_mode=padding_mode,
            groups=groups,
            bias=bias,
        )
        self.imag_conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            dilation=dilation,
            stride=stride,
            padding=padding,
            padding_mode=padding_mode,
            groups=groups,
            bias=bias,
        )

        # Initialize the weights using the complex initialization method
        weight = complex_weight_init(
            torch.empty_like(self.real_conv.weight, dtype=torch.cfloat)
        )
        self.real_conv.weight.data.copy_(weight.real)
        self.imag_conv.weight.data.copy_(weight.imag)

    def forward(self, real: Tensor, imag: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Calculate forward propagation.

        Args:
            real (Tensor): Real part of the input tensor with shape (batch, channels, length).
            imag (Tensor): Imaginary part of the input tensor with shape (batch, channels, length).

         Returns:
            Tuple[Tensor, Tensor]: Real and imaginary parts of the output tensor.
        """
        return complex_conv(real, imag, self.real_conv, self.imag_conv)


class ComplexConv2d(nn.Module):
    """
    A 2D complex convolutional layer module.

    This module performs a 2D convolution on complex-valued inputs by applying separate convolutions to
    the real and imaginary parts, followed by a combination to compute the resulting complex-valued output.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple[int, int]],
        stride: Optional[Union[int, Tuple[int, int]]] = 1,
        padding: Optional[Union[int, Tuple[int, int]]] = 0,
        dilation: Optional[Union[int, Tuple[int, int]]] = 1,
        groups: Optional[int] = 1,
        bias: Optional[bool] = True,
        padding_mode: Optional[str] = "reflect",
    ) -> None:
        """
        Initialize the ComplexConv2d module.

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            kernel_size (Union[int, Tuple[int, int]]): Size of the convolving kernel.
            stride (Union[int, Tuple[int, int]], optional): Stride of the convolution (default: 1).
            padding (Union[int, Tuple[int, int]], optional): Padding added to both sides of the input (default: 0).
            dilation (Union[int, Tuple[int, int]], optional): Spacing between kernel elements (default: 1).
            groups (int, optional): Number of blocked connections from input to output channels (default: 1).
            bias (bool, optional): If True, adds a learnable bias to the output (default: True).
            padding_mode (str, optional): Padding mode for convolution (default: 'reflect').
        """
        super().__init__()
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)
        self.kernel_size = kernel_size

        # Define real and imaginary convolution layers
        self.real_conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
            padding_mode=padding_mode,
        )
        self.imag_conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
            padding_mode=padding_mode,
        )

        # Initialize the weights using the complex initialization method
        weight = complex_weight_init(
            torch.empty_like(self.real_conv.weight, dtype=torch.cfloat)
        )
        self.real_conv.weight.data.copy_(weight.real)
        self.imag_conv.weight.data.copy_(weight.imag)

    def forward(self, real: Tensor, imag: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Calculate forward propagation.

        Args:
            real (Tensor): Real part of the input tensor with shape (batch, channels, height, width).
            imag (Tensor): Imaginary part of the input tensor with shape (batch, channels, height, width).

        Returns:
            Tuple[Tensor, Tensor]: Real and imaginary parts of the output tensor.
        """
        return complex_conv(real, imag, self.real_conv, self.imag_conv)


class ComplexActivation(nn.Module):
    """
    Applies an activation function to both the real and imaginary parts of a complex tensor.

    This class is a wrapper around a given activation function, ensuring that the same non-linearity is
    applied independently to the real and imaginary parts of complex inputs.
    """

    def __init__(self, activation: ModuleType) -> None:
        """
        Initialize the ComplexActivation module.

        Args:
            activation (ModuleType): The activation function to apply to the real and imaginary parts of the complex tensor.
        """
        super().__init__()
        self.activation = activation

    def forward(self, real: Tensor, imag: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Applies the activation function to the real and imaginary parts of the input complex tensor.

        Args:
            real (Tensor): The real part of the complex tensor. Can be of any shape.
            imag (Tensor): The imaginary part of the complex tensor. Must have the same shape as `real`.

        Returns:
            Tuple[Tensor, Tensor]: The real and imaginary parts after applying the activation function.
        """
        return self.activation(real), self.activation(imag)


class ComplexNormLayer(nn.Module):
    """
    A normalization module for complex tensors.

    This module normalizes complex-valued inputs by computing the covariance matrix between the real and imaginary components,
    and applying a transformation to whiten the input using the method from "Deep Complex Networks" (ICLR 2018).

    References:
        - https://openreview.net/forum?id=H1T2hmZAb
        - https://github.com/ChihebTrabelsi/deep_complex_networks
    """

    def __init__(
        self, channels: int, eps: Optional[float] = 1e-6, affine: Optional[bool] = True
    ) -> None:
        """
        Initialize the ComplexNormLayer module.

        Args:
            channels (int): Number of channels (features) in the input.
            eps (float, optional): A small constant added to the denominator for numerical stability (default: 1e-6).
            affine (bool, optional): If True, this module has learnable affine parameters (default: True).
        """
        super().__init__()
        self.channels = channels
        self.eps = eps
        self.affine = affine

        if self.affine:
            self.gamma_rr = nn.Parameter(torch.zeros(channels) + math.sqrt(0.5))
            self.gamma_ii = nn.Parameter(torch.zeros(channels) + math.sqrt(0.5))
            self.gamma_ri = nn.Parameter(torch.zeros(channels))
            self.beta_r = nn.Parameter(torch.zeros(channels))
            self.beta_i = nn.Parameter(torch.zeros(channels))

    def normalize(
        self,
        real: Tensor,
        imag: Tensor,
        dim: int,
        mean_r: Optional[Tensor] = None,
        mean_i: Optional[Tensor] = None,
        Vrr: Optional[Tensor] = None,
        Vii: Optional[Tensor] = None,
        Vri: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        """
        Normalize the real and imaginary parts of a complex tensor.

        Args:
            real (Tensor): The real part of the complex tensor with shape (batch, channels, ...).
            imag (Tensor): The imaginary part of the complex tensor with shape (batch, channels, ...).
            dim (int): The dimension along which statistics are calculated.
            mean_r (Tensor, optional): Precomputed mean of the real part. If None, it will be computed (default: None).
            mean_i (Tensor, optional): Precomputed mean of the imaginary part. If None, it will be computed (default: None).
            Vrr (Tensor, optional): Precomputed variance of the real part (default: None).
            Vii (Tensor, optional): Precomputed variance of the imaginary part (default: None).
            Vri (Tensor, optional): Precomputed covariance between real and imaginary parts (default: None).

        Returns:
            Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
                - Normalized real part
                - Normalized imaginary part
                - Mean of real part
                - Mean of imaginary part
                - Variance of real part
                - Variance of imaginary part
                - Covariance between real and imaginary parts
        """
        # Calculate means along dimensions to be reduced
        if mean_r is None:
            mean_r = real.mean(dim, keepdim=True)
            mean_i = imag.mean(dim, keepdim=True)

        # Centerize the input complex tensor
        real = real - mean_r
        imag = imag - mean_i

        # Calculate the covariance matrix
        if Vrr is None:
            Vrr = real.pow(2).mean(dim, keepdim=True) + self.eps
            Vii = imag.pow(2).mean(dim, keepdim=True) + self.eps
            Vri = (real * imag).mean(dim, keepdim=True) + self.eps

        # We require the covariance matrix's inverse square root. That first requires
        # square rooting, followed by inversion (I do this in that order because during
        # the computation of square root we compute the determinant we'll need for
        # inversion as well).

        # tau = Vrr + Vii = Trace. Guaranteed >= 0 because SPD
        tau = Vrr + Vii
        # delta = (Vrr * Vii) - (Vri ** 2) = Determinant. Guaranteed >= 0 because SPD
        delta = (Vrr * Vii) - (Vri**2)

        s = torch.sqrt(delta)  # Determinant of square root matrix
        t = torch.sqrt(tau + 2 * s)

        # The square root matrix could now be explicitly formed as
        #       [ Vrr+s Vri   ]
        # (1/t) [ Vir   Vii+s ]
        # https://en.wikipedia.org/wiki/Square_root_of_a_2_by_2_matrix
        # but we don't need to do this immediately since we can also simultaneously
        # invert. We can do this because we've already computed the determinant of
        # the square root matrix, and can thus invert it using the analytical
        # solution for 2x2 matrices
        #      [ A B ]             [  D  -B ]
        # inv( [ C D ] ) = (1/det) [ -C   A ]
        # http://mathworld.wolfram.com/MatrixInverse.html
        # Thus giving us
        #           [  Vii+s  -Vri   ]
        # (1/s)(1/t)[ -Vir     Vrr+s ]
        # So we proceed as follows:

        inverse_st = 1.0 / (s * t)
        Wrr = (Vii + s) * inverse_st
        Wii = (Vrr + s) * inverse_st
        Wri = -Vri * inverse_st

        # And we have computed the inverse square root matrix W = sqrt(V)!
        # Normalization. We multiply, x_normalized = W.x.

        # The returned result will be a complex standardized input
        # where the real and imaginary parts are obtained as follows:
        # real_normed = Wrr * real_centred + Wri * imag_centred
        # imag_normed = Wri * real_centred + Wii * imag_centred

        real = Wrr * real + Wri * imag
        imag = Wii * imag + Wri * real

        if self.affine:
            shape = [1, self.channels] + [1] * (real.ndim - 2)
            real = (
                self.gamma_rr.view(*shape) * real
                + self.gamma_ri.view(*shape) * imag
                + self.beta_r.view(*shape)
            )
            imag = (
                self.gamma_ri.view(*shape) * real
                + self.gamma_ii.view(*shape) * imag
                + self.beta_i.view(*shape)
            )

        return real, imag, mean_r, mean_i, Vrr, Vii, Vri


class ComplexBatchNorm1d(ComplexNormLayer):
    """
    A batch normalization module for 1D complex-valued tensors.

    This module applies batch normalization to both real and imaginary parts of complex data
    using a method adapted from Deep Complex Networks (ICLR 2018).

    References:
        - https://openreview.net/forum?id=H1T2hmZAb
        - https://github.com/ChihebTrabelsi/deep_complex_networks
    """

    def __init__(
        self,
        channels: int,
        eps: Optional[float] = 1e-6,
        affine: Optional[bool] = True,
        momentum: Optional[float] = 0.1,
        track_running_stats: Optional[bool] = True,
    ) -> None:
        """
        Initialize the ComplexBatchNorm1d module.

        Args:
            channels (int): Number of input channels.
            eps (float, optional): A small constant added to the denominator for numerical stability (default: 1e-6).
            affine (bool, optional): If True, this module has learnable affine parameters (default: True).
            momentum (float, optional): The value used for the running_mean and running_var computation.
                Can be set to None for cumulative moving average, i.e. simple average (default: None).
            track_running_stats (bool, optional): If True, tracks running mean and variance during training.
        """
        super().__init__(channels, eps, affine)
        self.momentum = momentum
        self.track_running_stats = track_running_stats
        if track_running_stats:
            self.register_buffer(
                "num_batches_tracked", torch.tensor(0, dtype=torch.long)
            )
            self.register_buffer("running_mean_r", torch.zeros(1, channels, 1))
            self.register_buffer("running_mean_i", torch.zeros(1, channels, 1))
            self.register_buffer(
                "running_cov_rr", torch.zeros(1, channels, 1) + math.sqrt(0.5)
            )
            self.register_buffer(
                "running_cov_ii", torch.zeros(1, channels, 1) + math.sqrt(0.5)
            )
            self.register_buffer("running_cov_ri", torch.zeros(1, channels, 1))
        self.reduced_dim = [0, 2]

    def forward(self, real: Tensor, imag: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Apply batch normalization to the input complex-valued tensor.

        Args:
            real (Tensor): Real part of the input tensor with shape (batch, channels, length).
            imag (Tensor): Imaginary part of the input tensor with shape (batch, channels, length).

        Returns:
            Tuple[Tensor, Tensor]: Normalized real and imaginary parts of the tensor.
        """
        # Get the running statistics if needed
        if (not self.training) and self.track_running_stats:
            mean_r = self.running_mean_r
            mean_i = self.running_mean_i
            Vrr = self.running_cov_rr + self.eps
            Vii = self.running_cov_ii + self.eps
            Vri = self.running_cov_ri + self.eps
        else:
            mean_r = mean_i = Vrr = Vii = Vri = None

        real, imag, mean_r, mean_i, Vrr, Vii, Vri = self.normalize(
            real,
            imag,
            dim=self.reduced_dim,
            mean_r=mean_r,
            mean_i=mean_i,
            Vrr=Vrr,
            Vii=Vii,
            Vri=Vri,
        )

        # Update the running statistics
        if self.training and self.track_running_stats:
            with torch.no_grad():
                # Update the number of tracked samples
                self.num_batches_tracked += 1

                # Get the weight for cumulative or exponential moving average
                if self.momentum is None:
                    exponential_average_factor = 1.0 / float(self.num_batches_tracked)
                else:
                    exponential_average_factor = self.momentum

                # Update the running mean and covariance matrix
                self.running_mean_r = (
                    exponential_average_factor * mean_r
                    + (1 - exponential_average_factor) * self.running_mean_r
                )
                self.running_mean_i = (
                    exponential_average_factor * mean_i
                    + (1 - exponential_average_factor) * self.running_mean_i
                )
                n = real.numel() / real.size(1)
                self.running_cov_rr = (
                    exponential_average_factor * Vrr * n / (n - 1)
                    + (1 - exponential_average_factor) * self.running_cov_rr
                )
                self.running_cov_ii = (
                    exponential_average_factor * Vii * n / (n - 1)
                    + (1 - exponential_average_factor) * self.running_cov_ii
                )
                self.running_cov_ri = (
                    exponential_average_factor * Vri * n / (n - 1)
                    + (1 - exponential_average_factor) * self.running_cov_ri
                )

        return real, imag


class ComplexLayerNorm1d(ComplexNormLayer):
    """A layer normalization module for 1D complex tensors."""

    def __init__(
        self, channels: int, eps: Optional[float] = 1e-6, affine: Optional[bool] = True
    ) -> None:
        """
        Initialize the ComplexLayerNorm1d module.

        Args:
            channels (int): Number of input channels (features).
            eps (float, optional): A small constant added to the denominator for numerical stability (default: 1e-6).
            affine (bool, optional): If True, this module has learnable affine parameters (default: True).
        """
        super().__init__(channels, eps, affine)
        self.reduced_dim = [1, 2]

    def forward(self, real: Tensor, imag: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Apply layer normalization to the input complex tensor.

        Args:
            real (Tensor): The real part of the input tensor with shape (batch, channels, length).
            imag (Tensor): The imaginary part of the input tensor with shape (batch, channels, length).

        Returns:
            Tuple[Tensor, Tensor]: The normalized real and imaginary parts of the complex tensor.
        """
        real, imag, *_ = self.normalize(real, imag, dim=self.reduced_dim)
        return real, imag


class ComplexBatchNorm2d(ComplexBatchNorm1d):
    """
    A batch normalization module for 2D complex-valued tensors.

    This module applies batch normalization to both real and imaginary parts of complex data
    using a method adapted from Deep Complex Networks (ICLR 2018).

    References:
        - https://openreview.net/forum?id=H1T2hmZAb
        - https://github.com/ChihebTrabelsi/deep_complex_networks
    """

    def __init__(
        self,
        channels: int,
        eps: Optional[float] = 1e-6,
        affine: Optional[bool] = True,
        momentum: Optional[float] = 0.1,
        track_running_stats: Optional[bool] = True,
    ) -> None:
        """
        Initialize the ComplexBatchNorm2d module.

        Args:
            channels (int): Number of input channels.
            eps (float, optional): A small constant added to the denominator for numerical stability (default: 1e-6).
            affine (bool, optional): If True, this module has learnable affine parameters (default: True).
            momentum (float, optional): The value used for the running_mean and running_var computation.
                Can be set to None for cumulative moving average, i.e. simple average (default: None).
            track_running_stats (bool, optional): If True, tracks running mean and variance during training.
        """
        super().__init__(channels, eps, affine)
        self.momentum = momentum
        self.track_running_stats = track_running_stats
        if track_running_stats:
            self.register_buffer(
                "num_batches_tracked", torch.tensor(0, dtype=torch.long)
            )
            self.register_buffer("running_mean_r", torch.zeros(1, channels, 1, 1))
            self.register_buffer("running_mean_i", torch.zeros(1, channels, 1, 1))
            self.register_buffer(
                "running_cov_rr", torch.zeros(1, channels, 1, 1) + math.sqrt(0.5)
            )
            self.register_buffer(
                "running_cov_ii", torch.zeros(1, channels, 1, 1) + math.sqrt(0.5)
            )
            self.register_buffer("running_cov_ri", torch.zeros(1, channels, 1, 1))
        self.reduced_dim = [0, 2, 3]

    def forward(self, real: Tensor, imag: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Apply batch normalization to the input complex-valued tensor.

        Args:
            real (Tensor): Real part of the input tensor with shape (batch, channels, height, width).
            imag (Tensor): Imaginary part of the input tensor with shape (batch, channels, height, width).

        Returns:
            Tuple[Tensor, Tensor]: Normalized real and imaginary parts of the tensor.
        """
        return super().forward(real, imag)


class ComplexLayerNorm2d(ComplexNormLayer):
    """A layer normalization module for 2D complex tensors."""

    def __init__(
        self, channels: int, eps: Optional[float] = 1e-6, affine: Optional[bool] = True
    ) -> None:
        """
        Initialize the ComplexLayerNorm2d module.

        Args:
            channels (int): Number of input channels.
            eps (float, optional): A small constant added to the denominator for numerical stability (default: 1e-6).
            affine (bool, optional): If True, this module has learnable affine parameters (default: True).
        """
        super().__init__(channels, eps, affine)
        self.reduced_dim = [1, 2, 3]

    def forward(self, real: Tensor, imag: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Apply layer normalization to the input complex-valued tensor.

        Args:
            real (Tensor): Real part of the input tensor with shape (batch, channels, height, width).
            imag (Tensor): Imaginary part of the input tensor with shape (batch, channels, height, width).

        Returns:
            Tuple[Tensor, Tensor]: Normalized real and imaginary parts of the tensor.
        """
        real, imag, *_ = self.normalize(real, imag, dim=self.reduced_dim)
        return real, imag
