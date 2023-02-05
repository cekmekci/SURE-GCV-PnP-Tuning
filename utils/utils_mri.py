import os
import glob
import torch
import numpy as np


def getMask(mask_path):
    """Returns the mask in Tensor format with 2 channels.

    Input:
    - mask_path: path to the mask. Here, the mask must be centralized.

    Output:
    - mask (tensor): Mask with the shape (im_size,im_size).
    """
    mask = np.fft.fftshift(np.load(mask_path))
    mask = torch.from_numpy(mask).type(torch.complex128)
    return mask

def F(x):
    """Takes orthonormal 2D DFT of a given complex 2D signal.

    Inputs:
    - x (tensor): Input tensor with the shape (im_size,im_size).

    Output:
    - out (tensor): Output tensor with the shape (im_size,im_size).
    """
    out = torch.fft.fftn(x, norm="ortho")
    return out

def Finverse(x):
    """Takes orthonormal 2D IDFT of a given complex 2D signal.

    Inputs:
    - x (tensor): Input tensor with the shape (im_size,im_size).

    Output:
    - out (tensor): Output tensor with the shape (im_size,im_size).
    """
    out = torch.fft.ifftn(x, norm="ortho")
    return out

def Fadjoint(x):
    """Takes orthonormal 2D IDFT of a given complex 2D signal.

    Inputs:
    - x (tensor): Input tensor with the shape (im_size,im_size).

    Output:
    - out (tensor): Output tensor with the shape (im_size,im_size).

    This is because adjoint operator the normalized DFT is the IDFT itself.
    """
    x_ifft = Finverse(x)
    return x_ifft

def Mask(x, mask):
    """Applies a given mask to the input.

    Inputs:
    - x (tensor): Input tensor with the shape (im_size,im_size).
    - mask (tensor): Mask with the shape (im_size, im_size).

    Note that the real and imaginary part of the mask must be same, and the mask
    must be non-centralized, i.e, the DC frequencies must not be located at the
    center.
    """
    measurements_zero_filled = mask * x
    return measurements_zero_filled

def Maskadjoint(x, mask):
    """Applies the adjoint of a given mask to the input.

    Inputs:
    - x (tensor): Input tensor with the shape (im_size,im_size).
    - mask (tensor): Mask with the shape (im_size, im_size).

    Note that the real and imaginary part of the mask must be same, and the mask
    must be non-centralized, i.e, the DC frequencies must not be located at the
    center.
    """
    zero_filled = mask * x
    return zero_filled

def A(x, mask):
    """Applies the forward operator on the input.

    Inputs:
    - x (tensor): Input tensor with the shape (im_size,im_size).
    - mask (tensor): Mask with the shape (im_size, im_size).

    A operator is the composition of Mask and orthonormal 2D DFT operators.
    """
    out = Mask(F(x), mask)
    return out

def Aadjoint(x, mask):
    """Applies the adjoint of the forward operator on the input.

    Inputs:
    - x (tensor): Input tensor with the shape (im_size,im_size).
    - mask (tensor): Mask with the shape (im_size, im_size).

    This operator is the adjoint of the A operator, which is equivalent to the
    composition of orthonormal 2D IDFT and adjoint of the Mask operator.
    """
    out = Fadjoint(Maskadjoint(x, mask))
    return out

def AHATIinverse(x, tau, mask):
    """Computes (A^H * A + tau * I)^(-1)(x) for MRI.

    Inputs:
    - x (tensor): Input tensor with the shape (im_size,im_size).
    - mask (tensor): Mask with the shape (im_size, im_size).
    """
    return Fadjoint(1 / (mask + tau) * F(x))
