import torch
import numpy as np

from numpy.fft import fft, fft2, ifft2, ifft, ifftshift, fftshift
from numpy.lib.stride_tricks import as_strided



def normal_pdf(length, sensitivity):
    return np.exp(-sensitivity * (np.arange(length) - length / 2)**2)


def cartesian_mask(shape, acc):
    sample_n=10
    centred=False
    """
    Sampling density estimated from implementation of kt FOCUSS
    shape: tuple - of form (..., nx, ny)
    acc: float - doesn't have to be integer 4, 8, etc..
    """
    N, Nx, Ny = int(np.prod(shape[:-2])), shape[-2], shape[-1]
    pdf_x = normal_pdf(Nx, 0.5/(Nx/10.)**2)
    lmda = Nx/(2.*acc)
    n_lines = int(Nx / acc)

    # add uniform distribution
    pdf_x += lmda * 1./Nx

    if sample_n:
        pdf_x[Nx//2-sample_n//2:Nx//2+sample_n//2] = 0
        pdf_x /= np.sum(pdf_x)
        n_lines -= sample_n

    mask = np.zeros((N, Nx))
    for i in range(N):
        idx = np.random.choice(Nx, n_lines, False, pdf_x)
        mask[i, idx] = 1

    if sample_n:
        mask[:, Nx//2-sample_n//2:Nx//2+sample_n//2] = 1

    size = mask.itemsize
    mask = as_strided(mask, (N, Nx, Ny), (size * Nx, size, 0))

    mask = mask.reshape(shape)

    if not centred:
        mask = ifftshift(mask, axes=(-1, -2))

    return mask


def CreateZeroFilledImageFn(fsimage,us_factor,mask_type):
    mask_fn= mask_type+'_mask'
    fs_kspace = np.fft.fft2(fsimage,norm='ortho')
    h,w = fsimage.shape
    mask = eval(mask_fn)((h,w),us_factor)
    us_kspace = fs_kspace*mask
    usimg = np.abs(np.fft.ifft2(us_kspace,norm='ortho'))
    return usimg