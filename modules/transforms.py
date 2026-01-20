import numpy as np
import pywt
from scipy.fftpack import dct, idct

def apply_dwt(image):
    coeffs = pywt.dwt2(image, 'haar')
    LL, (HL, LH, HH) = coeffs
    return LL, HL, LH, HH

def apply_idwt(LL, HL, LH, HH):
    return pywt.idwt2((LL, (HL, LH, HH)), 'haar')

# RENAME these to match your import error
def block_dct(block):
    return dct(dct(block.T, norm='ortho').T, norm='ortho')

def block_idct(block_dct):
    return idct(idct(block_dct.T, norm='ortho').T, norm='ortho')

def get_dct_blocks(subband, block_size=8):
    h, w = subband.shape
    blocks = []
    for i in range(0, h, block_size):
        for j in range(0, w, block_size):
            blocks.append(subband[i:i+block_size, j:j+block_size])
    return blocks

def rebuild_from_blocks(blocks, subband_shape, block_size=8):
    h, w = subband_shape
    new_subband = np.zeros(subband_shape)
    idx = 0
    for i in range(0, h, block_size):
        for j in range(0, w, block_size):
            new_subband[i:i+block_size, j:j+block_size] = blocks[idx]
            idx += 1
    return new_subband