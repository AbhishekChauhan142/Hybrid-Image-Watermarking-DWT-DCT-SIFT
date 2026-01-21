import numpy as np


def get_dct_matrix(N=8):
    C = np.zeros((N, N))
    for i in range(N):
        alpha = np.sqrt(1 / N) if i == 0 else np.sqrt(2 / N)
        for j in range(N):
            C[i, j] = alpha * np.cos((np.pi * (2 * j + 1) * i) / (2 * N))
    return C


DCT_8_MAT = get_dct_matrix(8)


def block_dct(block):
    return np.dot(np.dot(DCT_8_MAT, block), DCT_8_MAT.T)


def block_idct(block_dct):
    return np.dot(np.dot(DCT_8_MAT.T, block_dct), DCT_8_MAT)


def apply_dwt(image):
    h, w = image.shape
    # Horizontal Sum/Diff
    row_L = (image[:, 0::2] + image[:, 1::2]) / np.sqrt(2)
    row_H = (image[:, 0::2] - image[:, 1::2]) / np.sqrt(2)
    row_combined = np.hstack((row_L, row_H))
    # Vertical Sum/Diff
    col_L = (row_combined[0::2, :] + row_combined[1::2, :]) / np.sqrt(2)
    col_H = (row_combined[0::2, :] - row_combined[1::2, :]) / np.sqrt(2)

    half_h, half_w = h // 2, w // 2
    return col_L[:, :half_w], col_L[:, half_w:], col_H[:, :half_w], col_H[:, half_w:]


def apply_idwt(LL, HL, LH, HH):
    h_half, w_half = LL.shape
    h, w = h_half * 2, w_half * 2
    col_combined_L = np.hstack((LL, HL))
    col_combined_H = np.hstack((LH, HH))

    row_combined = np.zeros((h, w))
    row_combined[0::2, :] = (col_combined_L + col_combined_H) / np.sqrt(2)
    row_combined[1::2, :] = (col_combined_L - col_combined_H) / np.sqrt(2)

    image = np.zeros((h, w))
    image[:, 0::2] = (row_combined[:, :w_half] + row_combined[:, w_half:]) / np.sqrt(2)
    image[:, 1::2] = (row_combined[:, :w_half] - row_combined[:, w_half:]) / np.sqrt(2)
    return image


def get_dct_blocks(subband, block_size=8):
    h, w = subband.shape
    return [subband[i:i + block_size, j:j + block_size]
            for i in range(0, h, block_size) for j in range(0, w, block_size)]


def rebuild_from_blocks(blocks, subband_shape, block_size=8):
    h, w = subband_shape
    new_subband = np.zeros(subband_shape)
    idx = 0
    for i in range(0, h, block_size):
        for j in range(0, w, block_size):
            new_subband[i:i + block_size, j:j + block_size] = blocks[idx]
            idx += 1
    return new_subband