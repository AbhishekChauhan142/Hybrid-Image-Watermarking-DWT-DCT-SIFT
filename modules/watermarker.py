import numpy as np
from .transforms import apply_dwt, apply_idwt, block_dct, block_idct, get_dct_blocks, rebuild_from_blocks


class Watermarker:
    def __init__(self, alpha=20.0):
        self.alpha = alpha
        self.mid_band = [(3, 1), (4, 0), (3, 2), (4, 1), (2, 3), (1, 4), (0, 5), (5, 0), (2, 4), (3, 3)]

    def _get_pn(self, key, size):
        state = np.random.RandomState(key)
        return state.normal(0, 1, size), state.normal(0, 1, size)

    def embed(self, image, watermark, key):
        bits = (watermark.flatten() > 127).astype(int)
        LL, HL, LH, HH = apply_dwt(image)
        blocks = get_dct_blocks(HL)
        pn0, pn1 = self._get_pn(key, len(self.mid_band))

        out_blocks = []
        for i, b in enumerate(blocks):
            dct_b = block_dct(b)
            if i < len(bits):
                seq = pn1 if bits[i] == 1 else pn0
                for idx, (r, c) in enumerate(self.mid_band):
                    dct_b[r, c] += self.alpha * seq[idx]
            out_blocks.append(block_idct(dct_b))

        new_HL = rebuild_from_blocks(out_blocks, HL.shape)
        return apply_idwt(LL, new_HL, LH, HH)

    def extract(self, image, key, wm_shape):
        _, HL, _, _ = apply_dwt(image)
        blocks = get_dct_blocks(HL)
        pn0, pn1 = self._get_pn(key, len(self.mid_band))
        bits = []
        for i in range(wm_shape[0] * wm_shape[1]):
            dct_b = block_dct(blocks[i])
            coeffs = np.array([dct_b[r, c] for r, c in self.mid_band])
            bits.append(255 if np.mean(coeffs * pn1) > np.mean(coeffs * pn0) else 0)
        return np.array(bits).reshape(wm_shape).astype(np.uint8)