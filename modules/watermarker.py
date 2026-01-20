import numpy as np
from .transforms import apply_dwt, apply_idwt, block_dct, block_idct, get_dct_blocks, rebuild_from_blocks


class Watermarker:
    # Increased alpha from 0.6 to 15.0 for better visibility in DCT domain
    def __init__(self, alpha=15.0, block_size=8):
        self.alpha = alpha
        self.block_size = block_size
        # Mid-band indices (avoiding DC at 0,0 and high frequencies)
        self.mid_band_idx = [
            (3, 1), (4, 0), (3, 2), (4, 1), (2, 3), (1, 4), (0, 5),
            (5, 0), (2, 4), (3, 3), (4, 2), (5, 1), (1, 5), (0, 6),
            (6, 0), (5, 2), (4, 3), (3, 4), (2, 5), (1, 6)
        ]

    def _generate_pn_sequences(self, key, size):
        # Ensure identical sequences for embed/extract
        state = np.random.RandomState(key)
        pn0 = state.normal(0, 1, size)
        pn1 = state.normal(0, 1, size)
        return pn0, pn1

    def embed(self, image, watermark_img, key):
        # Flatten and binarize logo
        wm_bits = (watermark_img.flatten() > 127).astype(int)

        LL, HL, LH, HH = apply_dwt(image)
        blocks = get_dct_blocks(HL, self.block_size)
        pn0, pn1 = self._generate_pn_sequences(key, len(self.mid_band_idx))

        watermarked_blocks = []
        for i, block in enumerate(blocks):
            dct_blk = block_dct(block)
            if i < len(wm_bits):
                # Equation (4) & (5): Y = X + alpha * PN
                seq = pn1 if wm_bits[i] == 1 else pn0
                for idx, (r, c) in enumerate(self.mid_band_idx):
                    dct_blk[r, c] += self.alpha * seq[idx]
            watermarked_blocks.append(block_idct(dct_blk))

        new_HL = rebuild_from_blocks(watermarked_blocks, HL.shape)
        return apply_idwt(LL, new_HL, LH, HH)

    def extract(self, corrected_image, key, wm_shape):
        _, HL, _, _ = apply_dwt(corrected_image)
        blocks = get_dct_blocks(HL, self.block_size)
        pn0, pn1 = self._generate_pn_sequences(key, len(self.mid_band_idx))

        wm_size = wm_shape[0] * wm_shape[1]
        bits = []

        for i in range(wm_size):
            dct_blk = block_dct(blocks[i])
            # Extract the coefficients from the specific mid-band locations
            extracted_coeffs = np.array([dct_blk[r, c] for r, c in self.mid_band_idx])

            # Correlation-based decision (Equation 10)
            # Use dot product for faster/cleaner correlation check in signal processing
            corr0 = np.mean(extracted_coeffs * pn0)
            corr1 = np.mean(extracted_coeffs * pn1)

            bits.append(255 if corr1 > corr0 else 0)

        return np.array(bits).reshape(wm_shape).astype(np.uint8)