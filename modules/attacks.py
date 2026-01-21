import numpy as np
import cv2


def apply_attack(image, attack_type, intensity):
    img = image.copy().astype(np.float32)
    h, w = img.shape

    if attack_type == "Gaussian Noise":
        noise = np.random.normal(0, intensity, (h, w))
        return np.clip(img + noise, 0, 255)

    elif attack_type == "Rotation":
        # Manual Pixel Mapping (Nearest Neighbor)
        angle = np.radians(intensity)
        cos_a, sin_a = np.cos(angle), np.sin(angle)
        out = np.zeros_like(img)
        cx, cy = w // 2, h // 2
        for i in range(h):
            for j in range(w):
                tx, ty = j - cx, i - cy
                sx = int(tx * cos_a + ty * sin_a + cx)
                sy = int(-tx * sin_a + ty * cos_a + cy)
                if 0 <= sx < w and 0 <= sy < h:
                    out[i, j] = img[sy, sx]
        return out

    elif attack_type == "JPEG Compression":
        # Manual Block Quantization
        from .transforms import get_dct_blocks, rebuild_from_blocks, block_dct, block_idct
        Q = np.array(
            [[16, 11, 10, 16, 24, 40, 51, 61], [12, 12, 14, 19, 26, 58, 60, 55], [14, 13, 16, 24, 40, 57, 69, 56],
             [14, 17, 22, 29, 51, 87, 80, 62], [18, 22, 37, 56, 68, 109, 103, 77], [24, 35, 55, 64, 81, 104, 113, 92],
             [49, 64, 78, 87, 103, 121, 120, 101], [72, 92, 95, 98, 112, 100, 103, 99]]) * (intensity / 10 + 0.1)
        blocks = get_dct_blocks(img)
        quantized = [block_idct(np.round(block_dct(b) / Q) * Q) for b in blocks]
        return rebuild_from_blocks(quantized, img.shape)

    return img