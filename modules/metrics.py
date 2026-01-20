import numpy as np
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim


def calculate_psnr(img1, img2):
    # Ensure same type for PSNR
    i1 = np.clip(img1, 0, 255).astype(np.uint8)
    i2 = np.clip(img2, 0, 255).astype(np.uint8)
    return psnr(i1, i2)


def calculate_ssim(img1, img2):
    i1 = np.clip(img1, 0, 255).astype(np.uint8)
    i2 = np.clip(img2, 0, 255).astype(np.uint8)
    return ssim(i1, i2)


def calculate_nc(original_wm, extracted_wm):
    """
    Normalized Correlation as per the paper.
    NC = (W . W') / (sqrt(W.W) * sqrt(W'.W'))
    """
    # Convert to 0 and 1 for correlation calculation
    w = (original_wm.flatten() > 127).astype(np.float64)
    w_prime = (extracted_wm.flatten() > 127).astype(np.float64)

    numerator = np.sum(w * w_prime)
    denominator = np.sqrt(np.sum(w ** 2) * np.sum(w_prime ** 2))

    if denominator == 0:
        return 0.0
    return numerator / denominator