import numpy as np


def calculate_psnr(img1, img2):
    """PSNR = 10 * log10(MAX^2 / MSE)"""
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0: return 100.0
    return 20 * np.log10(255.0 / np.sqrt(mse))


def calculate_ssim(img1, img2):
    """Manual Global Structural Similarity Index"""
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)

    # Constants to avoid division by zero
    C1 = (0.01 * 255) ** 2
    C2 = (0.03 * 255) ** 2

    mu1, mu2 = np.mean(img1), np.mean(img2)
    sigma1_sq, sigma2_sq = np.var(img1), np.var(img2)
    sigma12 = np.mean((img1 - mu1) * (img2 - mu2))

    num = (2 * mu1 * mu2 + C1) * (2 * sigma12 + C2)
    den = (mu1 ** 2 + mu2 ** 2 + C1) * (sigma1_sq + sigma2_sq + C2)
    return num / den


def calculate_nc(original_wm, extracted_wm):
    """Normalized Correlation"""
    w = (original_wm.flatten() > 127).astype(np.float64)
    w_prime = (extracted_wm.flatten() > 127).astype(np.float64)

    numerator = np.sum(w * w_prime)
    denominator = np.sqrt(np.sum(w ** 2) * np.sum(w_prime ** 2))
    return numerator / denominator if denominator != 0 else 0.0