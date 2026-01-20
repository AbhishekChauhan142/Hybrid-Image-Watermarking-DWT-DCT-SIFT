import cv2
import numpy as np


def apply_attack(image, attack_type, intensity):
    img = np.clip(image, 0, 255).astype(np.uint8)

    if attack_type == "Gaussian Noise":
        noise = np.random.normal(0, intensity, img.shape)
        return np.clip(img + noise, 0, 255).astype(np.float32)

    elif attack_type == "Rotation":
        h, w = img.shape
        matrix = cv2.getRotationMatrix2D((w / 2, h / 2), intensity, 1.0)
        return cv2.warpAffine(img, matrix, (w, h)).astype(np.float32)

    elif attack_type == "JPEG Compression":
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), int(100 - intensity)]
        result, encimg = cv2.imencode('.jpg', img, encode_param)
        return cv2.imdecode(encimg, 0).astype(np.float32)

    return image.astype(np.float32)