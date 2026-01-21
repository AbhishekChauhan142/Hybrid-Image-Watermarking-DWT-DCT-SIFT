import cv2
import numpy as np


class SIFTEngine:
    def __init__(self):
        self.sift = cv2.SIFT_create()

    def get_features(self, img):
        img8 = np.clip(img, 0, 255).astype('uint8')
        return self.sift.detectAndCompute(img8, None)

    def correct_geometric_attacks(self, attacked, ref_kp, ref_des):
        kp_at, des_at = self.get_features(attacked)
        if des_at is None: return attacked

        bf = cv2.BFMatcher()
        matches = bf.knnMatch(ref_des, des_at, k=2)
        good = [m for m, n in matches if m.distance < 0.75 * n.distance]

        if len(good) > 10:
            src_pts = np.float32([ref_kp[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
            dst_pts = np.float32([kp_at[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
            M, _ = cv2.estimateAffinePartial2D(dst_pts, src_pts)
            if M is not None:
                return cv2.warpAffine(attacked, M, (attacked.shape[1], attacked.shape[0]))
        return attacked