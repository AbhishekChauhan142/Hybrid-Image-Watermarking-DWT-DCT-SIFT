import cv2
import numpy as np


class SIFTEngine:
    def __init__(self):
        self.sift = cv2.SIFT_create()

    def get_features(self, image):
        # Convert to uint8 for SIFT
        img_uint8 = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX).astype('uint8')
        kp, des = self.sift.detectAndCompute(img_uint8, None)
        return kp, des

    def correct_geometric_attacks(self, attacked_image, original_kp, original_des):
        kp_attacked, des_attacked = self.get_features(attacked_image)

        if des_attacked is None or len(des_attacked) < 4:
            return attacked_image

        # Feature Matching
        bf = cv2.BFMatcher()
        matches = bf.knnMatch(original_des, des_attacked, k=2)

        good_matches = []
        for m, n in matches:
            if m.distance < 0.75 * n.distance:
                good_matches.append(m)

        if len(good_matches) > 4:
            src_pts = np.float32([original_kp[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
            dst_pts = np.float32([kp_attacked[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

            # Find RST (Rotation, Scale, Translation) matrix
            M, mask = cv2.estimateAffinePartial2D(dst_pts, src_pts)
            if M is not None:
                h, w = attacked_image.shape
                corrected_img = cv2.warpAffine(attacked_image, M, (w, h))
                return corrected_img

        return attacked_image