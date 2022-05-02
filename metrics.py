import math
import cv2
import numpy as np
import skimage
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio


def calculate_psnr(gt, pred, method="scikit"):

    if method == "numpy":
        # gt and pred have range [0, 255]
        gt = gt.astype(np.float64)
        pred = pred.astype(np.float64)
        mse = np.mean((gt - pred) ** 2)
        if mse == 0:
            return float("inf")
        return 20 * math.log10(255.0 / math.sqrt(mse))
    elif method == "scikit":
        psnr_score = skimage.metrics.peak_signal_noise_ratio(gt, pred)
        return psnr_score


def numpy_ssim(gt, pred):
    C1 = (0.01 * 255) ** 2
    C2 = (0.03 * 255) ** 2

    gt = gt.astype(np.float64)
    pred = pred.astype(np.float64)
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())

    mu1 = cv2.filter2D(gt, -1, window)[5:-5, 5:-5]  # valid
    mu2 = cv2.filter2D(pred, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1**2
    mu2_sq = mu2**2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(gt**2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(pred**2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(gt * pred, -1, window)[5:-5, 5:-5] - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean()


def calculate_ssim(gt, pred, method="scikit"):
    if method == "numpy":
        if not gt.shape == pred.shape:
            raise ValueError("Input images must have the same dimensions.")
        if gt.ndim == 2:
            return numpy_ssim(gt, pred)
        elif gt.ndim == 3:
            if gt.shape[2] == 3:
                ssims = []
                for i in range(3):
                    ssims.append(numpy_ssim(gt, pred))
                return np.array(ssims).mean()
            elif gt.shape[2] == 1:
                return numpy_ssim(np.squeeze(gt), np.squeeze(pred))
        else:
            raise ValueError("Wrong input image dimensions.")

    elif method == "scikit":
        ssim_score = ssim(im1=gt, im2=pred, win_size=3)
        return ssim_score


if __name__ == "__main__":
    gt = np.random.rand(3, 256, 256)
    pred = np.random.rand(3, 256, 256)
    print(gt.shape, pred.shape)
    print(calculate_psnr(gt, gt))
    print(calculate_ssim(gt, gt))
