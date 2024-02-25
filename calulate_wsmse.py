import numpy as np
import math
import os

def compute_map_ws(img):
    """
    Compute the weighting map for weighted metrics calculation.
    img: Input image to calculate the weight map.
    """
    height = img.shape[0]
    y_indices = np.arange(height) - (height / 2) + 0.5
    w = np.cos(np.pi * y_indices / height)
    return np.tile(w[:, np.newaxis, np.newaxis], (1, img.shape[1], img.shape[2]))


def calculate_psnr_ws(img, img2, crop_border=0, input_order='HWC', **kwargs):
    """
    Calculate weighted mse between two images.
    img, img2: Input images for comparison.
    crop_border: Border width to crop from images before calculation.
    input_order: Format of input images ('HWC' or 'CHW').
    """
    if crop_border != 0:
        img = img[crop_border:-crop_border, crop_border:-crop_border, ...]
        img2 = img2[crop_border:-crop_border, crop_border:-crop_border, ...]

    img = img.astype(np.float64)
    img2 = img2.astype(np.float64)
    img_w = compute_map_ws(img)

    mse = np.mean(np.multiply((img - img2)**2, img_w))/np.mean(img_w)
    if mse == 0:
        return float('inf')
    return mse