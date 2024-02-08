# Package Imports
import numpy as np
import torch
from skimage.filters import threshold_li

# --------------------------------------------------------------------------------------------

# Single Dice Score Functionality
def dice_score(input_img, gen_img):

    # Image Binarization (Li's Method)
    #x_thresh, x = cv2.threshold(x, 128, 192, cv2.THRESH_OTSU)
    #y_thresh, y = cv2.threshold(y, 128, 192, cv2.THRESH_OTSU)
    input_img = input_img > threshold_li(input_img)
    gen_img = gen_img > threshold_li(gen_img)
    #print(x_thresh, y_thresh)

    # Dice Score Computation
    intersect = np.sum(input_img * gen_img)
    if (np.sum(input_img) == 0) and (np.sum(gen_img) == 0): return 1
    return (2 * intersect) / (np.sum(input_img) + np.sum(gen_img))

# Mean Dice Score Functionality
def mean_dice_score(input_img, gen_img):
    assert(np.all(input_img.shape == gen_img.shape)); mean_score = 0.
    if type(input_img) == torch.Tensor:
        input_img = input_img.numpy(); gen_img = gen_img.numpy()
    for i in range(input_img.shape[0]):
        for j in range(input_img.shape[1]):
            score = dice_score(input_img[i, j, :, :], gen_img[i, j, :, :])
            mean_score += score / (input_img.shape[1] * input_img.shape[0])
    return mean_score