import numpy as np
from scipy import spatial


def dice_coeff(im1, im2, empty_score=1.0):
    """Calculates the dice coefficient for the images"""

    im1 = np.asarray(im1).astype(np.bool)
    im2 = np.asarray(im2).astype(np.bool)

    if im1.shape != im2.shape:
        raise ValueError("Shape mismatch: im1 and im2 must have the same shape.")

    im1 = im1 > 0.5
    im2 = im2 > 0.5

    im_sum = im1.sum() + im2.sum()
    if im_sum == 0:
        return empty_score

    # Compute Dice coefficient
    intersection = np.logical_and(im1, im2)
    # print(im_sum)
    # dice=2*tp/(tp+fn)+(tp+fp)
    dice = 2. * intersection.sum() / im_sum
    recall = 1. * intersection.sum() / im1.sum()
    precision = 1. * intersection.sum() / im2.sum()
    f1 = 2 * recall * precision / (recall + precision + 1e-6)
    return dice, recall, precision, f1


def getIOU(im1, im2, empty_score=1.0):
    # IOU TP/(TP+FP+FN)
    # MIOU TP+TN/(TP+FP+FN)/2
    # TN=1-FN
    im1 = np.asarray(im1).astype(np.bool)
    im2 = np.asarray(im2).astype(np.bool)

    if im1.shape != im2.shape:
        raise ValueError("Shape mismatch: im1 and im2 must have the same shape.")

    im1 = im1 > 0.5
    im2 = im2 > 0.5

    tpfn = im1.sum()
    tpfp = im2.sum()
    intersection = np.logical_and(im1, im2)
    tp = 1. * intersection.sum()
    IOU=tp/(tpfn+tpfp-tp)
    return IOU

# def numeric_score(prediction, groundtruth):
#     """Computes scores:
#     FP = False Positives
#     FN = False Negatives
#     TP = True Positives
#     TN = True Negatives
#     return: FP, FN, TP, TN"""
#
#     FP = np.float(np.sum((prediction == 1) & (groundtruth == 0)))
#     FN = np.float(np.sum((prediction == 0) & (groundtruth == 1)))
#     TP = np.float(np.sum((prediction == 1) & (groundtruth == 1)))
#     TN = np.float(np.sum((prediction == 0) & (groundtruth == 0)))
#
#     return FP, FN, TP, TN
#
#
# def accuracy_score(prediction, groundtruth):
#     """Getting the accuracy of the model"""
#
#     FP, FN, TP, TN = numeric_score(prediction, groundtruth)
#     N = FP + FN + TP + TN
#     accuracy = np.divide(TP + TN, N)
#     return accuracy * 100.0
