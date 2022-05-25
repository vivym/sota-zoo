import numpy as np
import torch


@torch.no_grad
def pixel_accuracy(pred_mask: torch.Tensor, gt_mask: torch.Tensor) -> float:
    """
    Compute pixel accuracy.
    """

    accuracy = (pred_mask == gt_mask).sum() / gt_mask.numel()
    return accuracy.item()


@torch.no_grad
def mean_iou(pred_mask: torch.Tensor, gt_mask: torch.Tensor, num_classes: int) -> float:
    """
    Compute mIoU.
    """

    iou_per_class = []
    for c in range(num_classes):
        pred_true = pred_mask == c
        gt_true = gt_mask == c

        if not gt_true.any():
            iou_per_class.append(np.nan)
        else:
            intersect = (pred_true & gt_true).sum()
            union = (pred_true | gt_true).sum()
            iou_per_class.append(intersect / (union + 1e-8))

    return np.nanmean(iou_per_class)
