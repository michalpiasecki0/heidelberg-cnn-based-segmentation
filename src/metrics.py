import torch


def pixel_accuracy(
    logits: torch.Tensor, masks: torch.Tensor, threshold: float = 0.5
) -> float:
    """
    Computes pixel accuracy for binary segmentation.
    Args:
        logits: torch.Tensor, raw model outputs (before sigmoid)
        masks: torch.Tensor, ground truth masks (0 or 1)
        threshold: float, threshold for converting logits to binary predictions
    Returns:
        accuracy: float, pixel-wise accuracy
    """
    with torch.no_grad():
        probs = torch.sigmoid(logits)
        preds = (probs > threshold).float()
        correct = (preds == masks).float().sum()
        total = torch.numel(preds)
        accuracy = correct / total
    return accuracy


def intersection_over_union(
    logits: torch.Tensor, masks: torch.Tensor, threshold: float = 0.5
) -> float:
    """
    Computes Intersection over Union (IoU) for binary segmentation.
    Args:
        logits: torch.Tensor, raw model outputs (before sigmoid)
        masks: torch.Tensor, ground truth masks (0 or 1)
        threshold: float, threshold for converting logits to binary predictions
    Returns:
        iou: float, intersection over union score
    """
    with torch.no_grad():
        probs = torch.sigmoid(logits)
        preds = (probs > threshold).float()
        intersection = (preds * masks).sum()
        union = ((preds + masks) >= 1).float().sum()
        iou = intersection / (union + 1e-7)  # Add epsilon to avoid division by zero
    return iou
