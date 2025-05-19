import torch
def pixel_accuracy(logits, masks, threshold=0.5):
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