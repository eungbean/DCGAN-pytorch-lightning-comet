import torch.nn.functional as F


def adversarialLoss(y_hat, y):
    return F.binary_cross_entropy_with_logits(y_hat, y)
