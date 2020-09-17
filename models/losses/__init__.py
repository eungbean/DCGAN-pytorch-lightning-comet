import torch.nn.functional as F

def adversarial_loss():
    return F.binary_cross_entropy