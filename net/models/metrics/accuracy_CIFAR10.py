import numpy as np


def accuracy(pred, label):
    """
    To define the behavior of the metric when called.
    Args:
        pred: The prediction of the model.
        true: Target to evaluate the model.

    # calculates accuracy across all GPUs and all Nodes used in training
    """
    """
        Originally I tried to adopt the function below
        But it produces error
        https://github.com/PyTorchLightning/pytorch-lightning/issues/2305
        So I defiened my own accuracy metric.

        from pytorch_lightning.metrics.functional import accuracy

        pred_real = torch.round(pred_real).view(-1)
        pred_fake = torch.round(pred_fake).view(-1)
        accD_real = accuracy(pred_real.view(-1), label_real)
        accD_fake = accuracy(pred_fake.view(-1), label_fake), num_classes=1)
    """

    if not isinstance(pred, np.ndarray):
        pred = pred.data.cpu().numpy()

    if not isinstance(label, np.ndarray):
        label = label.data.cpu().numpy()

    pred = np.round(pred)
    acc = np.mean(pred == label)

    return acc

    # Original Function in main python script.

    # pred_real = torch.round(torch.sigmoid(pred_real)).data.cpu().numpy()
    # pred_fake = torch.round(torch.sigmoid(pred_fake)).data.cpu().numpy()
    # d_acc_real = np.mean(pred_real == label_real.data.cpu().numpy())
    # d_acc_fake = np.mean(pred_fake == label_fake.data.cpu().numpy())
