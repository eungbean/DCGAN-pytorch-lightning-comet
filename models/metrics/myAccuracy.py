from pytorch_lightning.metrics.functional import accuracy

# define a custom metric as a function
def my_metric(y_true, y_pred):
    pass


# or as a class when we need to accumulate
class MyAccuracy(accuracy):
    def forward(self, y_pred, y_true):
        """
        To define the behavior of the metric when called.
        Args:
            y_pred: The prediction of the model.
            y_true: Target to evaluate the model.

        # calculates accuracy across all GPUs and all Nodes used in training
        """
        return accuracy(pred, target)

    def get_metric(self):
        """
        Compute and return the metric.S
        """
        pass
