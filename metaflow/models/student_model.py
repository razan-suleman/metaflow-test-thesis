import torch.nn as nn
from .local_model import LocalCNN


class StudentModel(LocalCNN):
    """
    For now, the student is the same architecture as LocalCNN.
    You can later shrink it to test real distillation benefits.
    """
    pass
