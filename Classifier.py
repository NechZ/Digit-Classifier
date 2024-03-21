import torch
import torch.nn as nn
from torch import Tensor
from torchtyping import TensorType


class DigitClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        torch.manual_seed(0)
        self.first_linear = nn.Linear(784, 512)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.2)
        self.projection = nn.Linear(512, 10)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, input_images: TensorType[float]) -> Tensor:
        torch.manual_seed(0)
        out = self.softmax(self.projection(self.dropout(self.relu(self.first_linear(input_images)))))
        return out