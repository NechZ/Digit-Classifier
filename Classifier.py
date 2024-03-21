import torch
import torch.nn as nn
from torch import Tensor
from torchtyping import TensorType
import DataLoader


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


def train():
    # Load Data
    loader = DataLoader

    # Train Model
    model = DigitClassifier()
    loss_function = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters())

    epochs = 10
    for epoch in range(epochs):
        running_loss = 0.0
        for image, labels in loader.train_dataloader:
            images = torch.reshape(image, shape=(-1, 784))
            optimizer.zero_grad()
            output = model(images)
            loss = loss_function(output, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss
        print(f"Epoch {epoch + 1} - Training Loss: {running_loss / len(loader.train_dataloader)}")


def evaluate():
    # Load Data
    loader = DataLoader

    # Evaluate Model
    model = torch.load("Digit_Classifier.pth")
    correct_predictions = 0
    total_predictions = 0
    with torch.no_grad():
        for image, labels in loader.test_dataloader:
            images = torch.reshape(image, shape=(-1, 784))
            output = model(images)
            _, predicted = torch.max(output.data, 1)
            total_predictions += labels.size(0)
            correct_predictions += (predicted == labels).sum().item()

    accuracy = (correct_predictions / total_predictions) * 100
    print(f"Accuracy on the test set: {accuracy}%")
