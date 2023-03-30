import torch
from torch import nn
import torch.nn.functional as F


class Block(nn.Module):
    def __init__(self, size=32):
        super(Block, self).__init__()

        self.conv1 = nn.Conv2d(size, size, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(size)
        self.conv2 = nn.Conv2d(size, size, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(size)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):

        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)

        out += identity # skip connection
        out = self.relu(out)

        return out


class CNN(nn.Module):
    def __init__(self, n_input_channels=3, n_output=10):
        super().__init__()
        self.maxpool = nn.MaxPool2d(3, 2, 1)

        self.conv1 = nn.Conv2d(n_input_channels, 32, 3)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu = nn.ReLU(inplace=True)

        self.layer1 = Block(32)

        self.last_fc = nn.Linear(7200, n_output)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)

        x = torch.flatten(x, 1)
        x = self.last_fc(x)

        return x

    def predict(self, x):
        logits = self.forward(x)
        return F.softmax(logits)
