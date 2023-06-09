import torch
from torch import nn

class TopNetwork(nn.Module):

    def __init__(self, model, device):
        super(TopNetwork, self).__init__()

        self.model = model
        self.device = device

        self.heads = nn.ModuleList()
        self.total_outputs = []
        self.task_offset = []

    def add_head(self):
        self.heads.append(nn.Linear(64, 64).to(self.device))
        self.total_outputs = torch.tensor([head.out_features for head in self.heads])
        self.task_offset = torch.cat([torch.LongTensor(1).zero_(), self.total_outputs.cumsum(0)[:-1]])

    def forward(self, x):
        x, _ = self.model(x)
        y = []
        for head in self.heads:
            y.append(head(x))
        return y
