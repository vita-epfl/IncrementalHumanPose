import torch
from torch import nn

class TopNetwork(nn.Module):

    def __init__(self, model, device):
        super(TopNetwork, self).__init__()

        self.model = model
        last_layer = self.model.final_layer

        self.out_size = last_layer.out_features
        self.heads = nn.ModuleList()
        self.total_outputs = []
        self.task_offset = []

    def add_head(self, num_outputs):
        self.heads.append(nn.Linear(self.out_size, num_outputs).to(self.device))
        self.total_outputs = torch.tensor([head.out_features for head in self.heads])
        self.task_offset = torch.cat([torch.LongTensor(1).zero_(), self.total_outputs.cumsum(0)[:-1]])

    def forward(self, x):
        x = self.model(x)
        y = []
        for head in self.heads:
            y.append(head(x))
        return y
