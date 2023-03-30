import torch
from torch import nn
from util import train


class Incremental:
    def __init__(self, net, device="cpu"):
        self.net = net
        self.device = device

    def increment(self, incr_loader, epochs):
        pass


class Freeze(Incremental):

    def __init__(self, net, device="cpu"):
        super(Freeze, self).__init__(net, device)

    def increment(self, incr_loader, epochs):
        # freeze the parameters
        for param in self.net.parameters():
            param.requires_grad = False

        # expand the network
        last_fc = self.net.last_fc
        self.net.last_fc = nn.Sequential(
            nn.Linear(last_fc.in_features, last_fc.in_features // 4),
            nn.ReLU(),
            nn.Linear(last_fc.in_features // 4, last_fc.in_features),
            nn.ReLU(),
            last_fc
        ).to(self.device)

        # train the new layers
        train(self.net, incr_loader, epochs=epochs, device=self.device)
