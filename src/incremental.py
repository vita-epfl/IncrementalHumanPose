import torch
from torch import nn
import torch.nn.functional as F
from copy import deepcopy
from util import train


class Incremental:
    def __init__(self, net, device="cpu"):
        self.net = net
        self.device = device
        self.name = "Base class"

    def increment(self, t, incr_loader, epochs):
        pass


class Freeze(Incremental):

    def __init__(self, net, device="cpu"):
        super(Freeze, self).__init__(net, device)
        self.name = "Freeze"

    def increment(t, self, incr_loader, epochs):
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


class AddRegularization(Incremental):

    def __init__(self, net, device="cpu", reg_lambda=0.15):
        super(AddRegularization, self).__init__(net, device)
        self.reg_lambda = reg_lambda
        self.name = "AddRegularization"

    def increment(self, t, incr_loader, epochs):
        # get old parameters
        old_param = []

        for param in self.net.parameters():
            old_param.append(param.clone())

        # define the new loss function
        def regularized_loss(output, target):
            loss = F.cross_entropy(output, target)
            sum = 0
            for p_old, p_new in zip(old_param, self.net.parameters()):
                sum += (torch.linalg.norm(p_new.flatten(), 2) - torch.linalg.norm(p_old.flatten(), 2)) ** 2
            loss += self.reg_lambda * sum
            return loss

        # train the new model with regularization
        train(self.net.to(self.device), incr_loader, epochs=epochs, loss=regularized_loss, device=self.device)


class LearningWithoutForgetting(Incremental):
    def __init__(self, net, device="cpu", reg_lambda=1, temp=2):
        super(LearningWithoutForgetting, self).__init__(net, device)
        self.reg_lambda = reg_lambda
        self.temp = temp
        self.net_old = None
        self.optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
        self.name = "LearningWithoutForgetting"

    def increment(self, t, incr_loader, epochs):
        # train the new model
        self.net.train()

        steps = 0
        running_loss = 0
        print_every = 100

        for epoch in range(epochs):
            for data in incr_loader:
                images, labels = data[0].to(self.device), data[1].to(self.device)
                steps += 1

                # Forward prop on both the old and new models
                outputs_old = None
                if t > 0:
                    outputs_old = self.net_old(images)
                outputs = self.net(images)
                loss = self.criterion(t, outputs, labels, outputs_old)

                # Backward prop
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                running_loss += loss.item()

                # print statistics
                if steps % print_every == 0:
                    print(f'[Epoch {epoch + 1}] loss: {running_loss / print_every:.3f}')
                    running_loss = 0.0

        # Save the old model and freeze its parameters
        self.net_old = deepcopy(self.net)
        self.net_old.eval()
        for param in self.net_old.parameters():
            param.requires_grad = False

    def test(self, testloader,t):
        """
        from https://github.com/mmasana/FACIL/blob/master/src/approach/lwf.py
        """
        total_correct = 0
        total_labels = 0
        with torch.no_grad():
            self.net.eval()
            for data in testloader:
                images, labels = data[0].to(self.device), data[1].to(self.device)
                outputs = self.net(images)
                pred = torch.zeros_like(labels).to(self.device)
                for m in range(len(pred)):
                    this_task = t
                    pred[m] = outputs[this_task][m].argmax()
                correct = (pred == labels).float()
                total_correct += correct.sum().data.cpu().numpy().item()
                total_labels += len(labels)
            return total_correct / total_labels


    def cross_entropy(self, outputs, targets, exp=1.0, size_average=True, eps=1e-5):
        """
        Calculates cross-entropy with temperature scaling
        from https://github.com/mmasana/FACIL/blob/master/src/approach/lwf.py
        """
        out = torch.nn.functional.softmax(outputs, dim=1)
        tar = torch.nn.functional.softmax(targets, dim=1)
        if exp != 1:
            out = out.pow(exp)
            out = out / out.sum(1).view(-1, 1).expand_as(out)
            tar = tar.pow(exp)
            tar = tar / tar.sum(1).view(-1, 1).expand_as(tar)
        out = out + eps / out.size(1)
        out = out / out.sum(1).view(-1, 1).expand_as(out)
        ce = -(tar * out.log()).sum(1)
        if size_average:
            ce = ce.mean()
        return ce

    def criterion(self, t, outputs, targets, outputs_old=None):
        """
        Returns the loss value
        from https://github.com/mmasana/FACIL/blob/master/src/approach/lwf.py
        """
        loss = 0
        if t > 0:
            # Knowledge distillation loss for all previous tasks
            loss += self.reg_lambda * self.cross_entropy(torch.cat(outputs[:t], dim=1),
                                                   torch.cat(outputs_old[:t], dim=1), exp=1.0 / self.temp)
        return loss + torch.nn.functional.cross_entropy(outputs[t], targets)


