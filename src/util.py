import torch
from torch import nn
from torch import optim


def train(net, train_loader, epochs=10, device="cpu", loss=None, verbose=True):

    if loss is None:
        criterion = nn.CrossEntropyLoss()
    else:
        criterion = loss

    optimizer = optim.Adam(net.parameters(), lr=0.001, weight_decay=0.001)

    steps = 0
    running_loss = 0
    print_every = 100

    for epoch in range(epochs):
        for data in train_loader:
            images, labels = data[0].to(device), data[1].to(device)
            steps += 1

            net.train()
            optimizer.zero_grad()
            output = net(images)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            # print statistics
            if verbose & (steps % print_every == 0):
                print(f'[Epoch {epoch + 1}] loss: {running_loss / print_every:.3f}')
                running_loss = 0.0


def test(net, testloader, device="cpu"):
    correct = 0
    total = 0
    with torch.no_grad():
        net.eval()
        for data in testloader:
            images, labels = data[0].to(device), data[1].to(device)
            pred = net(images)
            _, predicted_labels = torch.max(pred.data, 1)
            total += labels.size(0)
            correct += (predicted_labels == labels).sum().item()

    return 100 * correct / total

