import torch

from data import IncrementalData
from util import train, test
from CNN import CNN
from incremental import Freeze

if __name__ == '__main__':
    # get the device
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"Using {device}")

    # load, normalize and split the data
    dataset = IncrementalData(dataset_name="CIFAR10")

    # get the base model
    net = CNN().to(device)

    # train the model with base classes
    train(net, dataset.base_train_loader, epochs=1, device=device)
    print(f'Test with base classes:{test(net, dataset.base_test_loader, device=device)}%')
    print(f'Test with all classes:{test(net, dataset.all_test_loader, device=device)}%')

    # choose an incremental strategy
    increment_strategy = Freeze(net, device)

    # train the model with the new, incremental classes
    increment_strategy.increment(dataset.incr_train_loader, epochs=5)
    print(f'Test with base classes:{test(net, dataset.base_test_loader, device=device)}%')
    print(f'Test with all classes:{test(net, dataset.all_test_loader, device=device)}%')



