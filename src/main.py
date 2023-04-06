import torch
from data import IncrementalData
from copy import deepcopy
from util import train, test
from CNN import CNN
from incremental import *

if __name__ == '__main__':
    # get the device
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"Using {device}")

    # load, normalize and split the data
    dataset = IncrementalData(dataset_name="CIFAR10", incremental_dataset_name="MNIST")

    # get the base model
    net = CNN().to(device)

    # train the model with base classes
    train(net, dataset.base_train_loader, epochs=5, device=device)
    print(f'Test with base classes:{test(net, dataset.base_test_loader, device=device)}%')
    print(f'Test with incr. classes:{test(net, dataset.incr_test_loader, device=device)}%')

    # copy the network for different incremental strategies
    net_copy_1 = deepcopy(net)
    net_copy_2 = deepcopy(net)

    # choose incremental strategies
    incremental_strategies = [Freeze(net, device),
                              AddRegularization(net_copy_1, device, reg_lambda=0.5),
                              LearningWithoutForgetting(net_copy_2, device)]

    nets = [net, net_copy_1, net_copy_2]

    # train the model with the new incremental classes
    for strategy, incr_net in zip(incremental_strategies, nets):
        print(f"For strategy {strategy.name}:")
        strategy.increment(dataset.incr_train_loader, epochs=2)
        print(f'Test with base classes:{test(incr_net, dataset.base_test_loader, device=device)}%')
        print(f'Test with incr. classes:{test(incr_net, dataset.incr_test_loader, device=device)}%')




