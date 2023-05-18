import torch
from data import IncrementalData
from copy import deepcopy

from top_network import TopNetwork
from util import train, test
from CNN import CNN
from hrnet import PoseHighResolutionNet as HRNet
from incremental import *

if __name__ == '__main__':
    # get the device
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"Using {device}")

    # load, normalize and split the data
    dataset = IncrementalData(dataset_name="CIFAR10", incremental_dataset_name="MNIST")

    # get the base model
    architecture =  {
        "PRETRAINED_LAYERS": ['conv1', 'bn1', 'conv2', 'bn2', 'layer1', 'transition1', 'stage2', 'transition2', 'stage3',
                            'transition3', 'stage4'],
        "FINAL_CONV_KERNEL": 1,
        "STAGE2": {"NUM_CHANNELS": [64, 64], "BLOCK": 'BASIC', "NUM_BRANCHES": 2, "FUSE_METHOD": 'SUM', "NUM_BLOCKS": [2, 2],
                 "NUM_MODULES": 3},
        "STAGE3": {"NUM_CHANNELS": [64, 64, 128, 128, 128], "BLOCK": 'BASIC', "NUM_BRANCHES": 5, "FUSE_METHOD": 'SUM',
                 "NUM_BLOCKS": [2, 2, 2, 2, 2], "NUM_MODULES": 1},
        "STAGE4": {"NUM_CHANNELS": [64, 64, 128, 128, 128], "BLOCK": 'BASIC', "NUM_BRANCHES": 5, "FUSE_METHOD": 'SUM',
                 "NUM_BLOCKS": [2, 2, 2, 2, 2], "NUM_MODULES": 1}},

    net = HRNet(arch=architecture, auxnet=True,
                         intermediate_features='conv').to(device)

    # wrap the base model
    incr_net = TopNetwork(net, device)
    incr_net.add_head(10) # for CIFAR
    incr_net.add_head(10) # for MNIST

    strategy = LearningWithoutForgetting(incr_net, device)

    # train the model with base classes
    t = 0
    strategy.increment(t, dataset.base_train_loader, epochs=5)
    print(f'Test with base classes:{strategy.test(dataset.base_test_loader,0)*100}%')
    print(f'Test with incr. classes:{strategy.test(dataset.incr_test_loader,1)*100}%')

    # train the model with the new incremental classes
    t += 1
    print(f"For strategy {strategy.name}:")
    strategy.increment(t, dataset.incr_train_loader, epochs=2)
    print(f'Test with base classes:{strategy.test(dataset.base_test_loader,0*100)}%')
    print(f'Test with incr. classes:{strategy.test(dataset.incr_test_loader,1*100)}%')




