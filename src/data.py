import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
from torch.utils.data import DataLoader, SubsetRandomSampler
from torch.utils.data.dataset import Subset

import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
from torch.utils.data import DataLoader, SubsetRandomSampler
from torch.utils.data.dataset import Subset


class IncrementalData:
    def __init__(self, dataset_name="CIFAR10", incremental_dataset_name="MNIST", batch_size=128, incremental_class=0,
                 data_root="./data", choose_incr_from_dataset=False):

        if (dataset_name == "CIFAR10"):
            self.dataset = torchvision.datasets.CIFAR10
            self.class_size = 10
            self.incremental_class = incremental_class
        elif (dataset_name == "CIFAR100"):  # too big for google colab memory
            self.dataset = torchvision.datasets.CIFAR100
            self.class_size = 100
            self.incremental_class = incremental_class
        else:
            raise NotImplementedError("This dataset is not yet implemented.")  # TODO: add other datasets MPII, COCO

        if (incremental_dataset_name == "MNIST"):
            self.incr_dataset = torchvision.datasets.MNIST
        else:
            raise NotImplementedError("This dataset is not yet implemented.")

        # define transforms
        train_transform = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),  # 0.5 is a random value
             transforms.RandomRotation(10),
             transforms.RandomHorizontalFlip(p=0.5)])

        val_test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

        incr_train_transform = transforms.Compose(
            [transforms.Resize(32),
             transforms.Grayscale(num_output_channels=3),
             transforms.ToTensor(),
             transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),  # 0.5 is a random value
             transforms.RandomRotation(10),
             transforms.RandomHorizontalFlip(p=0.5)])

        incr_test_transform = transforms.Compose([
            transforms.Resize(32),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

        # load datasets
        self.train_dataset = self.dataset(
            root=data_root, train=True,
            download=True, transform=train_transform,
        )
        self.test_dataset = self.dataset(
            root=data_root, train=False,
            download=True, transform=val_test_transform
        )

        targets_train = torch.tensor(self.train_dataset.targets)
        targets_test = torch.tensor(self.test_dataset.targets)

        if not choose_incr_from_dataset:
            self.incr_train_dataset = self.incr_dataset(
                root=data_root, train=True,
                download=True, transform=incr_train_transform,
            )
            self.incr_test_dataset = self.incr_dataset(
                root=data_root, train=False,
                download=True, transform=incr_test_transform
            )
            # data loaders
            self.base_train_loader = DataLoader(
                self.train_dataset,
                batch_size=batch_size, num_workers=2
            )

            self.base_test_loader = DataLoader(
                self.test_dataset,
                batch_size=batch_size, shuffle=False, num_workers=2
            )

            # with only the incremental classes
            self.incr_train_loader = DataLoader(
                self.incr_train_dataset,
                batch_size=batch_size, num_workers=2
            )

            self.incr_test_loader = DataLoader(
                self.incr_test_dataset,
                batch_size=batch_size, num_workers=2
            )
        else:
            # find training and incremental dataset's classes
            base_classes = [x for x in range(self.class_size) if x != self.incremental_class]
            incremental_classes = [self.incremental_class]  # more classes, more steps?

            base_train_idx = 0
            base_test_idx = 0
            for base_class in base_classes:
                base_train_idx += targets_train == base_class
                base_test_idx += targets_test == base_class

            incr_train_idx = 0
            incr_test_idx = 0
            for incr_class in incremental_classes:
                incr_train_idx += targets_train == incr_class
                incr_test_idx += targets_test == incr_class

            # define dataloaders
            self.base_train_loader = DataLoader(
                Subset(self.train_dataset, np.where(base_train_idx == 1)[0]),
                batch_size=batch_size, num_workers=2
            )

            self.base_test_loader = DataLoader(
                Subset(self.test_dataset, np.where(base_test_idx == 1)[0]),
                batch_size=batch_size, shuffle=False, num_workers=2
            )

            # with only the incremental classes
            self.incr_train_loader = DataLoader(
                Subset(self.train_dataset, np.where(incr_train_idx == 1)[0]),
                batch_size=batch_size, num_workers=2
            )

            self.incr_test_loader = DataLoader(
                Subset(self.test_dataset, np.where(incr_test_idx == 1)[0]),
                batch_size=batch_size, num_workers=2
            )
            # with all classes available
            self.all_test_loader = DataLoader(
                self.test_dataset,
                batch_size=batch_size, num_workers=2
            )