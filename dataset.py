import torch
from torchvision import transforms
import torchvision
from typing import Any, Callable, Dict, List, Optional, Tuple
from PIL import Image


class MNIST(torchvision.datasets.MNIST):
    def __init__(self, root, train, download,  transform):
        super().__init__(root=root, train=train, download=download, transform=transform)

    def __getitem__(self, index: int) -> Tuple[Any, Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], int(self.targets[index])

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img.numpy(), mode="L")

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target, index


class Binarize(object):
    """ This class introduces a binarization transformation
    """

    def __call__(self, pic):
        return torch.Tensor(pic.size()).bernoulli_(pic)

    def __repr__(self):
        return self.__class__.__name__ + '()'


def _data_transforms_mnist():
    """Get data transforms for mnist."""
    train_transform = transforms.Compose([
        transforms.ToTensor(),
        Binarize(),
    ])

    valid_transform = transforms.Compose([
        transforms.ToTensor(),
        Binarize(),

    ])

    return train_transform, valid_transform


def get_loaders(args, idx=True):
    if args.dataset == 'MNIST':
        train_transform, valid_transform = _data_transforms_mnist()
        if idx:
            train_data = MNIST(args.data, True, True, train_transform)
            test_data = MNIST(args.data, False, True, valid_transform)
        else:
            train_data = torchvision.datasets.MNIST(root=args.data, train=True, download=True, transform=train_transform)
            test_data = torchvision.datasets.MNIST(root=args.data, train=False, download=True, transform=valid_transform)

        # sub_idx = train_data.targets <= 2
        # train_data.targets = train_data.targets[sub_idx]
        # train_data.data = train_data.data[sub_idx]
        subset = list(range(0, 2000))
        train_data = torch.utils.data.Subset(train_data, subset)
        num_train, num_test = len(train_data), len(test_data)
        train_queue = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size, shuffle=True, num_workers=4,
                                                  pin_memory=True)

        test_queue = torch.utils.data.DataLoader(test_data, batch_size=num_test, shuffle=False, pin_memory=True)

        return train_queue, test_queue, num_train, num_test
