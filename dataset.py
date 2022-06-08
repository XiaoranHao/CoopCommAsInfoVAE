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




class CropCelebA64(object):
    """ This class applies cropping for CelebA64. This is a simplified implementation of:
    https://github.com/andersbll/autoencoding_beyond_pixels/blob/master/dataset/celeba.py
    """
    def __call__(self, pic):
        new_pic = pic.crop((15, 40, 178 - 15, 218 - 30))
        return new_pic

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

def _data_transforms_celeba64(size):
    train_transform = transforms.Compose([
        CropCelebA64(),
        transforms.Resize(size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])

    valid_transform = transforms.Compose([
        CropCelebA64(),
        transforms.Resize(size),
        transforms.ToTensor(),
    ])

    return train_transform, valid_transform


def get_loaders(args, idx=True):
    if args.dataset == 'MNIST':
        train_transform, valid_transform = _data_transforms_mnist()
        if idx:
            train_data = MNIST(args.data, True, True, train_transform)
            test_data = MNIST(args.data, True, True, train_transform)
        else:
            train_data = torchvision.datasets.MNIST(root=args.data, train=True, download=True, transform=train_transform)
            test_data = torchvision.datasets.MNIST(root=args.data, train=False, download=True, transform=valid_transform)

        # sub_idx = train_data.targets <= 2
        # train_data.targets = train_data.targets[sub_idx]
        # train_data.data = train_data.data[sub_idx]
        subset = list(range(0, 2000))
        train_data = torch.utils.data.Subset(train_data, subset)
        test_data = torch.utils.data.Subset(test_data, subset)

        num_train, num_test = len(train_data), len(test_data)
        train_queue = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size, shuffle=True, num_workers=8,
                                                  pin_memory=True)

        test_queue = torch.utils.data.DataLoader(test_data, batch_size=num_train, shuffle=False, pin_memory=True)

        return train_queue, test_queue, num_train, num_test


def test_loader(args):
    transform_continuous = transforms.Compose([transforms.ToTensor()])
    transform_binary = transforms.Compose([transforms.ToTensor(), Binarize()])

    data_continuous = MNIST("./", True, True, transform_continuous)
    data_binary = MNIST("./", True, True, transform_binary)


    subset = list(range(0, 2000))     
    train_data_continuous = torch.utils.data.Subset(data_continuous, subset)
    train_data_binary = torch.utils.data.Subset(data_binary, subset)


    subset = list(range(0, 2000))
    test_data_continuous = torch.utils.data.Subset(data_continuous, subset)
    test_data_binary = torch.utils.data.Subset(data_binary, subset)

    num_train, num_test = len(train_data_continuous), len(test_data_continuous)

    train_queue_continuous = torch.utils.data.DataLoader(train_data_continuous, batch_size=num_train, shuffle=False)
    train_queue_binary = torch.utils.data.DataLoader(train_data_binary, batch_size=args.batch_size, shuffle=True, pin_memory=True)

    test_queue_continuous = torch.utils.data.DataLoader(test_data_continuous, batch_size=num_test, shuffle=False)
    test_queue_binary = torch.utils.data.DataLoader(test_data_binary, batch_size=num_test, shuffle=False)

    return train_queue_continuous, test_queue_continuous, num_train, num_test, train_queue_binary, test_queue_binary