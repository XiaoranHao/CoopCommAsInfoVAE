import torch
from torchvision import transforms
import torchvision


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


def get_loaders(args):
    if args.dataset == 'MNIST':
        train_transform, valid_transform = _data_transforms_mnist()
        train_data = torchvision.datasets.MNIST(root=args.data, train=True, download=True, transform=train_transform)
        test_data = torchvision.datasets.MNIST(root=args.data, train=False, download=True, transform=valid_transform)

    #     idx = train_data.targets <= 2
    #     train_data.targets = train_data.targets[idx]
    #     train_data.data = train_data.data[idx]
    #     subset = list(range(0, 300))
    #     train_data= torch.utils.data.Subset(train_data, subset)

        train_queue = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size, shuffle=True,
                                                  pin_memory=True)

        test_queue = torch.utils.data.DataLoader(test_data, batch_size=10000, shuffle=False, pin_memory=True)

        return train_queue, test_queue
