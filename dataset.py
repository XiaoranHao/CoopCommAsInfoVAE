import torch
from torchvision import transforms
import torchvision
from PIL import Image


class MNIST(torchvision.datasets.MNIST):
    def __init__(self, root, train, download,  transform):
        super().__init__(root=root, train=train, download=download, transform=transform)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target, index) where target is index of the target class.
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

class celeba64(torchvision.datasets.ImageFolder):
    def __init__(self, root, transform):
        super().__init__(root=root, transform=transform)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path, target = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target, index


class Binarize(object):
    """ This class introduces a binarization transformation
    """

    def __call__(self, pic):
        return torch.Tensor(pic.size()).bernoulli_(pic)

    def __repr__(self):
        return self.__class__.__name__ + '()'

class Static_Binarize(object):
    """ This class introduces a static binarization transformation
    """

    def __call__(self, pic):
        return torch.where(pic > 0.5, 1., 0.)

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

def _data_transforms_celeba64(size=64):
    train_transform = transforms.Compose([
        CropCelebA64(),
        transforms.Resize(size),
        # transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])

    valid_transform = transforms.Compose([
        CropCelebA64(),
        transforms.Resize(size),
        transforms.ToTensor(),
    ])

    return train_transform, valid_transform
    
def generate25Gaussian(n_samples):
    mean = torch.arange(-2,2.1,1)
    mu = torch.cartesian_prod(mean, mean)
    scale = 0.02 * torch.ones_like(mu)
    dist = torch.distributions.Normal(loc=mu, scale=scale)
    samples = dist.sample([n_samples]).reshape(-1,2)
    return samples


class SyntheticGaussian(torch.utils.data.Dataset):
    def __init__(self, data, transform=None):
        self.data = data
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data = self.data[idx]
        if self.transform:
            data = self.transform(data)
        return data, idx

def get_loaders(dataset, path, subclass, subset, return_index, batch_size):
    if dataset == 'MNIST':
        train_transform, valid_transform = _data_transforms_mnist()
        if return_index:
            train_data = MNIST(root=path, train=True, download=True, transform=train_transform)
            test_data = MNIST(root=path, train=False, download=True, transform=valid_transform)
        else:
            train_data = torchvision.datasets.MNIST(root=path, train=True, download=True, transform=train_transform)
            test_data = torchvision.datasets.MNIST(root=path, train=False, download=True, transform=valid_transform)
        if subclass is not None:
            # select sub-classes of training data
            subcls_idx = train_data.targets <= subclass
            train_data.targets = train_data.targets[subcls_idx]
            train_data.data = train_data.data[subcls_idx]

            # select sub-classes of test data
            subcls_idx = test_data.targets <= subclass
            test_data.targets = test_data.targets[subcls_idx]
            test_data.data = test_data.data[subcls_idx] 
    elif dataset == 'mixedGaussian':
        data_gen = generate25Gaussian(300)    
        train_data = SyntheticGaussian(data_gen)
        data_gen = generate25Gaussian(500)    
        test_data = SyntheticGaussian(data_gen)

    elif dataset == 'celeba64':
        train_transform, valid_transform = _data_transforms_celeba64()
        if return_index:
            train_data = celeba64(root=path, transform=train_transform)
            # no test data for celeba64
            test_data = celeba64(root=path, transform=train_transform)

        else:
            train_data = torchvision.datasets.ImageFolder(root=path, transform=train_transform)
            test_data = torchvision.datasets.ImageFolder(root=path, transform=train_transform)

    if subset is not None:         
        subset_idx = list(range(0, subset))
        train_data = torch.utils.data.Subset(train_data, subset_idx)

    num_train, num_test = len(train_data), len(test_data)
    train_queue = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=8,
                                                pin_memory=True)
    reference_queue = torch.utils.data.DataLoader(train_data, batch_size=num_train, shuffle=False)

    test_queue = torch.utils.data.DataLoader(test_data, batch_size=num_train, shuffle=False)

    return train_queue, reference_queue, test_queue, num_train, num_test


# def test_loader(args):
#     transform_continuous = transforms.Compose([transforms.ToTensor()])
#     transform_binary = transforms.Compose([transforms.ToTensor(), Binarize()])

#     data_continuous = MNIST("./", True, True, transform_continuous)
#     data_binary = MNIST("./", True, True, transform_binary)


#     subset = list(range(0, 2000))     
#     train_data_continuous = torch.utils.data.Subset(data_continuous, subset)
#     train_data_binary = torch.utils.data.Subset(data_binary, subset)


#     subset = list(range(0, 2000))
#     test_data_continuous = torch.utils.data.Subset(data_continuous, subset)
#     test_data_binary = torch.utils.data.Subset(data_binary, subset)

#     num_train, num_test = len(train_data_continuous), len(test_data_continuous)

#     train_queue_continuous = torch.utils.data.DataLoader(train_data_continuous, batch_size=num_train, shuffle=False)
#     train_queue_binary = torch.utils.data.DataLoader(train_data_binary, batch_size=args.batch_size, shuffle=True, pin_memory=True)

#     test_queue_continuous = torch.utils.data.DataLoader(test_data_continuous, batch_size=num_test, shuffle=False)
#     test_queue_binary = torch.utils.data.DataLoader(test_data_binary, batch_size=num_test, shuffle=False)

#     return train_queue_continuous, test_queue_continuous, num_train, num_test, train_queue_binary, test_queue_binary





