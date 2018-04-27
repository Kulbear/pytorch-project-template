import torch
from torchvision import datasets, transforms


class MNISTDataLoader:
    """ A MNIST dataset loader.
    
    A MNIST dataset loader built based on PyTorch APIs.
    It provides ready-to-use batch style data loader.

    Parameters
    ----------
    config: Bunch dictionary
        A Bunch dictionary that supports attribute-style access.

    Attributes
    ----------
    train_loader: Iterable
        The training set loader that can be iterated to get batch data.
    test_loader: Iterable
        The test set loader that can be iterated to get batch data.
    """

    def __init__(self, config):
        self._config = config
        self.train_loader = self._build_train_loader()
        self.test_loader = self._build_test_loader()

    def _build_train_loader(self):
        """ Build a loader that provides batch load for the training set. 
        
        Returns
        -------
        loader: Iterable
            The training set loader that can be iterated to get batch data.
        """
        mnist_training_set = datasets.MNIST(
            '../data',
            train=True,
            download=True,
            transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307, ), (0.3081, ))
            ]))

        loader = torch.utils.data.DataLoader(
            mnist_training_set,
            batch_size=self._config.batch_size,
            shuffle=True)

        return loader

    def _build_test_loader(self):
        """ Build a loader that provides batch load for the test set. 

        Returns
        -------
        loader: Iterable
            The test set loader that can be iterated to get batch data.
        """
        mnist_test_set = datasets.MNIST(
            '../data',
            train=False,
            transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307, ), (0.3081, ))
            ]))

        loader = torch.utils.data.DataLoader(
            mnist_test_set,
            batch_size=self._config.test_batch_size,
            shuffle=True)

        return loader
