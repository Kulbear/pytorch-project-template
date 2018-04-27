class BaseTrainer:
    """ Base class for all trainers.

    Parameters
    ----------
    model: nn.Modules
        A provided PyTorch model that inheritates torch.nn.Modules.

    config: bunch dict
        A config file of Bunch dictionary format that supports 
        attribute-style access.

    data_loader: DatasetLoader
        A ready-to-use batch style data loader.
    """

    def __init__(self, model, config, data_loader):
        self.model = model
        self.config = config
        self.data_loader = data_loader

    def train(self):
        """ Implement the logic of training the model. """
        raise NotImplementedError

    def train_epoch(self):
        """ Implement the logic of training one epoch. """
        raise NotImplementedError

    def train_batch(self):
        """ Implement the logic of training with one batch. """
        raise NotImplementedError

    def test(self):
        """ Implement the logic of evaluating the test set performance. """
        raise NotImplementedError

    def test_batch(self):
        """ Implement the logic of testing with one batch. 
        
        May not need to be implemented if the test set size is small 
        enough for fitting into the momery.
        """
        pass
