import torch
import torch.nn.functional as F

from tqdm import tqdm

from .base_trainer import BaseTrainer

LOG_INTERVAL = 200
MOMENTUM = 0.5


class MNISTTrainer(BaseTrainer):
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

    device: {'cpu', 'cuda'}, default 'cuda'
        The computation device will be used.
        If 'cuda', the computation process will be done by GPU.
        If 'cpu', the compuation process will be done by CPU.

    verbose: {0, 1}, default 1
        The log verbose level.
        If 1, training set evaluation will be reported every 
        'LOG_INTERVAL' batches.
        If 0, no report for training set evaluation.

    random_seed: int, default 666
        The random seed used for PyTorch for easier reproduction.

    Attributes
    ----------
    optimizer: torch optimizer
        The optimizer used for training. 
    cur_epoch: int
        A variable for tracking the current number of epochs.
    """

    def __init__(self,
                 model,
                 config,
                 data_loader,
                 device='cuda',
                 verbose=1,
                 random_seed=666):
        super(MNISTTrainer, self).__init__(model, config, data_loader)
        assert device == 'cuda' or device == 'cpu'

        self.device = device
        self.verbose = verbose
        self.model = self.model.to(self.device)
        self.optimizer = torch.optim.SGD(
            model.parameters(), lr=config.learning_rate, momentum=MOMENTUM)
        self.cur_epoch = 0

        torch.manual_seed(random_seed)

    def train(self):
        """ Implement the logic of training the model. """
        self.model.train()
        for _ in range(self.config.epoch):
            self.cur_epoch += 1
            self.train_epoch()
            self.test()

    def train_epoch(self):
        """ Implement the logic of training one epoch. """
        for batch_idx, (data, target) in tqdm(
                enumerate(self.data_loader.train_loader)):
            self.train_batch(batch_idx, data, target)

    def train_batch(self, batch_idx, data, target):
        """ Implement the logic of training with one batch. """

        data, target = data.to(self.device), target.to(self.device)
        self.optimizer.zero_grad()
        output = self.model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        self.optimizer.step()
        if batch_idx % LOG_INTERVAL == 0 and self.verbose:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                self.cur_epoch, batch_idx * len(data),
                len(self.data_loader.train_loader.dataset),
                100. * batch_idx / len(self.data_loader.train_loader),
                loss.item()))

    def test(self):
        """ Implement the logic of evaluating the test set performance. """
        self.model.eval()
        test_loss = 0
        correct = 0
        with torch.no_grad():
            for data, target in self.data_loader.test_loader:
                batch_loss, batch_correct = self.test_batch(data, target)
                test_loss += batch_loss
                correct += batch_correct

        test_loss /= len(self.data_loader.test_loader.dataset)

        print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.
              format(
                  test_loss, correct, len(
                      self.data_loader.test_loader.dataset),
                  100. * correct / len(self.data_loader.test_loader.dataset)))

    def test_batch(self, data, target):
        """ Implement the logic of testing with one batch. """
        data, target = data.to(self.device), target.to(self.device)
        output = self.model(data)
        batch_loss = F.nll_loss(
            output, target, size_average=False).item()  # sum up batch loss
        pred = output.max(
            1, keepdim=True)[1]  # get the index of the max log-probability
        batch_correct = pred.eq(target.view_as(pred)).sum().item()

        return batch_loss, batch_correct
