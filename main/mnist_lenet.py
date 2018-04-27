import init_paths

from utils.helpers import read_json_config, get_args
from data_loaders.mnist_loader import MNISTDataLoader
from models.lenet import LeNet
from trainers.mnist_trainer import MNISTTrainer

try:
    args = get_args()
    config = read_json_config(args.config)
except:
    print('Error on reading config file.')
    
print(config.experiment_name)

data_loader = MNISTDataLoader(config)

model = LeNet()

trainer = MNISTTrainer(model, config, data_loader)
trainer.train()