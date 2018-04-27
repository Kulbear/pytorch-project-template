import os
import json

from helpers import mkdir
from sklearn.model_selection import ParameterGrid

configs_folder_name = 'configs'

parameter_dict = {
    'model_version': [1],
    'epoch': [10],
    'iteration': [200],
    'learning_rate': [1e-2, 1e-3],
    'batch_size': [128, 64],
    'test_batch_size': [512],
    'is_training': [1]  # often required by batch normalization
}

grid = sorted(
    list(ParameterGrid(parameter_dict)), key=lambda x: x['model_version'])

mkdir(configs_folder_name)

print('[INFO] Start creating configuration files in the folder "config"...')
for i in grid:
    i['experiment_name'] = "Model-V{}-Ep{}-It{}-Lr{}-Bs{}".format(
        i['model_version'], i['epoch'], i['iteration'], i['learning_rate'],
        i['batch_size'])

    config_name = '{}.json'.format(i['experiment_name'])

    with open('./configs/{}'.format(config_name), 'w') as f:
        print('[INFO] Writing - {}'.format(config_name))
        json.dump(i, f)

    with open('./main/main.sh', 'a') as f:
        f.write('python mnist_lenet.py -c ../configs/{}\n'.format(config_name))

print('[INFO] Finish creating configurations...')
