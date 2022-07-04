import time
import argparse
import os
import torch
import VAE
import dataset
import yaml

parser = argparse.ArgumentParser(description='EOT Experiment')
parser.add_argument('--config', type=str, default='./config/VAE.yaml',
                    help='config path (default: ./config/VAE.yaml)')
args = parser.parse_args()
config_path = args.config

with open(config_path, 'r') as file:
    config = yaml.safe_load(file)

torch.cuda.manual_seed(config['training']['seed'])
torch.cuda.manual_seed_all(config['training']['seed'])
torch.manual_seed(config['training']['seed'])

if __name__ == '__main__':

    train_queue, reference_queue, test_queue, num_train, num_test = dataset.get_loaders(**config['dataset'])
    model = VAE.VAE(**config['model'])
    model.cuda()
    start_time = time.time()
    model.train_model(train_queue, **config['training'])

    print('training time elapsed {}s'.format(time.time() - start_time))