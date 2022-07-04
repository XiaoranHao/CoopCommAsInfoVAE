import time
import argparse
import os
import torch
import SemiDualOT
import dataset
import yaml

parser = argparse.ArgumentParser(description='EOT Experiment')
parser.add_argument('--config', type=str, default='./config/VAE.yaml',
                    help='config path (default: ./config/VAE.yaml)')
parser.add_argument('--pretrain', type=str, default='./savedmodels/Gaussian_decoder.pth',
                    help='config path (default: ./savedmodels/Gaussian_decoder.pth)')
args = parser.parse_args()
config_path = args.config
decoder_path = args.pretrain

with open(config_path, 'r') as file:
    config = yaml.safe_load(file)

torch.cuda.manual_seed(config['training']['seed'])
torch.cuda.manual_seed_all(config['training']['seed'])
torch.manual_seed(config['training']['seed'])

if __name__ == '__main__':

    train_queue, reference_queue, test_queue, num_train, num_test = dataset.get_loaders(**config['dataset'])
    if config['model']['in_channels'] == 2:
        for b_,(data_all, idx) in enumerate(reference_queue):
            pass
    else:
        for b_,(data_all, target, idx) in enumerate(reference_queue):
            pass
    model = SemiDualOT.CoopCommSemiDual(**config['model'])
    model.c_function.load_state_dict(torch.load(decoder_path))
    model.cuda()
    start_time = time.time()
    model.train_model(train_queue, data_all, **config['training'])

    print('training time elapsed {}s'.format(time.time() - start_time))