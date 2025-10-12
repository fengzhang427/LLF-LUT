from thop import profile
import argparse
import torch
from models import LLF_LUT
import numpy as np
import yaml


def get_config(config):
    with open(config, 'r') as stream:
        loader = yaml.load(stream, Loader=yaml.FullLoader)
        return loader


def print_model_parm_nums(model):
    total = sum([param.nelement() for param in model.parameters()])
    print('  + Number of Params: %.3f M' % (total / 1e6))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='./configs/train/train-MIT-FiveK-480p.yml', help='Path to config file (.yaml).')
    opts = parser.parse_args()
    config = get_config(opts.config)
    # config['gpu_ids'] = None
    model = LLF_LUT(config).cuda()

    inputs = torch.randn(1, 3, 3840, 2160).cuda()
    macs, _ = profile(model, (inputs,))
    print_model_parm_nums(model)
    print('  + Number of MACs: %.5f GFLOPs' % (macs / 1e9))
