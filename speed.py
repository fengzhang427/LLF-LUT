import time
import argparse
import torch
from models import LLF_LUT
import numpy as np
import yaml


def get_config(config):
    with open(config, 'r') as stream:
        loader = yaml.load(stream, Loader=yaml.FullLoader)
        return loader


def test_speed(eval_model):
    t_list = []
    for i in range(1, 10):
        img_input = torch.randn(1, 3, 3840, 2160).cuda()
        torch.cuda.synchronize()
        t0 = time.time()
        for j in range(0, 100):
            img_output = eval_model(img_input)

        torch.cuda.synchronize()
        t1 = time.time()
        t_list.append(t1 - t0)
        print((t1 - t0))
    print(t_list)
    return t_list


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='./configs/train-MIT-FiveK-480p.yml',
                        help='Path to config file (.yaml).')
    opts = parser.parse_args()
    config = get_config(opts.config)
    # config['gpu_ids'] = None
    model = LLF_LUT(config).cuda()
    times = test_speed(model)
    avg_time = sum(times) / len(times)
    print('  + Runtime: %.5f ms' % avg_time)
