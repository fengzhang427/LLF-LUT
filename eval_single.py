import os
import glob
import cv2
import imageio.v3 as imageio
import argparse
import numpy as np
from tqdm import tqdm, trange
import torchvision.utils as vutils
import torch
import logging
import data.torchvision_x_functional as TF_x
from models import create_model
from utils import (mkdirs, parse_config, setup_logger, get_model_total_params)


def main():
    parser = argparse.ArgumentParser(description='Image Enhancement Evaluation')
    parser.add_argument('--config', type=str, default='./configs/eval/eval-HDRplus-480p.yml',
                        help='Path to config file (.yaml).')
    args = parser.parse_args()
    config = parse_config(args.config, is_train=False)

    single_input_paths = sorted(glob.glob(os.path.join('/data/990pro2', 'HDR_test_data', 'Ward_Disk_EXR', '*')))
    # single_input_paths = sorted(glob.glob(os.path.join('/data/990pro2', 'HDR_test_data', 'Ward hdr Data set', '*')))

    save_path = './LOL_results'
    mkdirs(save_path)

    setup_logger('base', save_path, 'test', level=logging.INFO, screen=True, tofile=True)
    model = create_model(config)
    model_params = get_model_total_params(model)

    logger = logging.getLogger('base')
    logger.info('use GPU {}'.format(config['gpu_ids']))
    logger.info('Model path: {}'.format(config['path']['pretrain_model']))
    logger.info('Model parameters: {} M'.format(model_params))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.load_state_dict(torch.load(config['path']['pretrain_model']), strict=True)
    model.eval()
    model = model.to(device)

    for idx, test_data_path in tqdm(enumerate(single_input_paths), total=len(single_input_paths), desc='Test', ncols=80,
                                    leave=False):
        img_name = os.path.split(test_data_path)[-1]
        img_input = imageio.imread(test_data_path, extension='.exr')
        input_image = TF_x.to_tensor(img_input).to(device)
        input_image = input_image.unsqueeze(0)

        with torch.no_grad():
            output_image, _, _ = model(input_image)

            # save image
            # vutils.save_image(input_image.data, os.path.join(save_path, 'input_%s.png' % img_name.split('.')[0]), nrow=1)
            vutils.save_image(output_image.data, os.path.join(save_path, 'result_%s.png' % img_name.split('.')[0]), nrow=1)

    logger.info('################ Final Results ################')
    logger.info('Model path: {}'.format(config['path']['pretrain_model']))


if __name__ == '__main__':
    main()
