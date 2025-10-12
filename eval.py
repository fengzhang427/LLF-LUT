import os
import glob
import cv2
import argparse
import numpy as np
import kornia
import skimage.metrics
import lpips
import torchvision.utils as vutils
import torch
import logging
from tqdm import tqdm, trange
from data import create_eval_dataset, create_eval_dataloader
from models import create_model
from utils import (mkdirs, parse_config, AverageMeter, tensor2im,
                   quality_assess, setup_logger, get_model_total_params)
from torch.nn.parallel import DataParallel, DistributedDataParallel


def main():
    parser = argparse.ArgumentParser(description='Image Enhancement Evaluation')
    parser.add_argument('--config', type=str, default='./configs/eval/eval-HDRplus-480p.yml',
                        help='Path to config file (.yaml).')
    args = parser.parse_args()
    config = parse_config(args.config, is_train=False)

    save_path = config['path']['save_path']
    mkdirs(save_path)
    setup_logger('base', save_path, 'test', level=logging.INFO, screen=True, tofile=True)
    model = create_model(config)
    model_params = get_model_total_params(model)

    logger = logging.getLogger('base')
    logger.info('use GPU {}'.format(config['gpu_ids']))
    logger.info('Data: {} - {}'.format(config['dataset']['name'],
                                       config['dataset']['dataset_root']))
    logger.info('Model path: {}'.format(config['path']['pretrain_model']))
    logger.info('Model parameters: {} M'.format(model_params))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.load_state_dict(torch.load(config['path']['pretrain_model']), strict=True)
    model.eval()
    model = model.to(device)

    # single_input_paths = sorted(
    #     glob.glob(os.path.join(config['dataset']['dataset_root'], 'test', '*')))

    test_set = create_eval_dataset(config['dataset'])
    test_loader = create_eval_dataloader(test_set, config['dataset'])

    metric_lpips = lpips.LPIPS(net='alex').to(device)
    PSNR = AverageMeter()
    SSIM = AverageMeter()
    LPIPS = AverageMeter()
    DeltaE = AverageMeter()

    for idx, test_data in tqdm(enumerate(test_loader), total=len(test_loader), desc='Test', ncols=80,
                               leave=False):
        input_image = test_data['A_input'].to(device)
        reference_image = test_data['A_expert'].to(device)
        img_name = test_data['input_name']

        with torch.no_grad():
            output_image, _, _ = model(input_image)

            # LPIPS for each image
            LPIPS.update(torch.mean(metric_lpips(output_image, reference_image, 0, 1)))

            # PSNR, SSIM, deltaE for each image
            output_numpy = tensor2im(output_image)
            reference_numpy = tensor2im(reference_image)
            res = quality_assess(output_numpy, reference_numpy, data_range=255)
            PSNR.update(res['PSNR'])
            SSIM.update(res['SSIM'])
            DeltaE.update(res['DeltaE'])

            # save image
            vutils.save_image(output_image.data, os.path.join(save_path, '%s.png' % img_name[0].split('.')[0]), nrow=1)

    logger.info('################ Final Results ################')
    logger.info('Data: {} - {}'.format(config['dataset']['name'], config['dataset']['dataset_root']))
    logger.info('Model path: {}'.format(config['path']['pretrain_model']))
    msg = 'Total Images: {:d} Average PSNR: {:.6f} dB ; SSIM: {:.6f}; LPIPS: {:.6f}; DeltaE: {:.6f}.'.format(
        len(test_loader), PSNR.average(),
        SSIM.average(), LPIPS.average(), DeltaE.average())
    logger.info(msg)


if __name__ == '__main__':
    main()
