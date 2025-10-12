import os
import math
import argparse
import random
import logging
from tqdm import tqdm, trange
import torch
import kornia
import torch.distributed as dist
import torch.multiprocessing as mp
from data.data_sampler import DistIterSampler
from data import create_dataloader, create_dataset, create_eval_dataset, create_eval_dataloader
from trainer import Trainer
from utils import (parse_config, dict2str, dict_to_nonedict, setup_logger, write_images,
                   mkdir_and_rename, mkdirs, set_random_seed, check_resume, tensor2im, quality_assess)


def init_dist(backend='nccl', **kwargs):
    ''' initialization for distributed training'''
    if mp.get_start_method(allow_none=True) != 'spawn':
        mp.set_start_method('spawn')
    rank = int(os.environ['RANK'])
    num_gpus = torch.cuda.device_count()
    torch.cuda.set_device(rank % num_gpus)
    dist.init_process_group(backend=backend, **kwargs)


def main():
    # config
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='./configs/train/train-HDRplus-480p.yml', help='Path to config file (.yaml).')
    parser.add_argument('--launcher', choices=['none', 'pytorch'], default='none',
                        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    args = parser.parse_args()
    config = parse_config(args.config, is_train=True)

    # distributed training settings
    if args.launcher == 'none':  # disabled distributed training
        config['dist'] = False
        rank = -1
        print('Disabled distributed training.')
    else:
        config['dist'] = True
        init_dist()
        world_size = torch.distributed.get_world_size()
        rank = torch.distributed.get_rank()

    # Loading resume state if exists
    if config['path'].get('resume_state', None):
        # distributed resuming: all load into default GPU
        device_id = torch.cuda.current_device()
        resume_state = torch.load(config['path']['resume_state'],
                                  map_location=lambda storage, loc: storage.cuda(device_id))
        check_resume(config, resume_state['iter'])  # check resume options
    else:
        resume_state = None

    # mkdir and loggers
    if rank <= 0:  # normal training (rank -1) OR distributed training (rank 0)
        if resume_state is None:
            mkdir_and_rename(config['path']['experiments_root'])  # rename experiment folder if exists
            mkdirs((path for key, path in config['path'].items() if not key == 'experiments_root'
                    and 'pretrain_model' not in key and 'resume' not in key))

        # config loggers. Before it, the log will not work
        setup_logger(
            'base', config['path']['log'], 'train_' + config['name'],
            level=logging.INFO, screen=True, tofile=True
        )
        logger = logging.getLogger('base')
        logger.info(dict2str(config))

        # tensorboard logger
        if config['use_tb_logger'] and 'debug' not in config['name']:
            version = float(torch.__version__[0:3])
            if version >= 1.1:  # PyTorch 1.1
                from torch.utils.tensorboard import SummaryWriter
            else:
                logger.info(
                    'You are using PyTorch {}. Tensorboard will use [tensorboardX]'.format(version))
                from tensorboardX import SummaryWriter
            tb_logger = SummaryWriter(log_dir='../tb_logger/' + config['name'])
    else:
        setup_logger('base', os.path.join(config['path']['experiments_root'] + "/logs/"), 'train', level=logging.INFO,
                     screen=True)
        logger = logging.getLogger('base')

    # convert to NoneDict, which returns None for missing keys
    config = dict_to_nonedict(config)

    # random seed
    seed = config['train']['manual_seed']
    if seed is None:
        seed = random.randint(1, 10000)
    if rank <= 0:
        logger.info('Random seed: {}'.format(seed))
    set_random_seed(seed)

    torch.backends.cudnn.benckmark = True
    # torch.backends.cudnn.deterministic = True

    # create train dataloader
    dataset_ratio = 200  # enlarge the size of each epoch
    train_set = create_dataset(config['dataset'])
    val_set = create_eval_dataset(config['dataset'])
    train_size = int(math.ceil(len(train_set) / config['dataset']['batch_size']))
    total_iters = int(config['train']['niter'])
    total_epochs = int(math.ceil(total_iters / train_size))
    if config['dist']:
        train_sampler = DistIterSampler(train_set, world_size, rank, dataset_ratio)
        total_epochs = int(math.ceil(total_iters / (train_size * dataset_ratio)))
    else:
        train_sampler = None
    train_loader = create_dataloader(train_set, config['dataset'], config, train_sampler)
    val_loader = create_eval_dataloader(val_set, config['dataset'])
    if rank <= 0:
        logger.info('Number of train images: {:,d}, iters: {:,d}'.format(
            len(train_set), train_size))
        logger.info('Total epochs needed: {:d} for iters {:,d}'.format(
            total_epochs, total_iters))
    assert train_loader is not None

    trainer = Trainer(config)

    # resume training
    if resume_state:
        logger.info('Resuming training from epoch: {}, iter: {}.'.format(
            resume_state['epoch'], resume_state['iter']))

        start_epoch = resume_state['epoch']
        current_step = resume_state['iter']
        trainer.resume_training(resume_state)  # handle optimizers and schedulers
    else:
        current_step = 0
        start_epoch = 0

    # record metrics
    f_every = open("%s/Metrics_MAX.txt" % config['path']['experiments_root'], 'a')
    max_psnr, max_ssim = 0, 0
    max_psnr_1, max_ssim_1, max_epoch_1 = 0, 0, 0
    max_psnr_2, max_ssim_2, max_epoch_2 = 0, 0, 0

    # training
    logger.info('Start training from epoch: {:d}, iter: {:d}'.format(start_epoch, current_step))
    for epoch in trange(start_epoch, total_epochs, desc='Train', ncols=80):
        if config['dist']:
            train_sampler.set_epoch(epoch)
        for _, train_data in tqdm(enumerate(train_loader), total=len(train_loader),
                                  desc='Train epoch={}, iter={}'.format(epoch, current_step),
                                  ncols=80, leave=False):
            current_step += 1
            if current_step > total_iters:
                break

            # training
            trainer.train_one_sample(train_data, current_step)

            # log
            if current_step % config['logger']['print_freq'] == 0:
                logs = trainer.get_current_log()
                message = '<epoch:{:3d}, iter:{:8,d}, lr:('.format(epoch, current_step)
                for v in trainer.get_current_learning_rate():
                    message += '{:.3e},'.format(v)
                message += ')>'
                for k, v in logs.items():
                    message += '{:s}: {:.4e} '.format(k, v)
                    # tensorboard logger
                    if config['use_tb_logger'] and 'debug' not in config['name']:
                        if rank <= 0:
                            tb_logger.add_scalar(k, v, current_step)
                if rank <= 0:
                    logger.info(message)

                # image save
                print("[*] Evaluating for phase train / iter %d..." % current_step)
                psnr = 0
                ssim = 0
                image_batch = []
                print_list = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20]
                for batch_idx, val_data in tqdm(enumerate(val_loader), total=len(val_loader), desc='Valid', ncols=80,
                                                leave=False):
                    input_image, enhanced_image, reference_image = trainer.eval_sample(val_data)
                    # reference_numpy = tensor2im(reference_image)
                    # enhanced_numpy = tensor2im(enhanced_image)
                    # res = quality_assess(enhanced_numpy, reference_numpy, data_range=255)
                    # psnr += res['PSNR']
                    # ssim += res['SSIM']
                    psnr += kornia.metrics.psnr(enhanced_image, reference_image, max_val=1.0)
                    ssim += kornia.metrics.ssim(enhanced_image, reference_image, window_size=11, max_val=1.0).mean().item()

                    if (batch_idx + 1) in print_list:
                        sample = input_image, enhanced_image, reference_image
                        image_batch.append(torch.cat(sample, dim=0))
                        iter_directory = os.path.join(config['path']['images'], 'eval_%s' % (current_step))
                        if not os.path.exists(iter_directory):
                            print("Creating directory: {}".format(iter_directory))
                            os.makedirs(iter_directory)
                        write_images(sample, 3, '%s/sample_%s.bmp' % (iter_directory, batch_idx + 1))
                metrics_data = [
                    "Iters: %s, PSNR: %s, SSIM: %s" % (current_step, psnr / len(val_loader), ssim / len(val_loader)), '\n']
                f = open("%s/Metrics.txt" % config['path']['experiments_root'], 'a')
                f.writelines(metrics_data)
                f.close()
                print("===> Iteration[{}]: psnr: {}, ssim:{}".format(epoch + 1, psnr / len(val_loader),
                                                                     ssim / len(val_loader)))

            # save models and training states
            if current_step % config['logger']['save_checkpoint_freq'] == 0:
                if rank <= 0:
                    logger.info('Saving models and training states.')
                    trainer.save(current_step)
                    trainer.save_training_state(epoch, current_step)
                # psnr validation
                psnr = 0
                ssim = 0
                for _, val_data in tqdm(enumerate(val_loader), total=len(val_loader), desc='Valid', ncols=80,
                                        leave=False):
                    input_image, enhanced_image, reference_image = trainer.eval_sample(val_data)
                    # reference_numpy = tensor2im(reference_image)
                    # enhanced_numpy = tensor2im(enhanced_image)
                    # res = quality_assess(enhanced_numpy, reference_numpy, data_range=255)
                    # psnr += res['PSNR']
                    # ssim += res['SSIM']
                    psnr += kornia.metrics.psnr(enhanced_image, reference_image, max_val=1.0)
                    ssim += kornia.metrics.ssim(enhanced_image, reference_image, window_size=11, max_val=1.0).mean().item()
                avg_psnr, avg_ssim = psnr / len(val_loader), ssim / len(val_loader)
                if avg_psnr > max_psnr:
                    max_psnr = avg_psnr
                    max_psnr_1 = avg_psnr
                    max_ssim_1 = avg_ssim
                    max_epoch_1 = epoch + 1
                else:
                    pass
                if avg_ssim > max_ssim:
                    max_ssim = avg_ssim
                    max_psnr_2 = avg_psnr
                    max_ssim_2 = avg_ssim
                    max_epoch_2 = epoch + 1
                else:
                    pass

    if rank <= 0:
        metrics_data_every = ["MAX_PSNR_Epoch: %s, PSNR: %s, SSIM: %s" % (max_epoch_1, max_psnr_1, max_ssim_1), '\n',
                              "MAX_SSIM_Epoch: %s, PSNR: %s, SSIM: %s" % (max_epoch_2, max_psnr_2, max_ssim_2)]
        f_every.writelines(metrics_data_every)
        logger.info('Saving the final model.')
        trainer.save('latest')
        logger.info('End of training.')


if __name__ == '__main__':
    main()
