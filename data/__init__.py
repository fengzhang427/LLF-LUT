'''Create dataset and dataloader'''
import logging
import torch
import torch.utils.data
from .dataset import ImageDataset_HDRplus_480p, ImageDataset_HDRplus_4K, ImageDataset_FiveK_480p, ImageDataset_FiveK_4K, ImageDataset_LOL, ImageDataset_FiveK_8bit


def create_dataloader(dataset, dataset_config, config, sampler):
    if config['dist']:
        world_size = torch.distributed.get_world_size()
        num_workers = dataset_config['n_workers']
        assert dataset_config['batch_size'] % world_size == 0
        batch_size = dataset_config['batch_size'] // world_size
        shuffle = False
    else:
        num_workers = dataset_config['n_workers'] * len(config['gpu_ids'])
        batch_size = dataset_config['batch_size']
        shuffle = True
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                                       num_workers=num_workers, sampler=sampler, drop_last=True,
                                       pin_memory=True)


def create_dataset(dataset_config):
    if dataset_config['name'] == 'HDRplus_480p':
        dataset = ImageDataset_HDRplus_480p(dataset_config['dataset_root'], mode="train")
    elif dataset_config['name'] == 'MIT_FiveK_480p':
        dataset = ImageDataset_FiveK_480p(dataset_config['dataset_root'], mode="train")
    elif dataset_config['name'] == 'HDRplus_4K':
        dataset = ImageDataset_HDRplus_4K(dataset_config['dataset_root'], mode="train")
    elif dataset_config['name'] == 'MIT_FiveK_4K':
        dataset = ImageDataset_FiveK_4K(dataset_config['dataset_root'], mode="train")
    elif dataset_config['name'] == 'LOL':
        dataset = ImageDataset_LOL(dataset_config['dataset_root'], mode="train")
    elif dataset_config['name'] == 'MIT_FiveK_8bit':
        dataset = ImageDataset_FiveK_8bit(dataset_config['dataset_root'], mode="train")
    else:
        raise NotImplementedError('Dataset [{:s}] is not recognized.'.format(dataset_config['name']))

    logger = logging.getLogger('base')
    logger.info('Dataset [{:s} - {:s}] is created.'.format(dataset.__class__.__name__,
                                                           dataset_config['name']))
    return dataset


def create_eval_dataset(dataset_config):
    if dataset_config['name'] == 'HDRplus_480p':
        dataset = ImageDataset_HDRplus_480p(dataset_config['dataset_root'], mode="test")
    elif dataset_config['name'] == 'MIT_FiveK_480p':
        dataset = ImageDataset_FiveK_480p(dataset_config['dataset_root'], mode="test")
    elif dataset_config['name'] == 'HDRplus_4K':
        dataset = ImageDataset_HDRplus_4K(dataset_config['dataset_root'], mode="test")
    elif dataset_config['name'] == 'MIT_FiveK_4K':
        dataset = ImageDataset_FiveK_4K(dataset_config['dataset_root'], mode="test")
    elif dataset_config['name'] == 'LOL':
        dataset = ImageDataset_LOL(dataset_config['dataset_root'], mode="test")
    elif dataset_config['name'] == 'MIT_FiveK_8bit':
        dataset = ImageDataset_FiveK_8bit(dataset_config['dataset_root'], mode="test")
    else:
        raise NotImplementedError('Dataset [{:s}] is not recognized.'.format(dataset_config['name']))

    return dataset


def create_eval_dataloader(dataset, dataset_config):
    num_workers = dataset_config['n_workers']
    shuffle = False
    return torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=shuffle, num_workers=num_workers, pin_memory=True)
