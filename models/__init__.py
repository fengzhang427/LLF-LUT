import importlib
from os import path as osp
import logging
from .LLF_LUT import LLF_LUT

logger = logging.getLogger('base')


def create_model(config):

    if config['model'] == 'LLF_LUT':
        model = LLF_LUT(config)

    else:
        raise NotImplementedError('Model [{:s}] not recognized.'.format(config['model']))

    logger.info('Model [{:s}] is created.'.format(model.__class__.__name__))

    return model

