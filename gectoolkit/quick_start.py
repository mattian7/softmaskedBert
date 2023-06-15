# -*- encoding: utf-8 -*-
# @Author: Yunshi Lan
# @Time: 2023/01/27 10:20
# @File: quick_start.py


import os
import sys
from logging import getLogger
from .config.configuration import Config
from .data.utils import get_evaluator_module
from .data.utils import get_dataset_module, get_dataloader_module
from .utils.file_reader import get_model, init_seed
from .utils.utils import get_trainer
from .utils.logger import init_logger


def train_with_train_valid_test_split(temp_config):
    '''
    Train GEC models with evaluation
    '''
    #print('temp_config', temp_config)
    if temp_config['training_resume'] or temp_config['resume']:
        config = Config.load_from_pretrained(temp_config['checkpoint_dir'])
    else:
        config = temp_config
    config._update(temp_config.internal_config_dict)
    device = config["device"]

    logger = getLogger()
    logger.info(config); #exit()

    dataset = get_dataset_module(config)
    dataset._load_dataset()
    # print('config', config)
    dataloader = get_dataloader_module(config)(config, dataset)
    model = get_model(config["model"])(config, dataloader.pretrained_tokenzier)
    # for name, param in model.named_parameters():
    #     print(name)
    # exit()
    if device:
        model = model.cuda(device)

    evaluator = get_evaluator_module(config)(config, dataloader.pretrained_tokenzier)
    trainer = get_trainer(config)(config, model, dataloader, evaluator)

    if temp_config['training_resume'] or temp_config['resume']:
        trainer._load_checkpoint()

    logger.info(model)
    trainer.fit()

def test_with_train_valid_test_split(temp_config):
    '''
    Evaluate existing GEC models
    '''
    config = Config.load_from_pretrained(temp_config['checkpoint_dir'])
    config._update(temp_config.internal_config_dict)
    device = config["device"]

    logger = getLogger()
    logger.info(config)

    dataset = get_dataset_module(config)
    dataset._load_dataset()
    dataloader = get_dataloader_module(config)(config, dataset)
    model = get_model(config["model"])(config, dataloader.pretrained_tokenzier)
    if device:
        model = model.cuda(device)

    evaluator = get_evaluator_module(config)(config, dataloader.pretrained_tokenzier)
    trainer = get_trainer(config)(config, model, dataloader, evaluator)
    trainer._load_checkpoint()

    trainer.test()


def run_toolkit(model_name, dataset_name, language_name, config_dict={}):
    '''
    Run GEC toolkit
    '''
    config = Config(model_name, dataset_name, language_name, config_dict)

    init_seed(config['random_seed'], True)
    #config['test_only'] = True
    init_logger(config)
    if config['test_only']:
        test_with_train_valid_test_split(config)
    else:
        train_with_train_valid_test_split(config)

