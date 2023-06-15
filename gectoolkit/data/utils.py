# -*- encoding: utf-8 -*-
# @Author: Yunshi Lan
# @Time: 2022/1/27 10:35
# @File: utils.py

from typing import Union, Type

from ..config.configuration import Config
from ..data.dataset.gec_dataset import ChineseDataset
from ..data.dataloader.gec_dataloader import GECDataLoader
from ..evaluate.evaluator import Evaluator
from ..utils.enum_type import DatasetLanguage


def create_dataset(config):
    """Create dataset according to config

    Args:
        config (gectoolkit.config.configuration.Config): An instance object of Config, used to record parameter information.

    Returns:
        Dataset: Constructed dataset.
    """
    try:
        return eval('Dataset{}'.format(config['model']))(config)
    except:
        pass
    if config['transformers_pretrained_model'] is not None or config['pretrained_model'] is not None:
        return PretrainDataset(config)
    task_type = config['language_name'].lower()
    if task_type == DatasetLanguage.en:
        return EnglishDataset(config)
    elif task_type == DatasetLanguage.zh:
        return ChineseDataset(config)
    else:
        return AbstractDataset(config)


def create_dataloader(config):
    """Create dataloader according to config

    Args:
        config (mwptoolkit.config.configuration.Config): An instance object of Config, used to record parameter information.

    Returns:
        Dataloader module
    """
    try:
        return eval('DataLoader{}'.format(config['model']))
    except:
        pass
    if config['transformers_pretrained_model'] is not None or config['pretrained_model'] is not None:
        return PretrainDataLoader
    task_type = config['language_name'].lower()
    if task_type == DatasetLanguage.en:
        return EnglishDataset(config)
    elif task_type == DatasetLanguage.zh:
        return ChineseDataset(config)
    else:
        return AbstractDataset(config)


def get_dataset_module(config: Config) \
        -> Type[Union[
            ChineseDataset]]:
    """
    return a dataset module according to config

    :param config: An instance object of Config, used to record parameter information.
    :return: dataset module
    """
    try:
        return eval('Dataset{}'.format(config['model']))
    except:
        pass
    if config['transformers_pretrained_model'] is not None or config['pretrained_model'] is not None:
        return PretrainDataset
    task_type = config['language_name'].lower()
    if task_type == DatasetLanguage.en:
        return EnglishDataset(config)
    elif task_type == DatasetLanguage.zh:
        return ChineseDataset(config)
    else:
        return ChineseDataset(config)


def get_dataloader_module(config: Config) \
        -> Type[Union[
            ChineseDataset]]:
    """Create dataloader according to config

        Args:
            config (mwptoolkit.config.configuration.Config): An instance object of Config, used to record parameter information.

        Returns:
            Dataloader module
        """
    try:
        return eval('DataLoader{}'.format(config['model']))
    except:
        pass
    if config['transformers_pretrained_model'] is not None or config['pretrained_model'] is not None:
        return PretrainDataLoader
    return GECDataLoader


def get_evaluator_module(config: Config) \
        -> Type[Union[Evaluator]]:
    """return a evaluator module according to config

    :param config: An instance object of Config, used to record parameter information.
    :return: evaluator module
    """
    evaluator_module = Evaluator

    return evaluator_module



