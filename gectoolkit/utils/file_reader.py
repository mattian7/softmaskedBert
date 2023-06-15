# -*- encoding: utf-8 -*-
# @Author: Yunshi Lan
# @Time: 2022/1/27 10:35
# @File: utils.py

import json
import math
import copy
import importlib
import random
import re
import numpy as np
import torch
from collections import OrderedDict

def write_json_data(data, filename):
    """
    write data to a json file
    """
    with open(filename, 'w+', encoding='utf-8') as f:
        json.dump(data, f, indent=4, ensure_ascii=False)
    f.close()


def read_json_data(filename):
    '''
    load data from a json file
    '''
    f = open(filename, 'r', encoding="utf-8")
    return json.load(f)

def init_seed(seed, reproducibility):
    r""" init random seed for random functions in numpy, torch, cuda and cudnn

    Args:
        seed (int): random seed
        reproducibility (bool): Whether to require reproducibility
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if reproducibility:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    else:
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False

def get_model(model_name):
    r"""Automatically select model class based on model name

    Args:
        model_name (str): model name

    Returns:
        Model: model class
    """
    model_submodule = ['RNN']
    try:
        model_file_name = model_name.lower()
        for submodule in model_submodule:
            module_path = '.'.join(['gectoolkit.model', submodule, model_file_name])
            #print('module_path', module_path)
            if importlib.util.find_spec(module_path, __name__):
                model_module = importlib.import_module(module_path, __name__)

        model_class = getattr(model_module, model_name)
    except:
        raise NotImplementedError("{} can't be found".format(model_file_name))
    return model_class

