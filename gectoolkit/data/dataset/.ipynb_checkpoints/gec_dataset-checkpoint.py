# -*- encoding: utf-8 -*-
# @Author: Yunshi Lan
# @Time: 2022/1/27 11:31
# @File: abstract_dataset.py


import random
import os
import copy
import re
import torch

from ...config.configuration import Config
from ...utils.file_reader import read_json_data, write_json_data
# from gectoolkit.utils.preprocess_tools import get_group_nums, get_deprel_tree, get_span_level_deprel_tree
# from gectoolkit.utils.preprocess_tools import id_reedit
# from gectoolkit.utils.preprocess_tool.equation_operator import from_postfix_to_infix, from_prefix_to_infix, operator_mask,EN_rule1_stat,EN_rule2
# from gectoolkit.utils.enum_type import DatasetName,FixType


class ChineseDataset(object):
    """Chinese dataset

    the base class of dataset class
    """
    def __init__(self, config):
        super().__init__()
        self.model = config["model"]
        self.dataset = config["dataset"]
        
        self.dataset_path = config['dataset_dir'] if config['dataset_dir'] else config["dataset_path"]
        self.language_name = config["language"]

        self.validset_divide = config["validset_divide"]
        self.shuffle = config["shuffle"]
        
        self.device = config["device"]

        self.resume_training = config['resume_training'] if config['resume_training'] else config['resume']

        self.from_pretrained = False
        self.datas = []

        self.trainset = []
        self.validset = []
        self.testset = []
        self.validset_id = []
        self.trainset_id = []
        self.testset_id = []
        self.folds = []
        self.folds_id = []
        
        self._load_dataset()

    def _load_all_data(self):
        trainset_file = os.path.join(self.dataset_path, 'trainset.json')
        validset_file = os.path.join(self.dataset_path, 'validset.json')
        testset_file = os.path.join(self.dataset_path, 'testset.json')

        if os.path.isabs(trainset_file):
            trainset = read_json_data(trainset_file)
        else:
            trainset = read_json_data(os.path.join(os.getcwd(), trainset_file))
        if os.path.isabs(validset_file):
            validset = read_json_data(validset_file)
        else:
            validset = read_json_data(os.path.join(os.getcwd(), validset_file))
        if os.path.isabs(testset_file):
            testset = read_json_data(testset_file)
        else:
            testset = read_json_data(os.path.join(os.getcwd(), testset_file))

        return trainset + validset + testset

    def _load_dataset(self):
        '''
        read dataset from files
        '''
        if self.trainset_id and self.testset_id:
            self._init_split_from_id()
        else:
            # print('self.dataset_path', self.dataset_path); exit()
            trainset_file = os.path.join(self.dataset_path, 'trainset.json')
            validset_file = os.path.join(self.dataset_path, 'validset.json')
            testset_file = os.path.join(self.dataset_path, 'testset.json')
        
            if os.path.isabs(trainset_file):
                self.trainset = read_json_data(trainset_file)
            else:
                self.trainset = read_json_data(os.path.join(os.getcwd(),trainset_file))#[:1000]
            if os.path.isabs(validset_file):
                self.validset = read_json_data(validset_file)
            else:
                self.validset = read_json_data(os.path.join(os.getcwd(),validset_file))#[:500]
            if os.path.isabs(testset_file):
                self.testset = read_json_data(testset_file)
            else:
                #exit()
                self.testset = read_json_data(os.path.join(os.getcwd(),testset_file))#[:500]

            if self.validset_divide is not True:
                self.testset = self.validset + self.testset
                self.validset = []


    def dataset_load(self):
        r"""dataset process and build vocab.

        when running k-fold setting, this function required to call once per fold.
        """
        parameters = self._build_vocab()
        if not self.resume_training and not self.from_pretrained:
            for key, value in parameters.items():
                setattr(self, key, value)
        if self.resume_training:
            self.resume_training=False
            
        if self.shuffle:
            random.shuffle(self.trainset)

    def parameters_to_dict(self):
        """
        return the parameters of dataset as format of dict.
        :return:
        """
        parameters_dict = {}
        for name, value in vars(self).items():
            if hasattr(eval('self.{}'.format(name)), '__call__') or re.match('__.*?__', name):
                continue
            else:
                parameters_dict[name] = copy.deepcopy(value)
        return parameters_dict

    def _preprocess(self):
        raise NotImplementedError

    def _build_vocab(self):
        raise NotImplementedError

    def _update_vocab(self, vocab_list):
        raise NotImplementedError

    def save_dataset(self,trained_dir):
        raise NotImplementedError

    @classmethod
    def load_from_pretrained(cls, pretrained_dir):
        raise NotImplementedError
