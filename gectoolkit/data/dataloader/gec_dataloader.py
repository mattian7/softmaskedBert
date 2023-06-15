# -*- encoding: utf-8 -*-
# @Author: Yunshi Lan
# @Time: 2023/01/30 11:19
# @File: gec_dataloader.py


import math
import torch
from typing import List
import numpy as np
import re

from ...config import Config
from ..dataset.abstract_dataset import AbstractDataset
from ..dataloader.abstract_dataloader import AbstractDataLoader
from ...utils.enum_type import SpecialTokens


from transformers import AutoTokenizer


class GECDataLoader(AbstractDataLoader):
    """dataloader class for deep-learning model EPT

    """
    def __init__(self, config:Config, dataset:AbstractDataset):
        super().__init__(config, dataset)

        self.trainset_nums = len(dataset.trainset)
        self.validset_nums = len(dataset.validset)
        self.testset_nums = len(dataset.testset)
        self.max_input_len = config["max_input_len"]

        if config["dataset"] in ['csc']:
            self.pretrained_tokenzier = AutoTokenizer.from_pretrained(config["pretrained_model_path"])
            special_tokens = [SpecialTokens.__dict__[k] for k in SpecialTokens.__dict__ if not re.search('^\_', k)]
            special_tokens.sort()
            self.pretrained_tokenzier.add_special_tokens({'additional_special_tokens': special_tokens})
        else:
            self.pretrained_tokenzier = AutoTokenizer.from_pretrained(config["pretrained_model_path"])
            self.pretrained_tokenzier.add_special_tokens({'additional_special_tokens': ['[N]']})

        dataset = self.pretrained_tokenzier
        self.model = config["model"].lower()

        self.__init_batches()

    def __build_batch(self, batch_data):
        """load one batch

        Args:
            batch_data (list[dict])
        
        Returns:
            loaded batch data (dict)
        """
        source_list_batch = []
        target_list_batch = []
        source_batch = []
        target_batch = []
        #print('self.max_len', self.max_len); exit()

        for data in batch_data:
            source = [w for w in data['source_text'].strip()][:self.max_input_len]
            target = [w for w in data['target_text'].strip()][:self.max_input_len]

            sor_list = list()
            for token in source:
                sor_list.append(self.pretrained_tokenzier.convert_tokens_to_ids(token))

            tag_list = list()
            for token in target:
                tag_list.append(self.pretrained_tokenzier.convert_tokens_to_ids(token))

            source_batch.append(source)
            target_batch.append(target)
            source_list_batch.append(sor_list)
            target_list_batch.append(tag_list)

        return {
            "source_batch": source_batch,
            "target_batch": target_batch,
            "source_list_batch": source_list_batch,
            "target_list_batch": target_list_batch
        }

    def __init_batches(self):
        self.trainset_batches = []
        self.validset_batches = []
        self.testset_batches = []
        for set_type in ['train', 'valid', 'test']:
            if set_type == 'train':
                datas = self.dataset.trainset
                batch_size = self.train_batch_size
            elif set_type == 'valid':
                datas = self.dataset.validset
                batch_size = self.test_batch_size
            elif set_type == 'test':
                datas = self.dataset.testset
                batch_size = self.test_batch_size
            else:
                raise ValueError("{} type not in ['train', 'valid', 'test'].".format(type))
            num_total = len(datas)
            batch_num = math.ceil(num_total / batch_size)
            for batch_i in range(batch_num):
                start_idx = batch_i * batch_size
                end_idx = (batch_i + 1) * batch_size
                if end_idx <= num_total:
                    batch_data = datas[start_idx:end_idx]
                else:
                    batch_data = datas[start_idx:num_total]
                built_batch = self.__build_batch(batch_data)
                if set_type == 'train':
                    self.trainset_batches.append(built_batch)
                elif set_type == 'valid':
                    self.validset_batches.append(built_batch)
                elif set_type == 'test':
                    self.testset_batches.append(built_batch)
                else:
                    raise ValueError("{} type not in ['train', 'valid', 'test'].".format(type))
        self.__trainset_batch_idx = -1
        self.__validset_batch_idx = -1
        self.__testset_batch_idx = -1
        self.trainset_batch_nums = len(self.trainset_batches)
        self.validset_batch_nums = len(self.validset_batches)
        self.testset_batch_nums = len(self.testset_batches)
        #exit()

    def build_batch_for_predict(self, batch_data: List[dict]):
        raise NotImplementedError


    def truncate_tensor(self, sequence):
        max_len = 0
        for instance in sequence:
            max_len = max(len(instance), max_len)
        result_batch_tag_list = list()
        for instance in sequence:
            one_tag_list = []
            one_tag_list.extend(instance)
            len_diff = max_len - len(one_tag_list)
            for _ in range(len_diff):
                one_tag_list.append(self.pretrained_tokenzier.convert_tokens_to_ids('<-PAD->')) # for padding
            result_batch_tag_list.append(one_tag_list)

        result_batch_tag_matrix = np.array(result_batch_tag_list)
        result_batch_tag_matrix = torch.tensor(result_batch_tag_matrix)

        return result_batch_tag_matrix