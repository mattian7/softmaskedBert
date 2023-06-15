# -*- encoding: utf-8 -*-
# @Author: Yunshi Lan
# @Time: 2023/02/10 15:22
# @File: ttt.py

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import copy
import time
from collections import defaultdict
from ...utils.enum_type import DatasetType
from ...utils.utils import time_since

import transformers
# 引入Bert的BertTokenizer与BertModel, 并单独取出BertModel中的词嵌入word_embeddings层
from transformers import BertConfig,BertModel, BertTokenizer



from ...utils.enum_type import SpecialTokens

def tensor_ready(batch, tokenizer, is_train=False):
    source_list_batch = batch["source_list_batch"]
    target_list_batch = batch["target_list_batch"]
    label_list_batch = batch["label"]
    source_max_len = np.max([len(sent) for sent in source_list_batch])+2
    target_max_len = np.max([len(sent) for sent in target_list_batch])+2
    label_max_len = np.max([len(sent) for sent in label_list_batch])+2

    text_list_batch = []
    tag_list_batch = []
    lb_list_batch = []
    ma_list_batch = []
    for idx, text_list in enumerate(source_list_batch):

        #text_list = text_list
        tag_list = tokenizer.convert_tokens_to_ids([SpecialTokens.SOS_TOKEN])
        text_list = tokenizer.convert_tokens_to_ids([SpecialTokens.SOS_TOKEN])
        label_list = [0] + [int(x) for x in label_list_batch[idx].strip().split()] + [0]
        mask_list = [1]*len(label_list)

        tag_list += target_list_batch[idx]
        tag_list += tokenizer.convert_tokens_to_ids([SpecialTokens.EOS_TOKEN])
        tag_list += tokenizer.convert_tokens_to_ids([SpecialTokens.PAD_TOKEN])*(target_max_len-len(target_list_batch[idx]))

        text_list += source_list_batch[idx]
        text_list += tokenizer.convert_tokens_to_ids([SpecialTokens.EOS_TOKEN])
        text_list += tokenizer.convert_tokens_to_ids([SpecialTokens.PAD_TOKEN])*(source_max_len-len(source_list_batch[idx]))

        label_list += tokenizer.convert_tokens_to_ids([SpecialTokens.PAD_TOKEN])*(label_max_len-len(label_list_batch[idx]))
        mask_list += tokenizer.convert_tokens_to_ids([SpecialTokens.PAD_TOKEN])*(label_max_len-len(label_list_batch[idx]))

        text_list_batch.append(text_list)
        tag_list_batch.append(tag_list)
        lb_list_batch.append(label_list)
        ma_list_batch.append(mask_list)

    batch["ready_source_batch"] = text_list_batch
    batch["ready_target_batch"] = tag_list_batch
    batch["ready_label_batch"] = lb_list_batch
    batch["ready_mask_batch"] = ma_list_batch

    return batch


class bert:
    def __init__(self, config, tokenizer):
        """
        初始化
        Args:
            config: 实例化的参数管理器
        """
        self.config = config
        self.bert = BertModel.from_pretrained(self.config["pretrained_model_path"])
        # 加载预训练的模型
        self.embedding = self.bert.embeddings  # 实例化BertEmbeddings类
        self.bert_encoder = self.bert.encoder
        # 实例化BertEncoder类，即attention结构，默认num_hidden_layers=12，也可以去本地bert模型的config.json文件里修改
        # 论文里也是12，实际运用时有需要再改
        # 查了源码，BertModel这个类还有BertEmbeddings、BertEncoder、BertPooler属性，在此之前我想获得bert embeddings都是直接用BertModel的call方法的，学习了
        self.tokenizer = tokenizer
        self.masked_e = self.embedding(torch.tensor([[self.tokenizer.mask_token_id]], dtype=torch.long))
        # 加载[mask]字符对应的编码，并计算其embedding
        self.vocab_size = self.tokenizer.vocab_size  # 词汇量


class biGruDetector(nn.Module):
    def __init__(self, input_size, hidden_size, num_layer=1):
        """
        类初始化
        Args:
            input_size: embedding维度
            hidden_size: gru的隐层维度
            num_layer: gru层数
        """
        super(biGruDetector, self).__init__()
        self.rnn = nn.GRU(input_size, hidden_size, num_layers=num_layer,
                          bidirectional=True, batch_first=True)
        # GRU层
        self.linear = nn.Linear(hidden_size * 2, 1)
        # 线性层
        # 因为双向GRU，所以输入维度是hidden_size*2；因为只需要输出个概率，所以第二个维度是1

    def forward(self, inp):
        rnn_output, _ = self.rnn(inp)
        # rnn输出output和最后的hidden state，这里只需要output；
        # 在batch_first设为True时，shape为（batch_size,sequence_length,2*hidden_size）;
        # 因为是双向的，所以最后一个维度是2*hidden_size。
        output = nn.Sigmoid()(self.linear(rnn_output))
        # sigmoid函数，没啥好说的，论文里就是这个结构
        return output
        # output维度是[batch_size, sequence_length, 1]


class softMaskedBert(nn.Module):
    """
    softmasked bert模型
    """
    def __init__(self, config, vocab_size, masked_e, bert_encoder):
        """
        类初始化
        Args:
            config: 实例化的参数管理器
        """
        super(softMaskedBert, self).__init__()
        self.config = config  # 加载参数管理器
        self.vocab_size = vocab_size
        self.masked_e = masked_e
        self.bert_encoder = bert_encoder

        self.linear = nn.Linear(self.config.embedding_size, self.vocab_size)  # 线性层，没啥好说的
        self.log_softmax = nn.LogSoftmax(dim=-1)


    def forward(self, bert_embedding, p, input_mask=None):
        """
        call方法
        Args:
            bert_embedding: 输入序列的bert_embedding
            p: 检测器的输出，表示输入序列对应位置的字符错误概率，维度：[batch_size, sequence_length, 1]
            input_mask: extended_attention_mask，不是单纯的输入序列的mask，具体使用方法见下面的代码注释
        Returns:
            模型输出，经过了softmax和log，维度[batch_size,sequence_length,num_vocabulary]
        """
        soft_bert_embedding = p * self.masked_e + (1 - p) * bert_embedding  # detector输出和[mask]的embedding加权求和
        bert_out = self.bert_encoder(hidden_states=soft_bert_embedding, attention_mask=input_mask)

        h = bert_out[0] + bert_embedding  # 残差
        out = self.log_softmax(self.linear(h))  # 线性层，再softmax输出

        return out


class RNN(nn.Module):
    def __init__(self, config, dataset):
        super().__init__()
        self.config = config
        self.dropout = config["dropout"]
        self.device = config["device"]
        self.batch_size = config["batch_size"]
        self.embedding_size = config["embed_dim"]
        self.hidden_dim = config["hidden_dim"]
        self.gamma = config["gamma"]
        self.lr = config["learning_rate"]
        self.num_class = num_class = len(dataset.vocab)
        self.vocab = dataset
        self.padding_id = dataset.convert_tokens_to_ids(SpecialTokens.PAD_TOKEN)  # 特殊pad符号对应的整数ID

        self.tokenizer = dataset


        self.bert = BertModel.from_pretrained(self.config["pretrained_model_path"])
        self.embedding = self.bert.embeddings  # 实例化BertEmbeddings类

        self.bert_encoder = self.bert.encoder
        self.vocab_size = self.tokenizer.vocab_size
        self.masked_e = self.embedding(torch.tensor([[self.tokenizer.mask_token_id]], dtype=torch.long))

        self.detector_model = biGruDetector(self.embedding_size, self.hidden_dim)  # 实例化检测器
        #self.detector_optimizer = torch.optim.Adam(self.detector_model.parameters(), lr=self.config.lr)  # 检测器的优化器
        self.detector_criterion = nn.BCELoss()  # 检测器部分的损失，Binary CrossEntropy
        self.detector_optimizer = torch.optim.Adam(self.detector_model.parameters(), lr=self.lr)

        self.model = softMaskedBert(self.config, self.vocab_size, self.masked_e.to(self.device), self.bert_encoder)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

        self.criterion = nn.NLLLoss()
        self.gamma2 = 0.7


    def forward(self, batch, dataloader, is_display):

        batch = tensor_ready(batch, self.vocab, is_train=True)
        batch_inp_ids = batch["ready_source_batch"]
        batch_out_ids = batch["ready_target_batch"]
        batch_labels = ["ready_label_batch"]
        batch_mask = batch["ready_mask_batch"]

        batch_inp_embedding = self.embedding(batch_inp_ids).to(self.device)
        batch_out_ids = batch_out_ids.to(self.device)  # 选择计算设备，下同
        batch_labels = batch_labels.to(self.device)
        batch_mask = batch_mask.to(self.device)


        prob = self.detector_model(batch_inp_embedding)  # 检测器模块的输出

        detector_loss = self.detector_criterion(prob.squeeze() * batch_mask, batch_labels.float())

        out = self.model(batch_inp_embedding, prob, self.bert.get_extended_attention_mask(batch_mask, batch_out_ids.shape, batch_inp_embedding.device))
        # 这个get_extended_attention_mask来自BertModel继承的类，
        # 官方手册里也没介绍，当然一般也不会介绍这些具体实现，看了源码之后，直接拿出来调用

        model_loss = self.criterion((out * batch_mask.unsqueeze(-1)).reshape(-1, out.shape[-1]),
                                    batch_out_ids.reshape(-1))

        loss = self.gama2 * model_loss + (1 - self.gama2) * detector_loss

        self.decode_result = torch.argmax(out, dim=2)
        loss_dic = {"decode_result": self.decode_result,
                    "loss": loss}

        if is_display:
            print(self.decode_result);  # exit()
        # print('loss_dic', loss_dic)
        return loss_dic
        #return loss

    def model_test(self, batch, dataloader, is_display=False):
        batch = tensor_ready(batch, self.vocab, is_train=True)
        batch_inp_ids = batch["ready_source_batch"]
        batch_out_ids = batch["ready_target_batch"]
        batch_labels = ["ready_label_batch"]
        batch_mask = batch["ready_mask_batch"]

        batch_inp_embedding = self.embedding(batch_inp_ids).to(self.device)
        batch_out_ids = batch_out_ids.to(self.device)  # 选择计算设备，下同
        batch_labels = batch_labels.to(self.device)
        batch_mask = batch_mask.to(self.device)

        prob = self.detector_model(batch_inp_embedding)  # 检测器模块的输出

        detector_loss = self.detector_criterion(prob.squeeze() * batch_mask, batch_labels.float())

        out = self.model(batch_inp_embedding, prob,
                         self.bert.get_extended_attention_mask(batch_mask, batch_out_ids.shape,
                                                               batch_inp_embedding.device))
        # 这个get_extended_attention_mask来自BertModel继承的类，
        # 官方手册里也没介绍，当然一般也不会介绍这些具体实现，看了源码之后，直接拿出来调用

        model_loss = self.criterion((out * batch_mask.unsqueeze(-1)).reshape(-1, out.shape[-1]),
                                    batch_out_ids.reshape(-1))

        loss = self.gama2 * model_loss + (1 - self.gama2) * detector_loss

        self.decode_result = torch.argmax(out, dim=2)
        loss_dic = {"decode_result": self.decode_result,
                    "loss": loss}

        if is_display:
            print(self.decode_result);  # exit()


        self.decode_result = post_process_decode_result(self.decode_result, self.tokenizer)
        return self.decode_result, batch["ready_target_batch"]


def post_process_decode_result(sentences, tokenizer):
    # print(tokenizer.convert_tokens_to_ids([SpecialTokens.EOS_TOKEN]))
    new_sentences = []
    for sent_idx, sent in enumerate(sentences):
        new_sent = []
        for w in sent:
            if w in tokenizer.convert_tokens_to_ids([SpecialTokens.EOS_TOKEN]):
                break
            new_sent += [w]

        # print(tokenizer.convert_ids_to_tokens(new_sent))
        new_sentences += [new_sent]
    # print(sentences, new_sentences)
    return new_sentences


