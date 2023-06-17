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
from torch.nn.utils.rnn import pad_sequence

import transformers
from transformers import BertConfig, BertModel, BertTokenizer



from ...utils.enum_type import SpecialTokens

def tensor_ready(batch, tokenizer, is_train=False):
    
    source_list_batch = batch["source_list_batch"]
    target_list_batch = batch["target_list_batch"]
    label_list_batch = batch["label_list_batch"]
    mask_list_batch = batch["mask_list_batch"]
    #print(source_list_batch)
    batch["ready_source_batch"] = pad_sequence([torch.tensor(x) for x in source_list_batch],batch_first=True)
    batch["ready_target_batch"] = pad_sequence([torch.tensor(x) for x in target_list_batch],batch_first=True)
    batch["ready_label_batch"] = pad_sequence([torch.tensor(x).float() for x in label_list_batch],batch_first=True)
    batch["ready_mask_batch"] = pad_sequence([torch.tensor(x).float() for x in mask_list_batch],batch_first=True)

    return batch


class bert:
    def __init__(self, config, tokenizer):
        self.config = config
        self.bert = BertModel.from_pretrained(self.config["pretrained_model_path"])
        self.embedding = self.bert.embeddings 
        self.bert_encoder = self.bert.encoder
        self.tokenizer = tokenizer
        self.masked_e = self.embedding(torch.tensor([[self.tokenizer.mask_token_id]], dtype=torch.long))
        self.vocab_size = self.tokenizer.vocab_size  

class biGruDetector(nn.Module):
    def __init__(self, input_size, hidden_size, num_layer=1):
        super(biGruDetector, self).__init__()
        self.rnn = nn.GRU(input_size, hidden_size, num_layers=num_layer,
                          bidirectional=True, batch_first=True)
        self.linear = nn.Linear(hidden_size * 2, 1)

    def forward(self, inp):
        rnn_output, _ = self.rnn(inp)
        output = nn.Sigmoid()(self.linear(rnn_output))
        return output


class softMaskedBert(nn.Module):
    def __init__(self, config, vocab_size, masked_e):
        super(softMaskedBert, self).__init__()
        self.config = config 
        self.vocab_size = vocab_size
        self.masked_e = masked_e
        self.bert_encoder = BertModel.from_pretrained(self.config["pretrained_model_path"]).encoder
        self.linear = nn.Linear(self.config["embed_dim"], self.vocab_size) 
        self.log_softmax = nn.LogSoftmax(dim=-1)


    def forward(self, bert_embedding, p, input_mask=None):
        soft_bert_embedding = p * self.masked_e + (1 - p) * bert_embedding  
        bert_out = self.bert_encoder(hidden_states=soft_bert_embedding, attention_mask=input_mask)

        h = bert_out[0] + bert_embedding 
        out = self.log_softmax(self.linear(h)) 

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
        self.padding_id = dataset.convert_tokens_to_ids(SpecialTokens.PAD_TOKEN) 
        embedding = BertModel.from_pretrained(self.config["pretrained_model_path"]).embeddings
        self.tokenizer = dataset
        self.vocab_size = self.tokenizer.vocab_size
        self.masked_e = embedding(torch.tensor([[self.tokenizer.mask_token_id]], dtype=torch.long))
        
        self.detector_model = biGruDetector(self.embedding_size, self.hidden_dim) 
        self.detector_model.to(self.device)
        self.detector_criterion = nn.BCELoss()  

        self.model = softMaskedBert(self.config, self.vocab_size, self.masked_e.to(self.device))
        self.model.to(self.device)

        self.criterion = nn.NLLLoss()
        self.gamma2 = 0.7

    def forward(self, batch, dataloader, bert, embedding, is_display):
        
        self.detector_model.train()
        self.model.train() 
        batch = tensor_ready(batch, self.vocab, is_train=True)
        batch_inp_ids = batch["ready_source_batch"]
        batch_out_ids = batch["ready_target_batch"]
        batch_labels = batch["ready_label_batch"]
        batch_mask = batch["ready_mask_batch"]
        
        #batch_inp_ids = batch_inp_ids.to(self.device)
        
        batch_inp_embedding = embedding(batch_inp_ids).to(self.device)
        batch_out_ids = batch_out_ids.to(self.device) 
        batch_labels = batch_labels.to(self.device)
        batch_mask = batch_mask.to(self.device)


        prob = self.detector_model(batch_inp_embedding)

        detector_loss = self.detector_criterion(prob.squeeze() * batch_mask, batch_labels.float())
        
        out = self.model(batch_inp_embedding, prob, bert.get_extended_attention_mask(batch_mask, batch_out_ids.shape, batch_inp_embedding.device))
        model_loss = self.criterion((out * batch_mask.unsqueeze(-1)).reshape(-1, out.shape[-1]),
                                    batch_out_ids.reshape(-1))
        model_loss = model_loss.to(self.device)
        detector_loss = detector_loss.to(self.device)
            
        loss = self.gamma2 * model_loss + (1 - self.gamma2) * detector_loss
        #loss = model_loss
        loss = loss.to(self.device)
        
        
        self.decode_result = torch.argmax(out, dim=2)
        loss_dic = {"decode_result": self.decode_result,
                    "loss": loss}

        if is_display:
            print(self.decode_result);  # exit()
        
        
        return loss_dic

    def model_test(self, batch, dataloader, bert, embedding, is_display=False):
        
        self.detector_model.eval()
        self.model.eval() 

        batch = tensor_ready(batch, self.vocab, is_train=False)
        
        batch_inp_ids = batch["ready_source_batch"]
        batch_out_ids = batch["ready_target_batch"]
        batch_labels = batch["ready_label_batch"]
        batch_mask = batch["ready_mask_batch"]
        
        batch_inp_embedding = embedding(batch_inp_ids).to(self.device)
        batch_out_ids = batch_out_ids.to(self.device)  
        batch_labels = batch_labels.to(self.device)
        batch_mask = batch_mask.to(self.device)

        prob = self.detector_model(batch_inp_embedding)  

        
        input_mask = bert.get_extended_attention_mask(batch_mask, batch_inp_ids.shape, batch_inp_embedding.device)
        out = self.model(batch_inp_embedding, prob, input_mask)
        
        
        self.decode_result = torch.argmax(out, dim=-1)

        if is_display:
            print(self.decode_result);  


        self.decode_result = post_process_decode_result(self.decode_result, self.tokenizer)
        return self.decode_result, batch["ready_target_batch"]

    
def post_process_decode_result(sentences, tokenizer):
    new_sentences = []
    for sent_idx, sent in enumerate(sentences):
        new_sent = []
        for w in sent[1:]:
            if w in tokenizer.convert_tokens_to_ids([SpecialTokens.SEP_TOKEN]):
                break
            if w in tokenizer.convert_tokens_to_ids([SpecialTokens.CLS_TOKEN]):
                continue
            new_sent += [w]
        new_sent = new_sent
        new_sentences += [new_sent]
    return new_sentences


