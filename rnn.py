# -*- encoding: utf-8 -*-
# @Author: Yunshi Lan
# @Time: 2023/02/10 15:22
# @File: ttt.py

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import copy

from torch import nn
from ...utils.enum_type import SpecialTokens

def tensor_ready(batch, tokenizer, is_train=False):
    source_list_batch = batch["source_list_batch"]
    target_list_batch = batch["target_list_batch"]
    source_max_len = np.max([len(sent) for sent in source_list_batch])
    target_max_len = np.max([len(sent) for sent in target_list_batch])+2

    text_list_batch = []
    tag_list_batch = []
    for idx, text_list in enumerate(source_list_batch):

        text_list = text_list
        tag_list = tokenizer.convert_tokens_to_ids([SpecialTokens.SOS_TOKEN])

        if is_train: # if the dataset is prepared for training
            tag_list += target_list_batch[idx]
            tag_list += tokenizer.convert_tokens_to_ids([SpecialTokens.EOS_TOKEN]) # Adding an End-Of-Sequence symbol (EOS) at the end of a sequence
            tag_list += tokenizer.convert_tokens_to_ids([SpecialTokens.PAD_TOKEN])*(target_max_len-len(target_list_batch[idx])) # padding to the largest length
            text_list += tokenizer.convert_tokens_to_ids([SpecialTokens.PAD_TOKEN])*(source_max_len-len(source_list_batch[idx])) # padding to the largest length

        text_list_batch.append(text_list)
        tag_list_batch.append(tag_list)

    batch["ready_source_batch"] = text_list_batch
    batch["ready_target_batch"] = tag_list_batch
    #print(tag_list_batch)
    #print('text_list_batch', len(text_list_batch), len(tag_list_batch))

    return batch

class RNN(nn.Module):
    def __init__(self, config, dataset):
        super().__init__()
        self.dropout = config["dropout"]
        self.device = config["device"]
        self.batch_size = config["batch_size"]
        self.embedding_size = config["embed_dim"]
        self.hidden_dim = config["hidden_dim"]
        self.gamma = config["gamma"]
        self.num_class = num_class = len(dataset.vocab)
        self.vocab = dataset
        self.padding_id = dataset.convert_tokens_to_ids(SpecialTokens.PAD_TOKEN) # 特殊pad符号对应的整数ID
        
        self.tokenizer = dataset
        dataset_size = len(dataset.vocab)
        
        # defien modules in the model
        self.in_tok_embed = nn.Embedding(dataset_size, self.embedding_size, self.padding_id) # the third parameter, the integer which this padding_id representing will be replaced by zero to make a difference from other words
        self.out_tok_embed = nn.Linear(self.embedding_size, dataset_size) # full linear
        self.out_tok_embed.weight = copy.deepcopy(self.in_tok_embed.weight) # deep copy the weight of embedding layer 
        self.out_mlp = nn.Linear(self.hidden_dim, self.embedding_size) # full linear，should * num_layers
        self.encoder = nn.LSTM(self.embedding_size, self.hidden_dim, batch_first=True) # lstm
        self.decoder = nn.LSTM(self.embedding_size, self.hidden_dim, batch_first=True) # lstm

    def fc_nll_loss(self, y_pred, y, y_mask, gamma=None, avg=True):
        # compute cross entropy loss
        if gamma is None:
            gamma = 2
        p = torch.gather(y_pred, 2, y.view(y.size(0), y.size(1), 1).long())
        g = (1-torch.clamp(p, min=0.01, max=0.99))**gamma
        #g = (1 - p) ** gamma 
        cost = -g * torch.log(p+1e-8)
        cost = cost.view(y.shape)
        y_mask = y_mask.view(y.shape)
        #print(torch.sum(y_mask, 0)); exit()
        if avg:
            cost = torch.sum(cost * y_mask, 1) / torch.sum(y_mask, 1)
        else:
            cost = torch.sum(cost * y_mask, 1)
        cost = cost.view((y.size(0), -1))
        return torch.mean(cost), g.view(y.shape)

    def forward(self, batch, dataloader, is_display):
        '''
        parameters in batch.
        batch['source_batch'] is a list with the source text sentences;
        batch['target_batch'] is a list with the target text sentences;
        batch['source_list_batch'] is the indexed source sentences;
        batch['target_list_batch'] is the indexed target sentences;
        '''

        # convert indexed sentences into the required format for the model training
        batch = tensor_ready(batch, self.vocab, is_train = True)

        # trauncate indexed sentences into a batch where the sentence lengths are same in the same batch
        source_data = dataloader.truncate_tensor(batch["ready_source_batch"])
        target_data = dataloader.truncate_tensor(batch["ready_target_batch"])

        # put tensor into gpu if there is one
        if self.device:
            source_data = source_data.cuda(self.device)
            target_data = target_data.cuda(self.device)

        # generate mask tensor to mask the padding position to exclude them during loss caculation
        source_mask_matrix = torch.sum(1 - torch.eq(source_data, self.padding_id).to(torch.int), -1) - 1 # the num of non_pad of the sequence
        in_mask_matrix = 1 - torch.eq(target_data, self.padding_id).to(torch.int) # a sequence, all non_pad become 1, all pad become 0 -> mask
        mask_matrix = in_mask_matrix.to(torch.bool)#.contiguous()# .cuda(self.device) # a sequence, all non_pad become True, all pad become False -> mask for transformer pay attention to the true/false
        #print('mask_matrix', mask_matrix)
        #tag_matrix = torch.LongTensor(target_data)#.contiguous()

        # convert indexed tokens into embeddings
        source_seq_rep = self.in_tok_embed(source_data) # embedding
        #source_seq_rep = [self.in_tok_embed(source_data) for data in source_data]
        target_seq_rep = self.in_tok_embed(target_data) # same embedding means same dictionary
        batch_size, seq_len, embedding_size = target_seq_rep.size()

        # initialize hidden state for the first time step
        h_0 = torch.randn(1, batch_size, self.hidden_dim) # num_layers, batch_size, hidden_dim
        c_0 = torch.randn(1, batch_size, self.hidden_dim)
        if self.device:
            h_0 = h_0.cuda(self.device)
            c_0 = c_0.cuda(self.device)

        # encode source data
        source_seq_rep, _ = self.encoder(source_seq_rep, (h_0, c_0)) # encoder in noamal situation, the first output will be the second parameter of decoder, the second output will be the third parameter of decoder
        gather_index = source_mask_matrix.view(batch_size, 1).repeat(1, self.hidden_dim) #(batch_size,)->(batch_size, hidden_dim)
        # without view, get the (batch(i), num of non_pad, i)
        # with view (batch_size, 1, self.hidden_dim)->(1, batch_size, self.hidden_dim)
        h_t = torch.gather(source_seq_rep, 1, gather_index.view(batch_size, 1, self.hidden_dim)).view(1, batch_size, self.hidden_dim)

        # decode target data
        target_seq_rep, _ = self.decoder(target_seq_rep, (h_t, c_0)) # to change num_layers, should replace h_t with (h_t.repeat(self.num_layers,1,1)
        target_seq_rep = F.dropout(target_seq_rep, p=self.dropout) # add dropout to avoid over fitting
        probs = torch.softmax(self.out_tok_embed(self.out_mlp(target_seq_rep)), -1) # linear->embedding->softmax
        #print(probs[:, :3, :]); exit()
        self.decode_result = probs.max(-1)[1]

        # conmpute loss 
        loss_ft_fc, g = self.fc_nll_loss(probs[:, :-1, :], target_data[:, 1:], mask_matrix[:, 1:], gamma=self.gamma)
        loss_dic = {"decode_result": self.decode_result,
                    "loss": loss_ft_fc}

        if is_display:
            print(self.decode_result[0]); #exit()
        #print('loss_dic', loss_dic)
        return loss_dic

    def model_test(self, batch, dataloader, is_display = False):
        # convert indexed sentences into the required format for the model testing
        batch = tensor_ready(batch, self.vocab)
        source_data = dataloader.truncate_tensor(batch["ready_source_batch"])
        target_data = dataloader.truncate_tensor(batch["ready_target_batch"])

        # put tensor into gpu if there is one
        if self.device:
            source_data = source_data.cuda(self.device)
            target_data = target_data.cuda(self.device)

        source_mask_matrix = torch.sum(1 - torch.eq(source_data, self.padding_id).to(torch.int), -1) - 1
        # convert indexed tokens into embeddings
        source_seq_rep = self.in_tok_embed(source_data)
        batch_size, seq_len, embedding_size = source_seq_rep.size()
        target_seq_rep = self.in_tok_embed(target_data).view(1, 1, embedding_size)

        # encode source data
        h_0 = torch.randn(1, batch_size, self.hidden_dim)
        c_0 = torch.randn(1, batch_size, self.hidden_dim)
        if self.device:
            h_0 = h_0.cuda(self.device)
            c_0 = c_0.cuda(self.device)

        source_seq_rep, _ = self.encoder(source_seq_rep, (h_0, c_0))
        c_t = c_0
        gather_index = source_mask_matrix.view(batch_size, 1).repeat(1, self.hidden_dim)
        h_t = torch.gather(source_seq_rep, 1, gather_index.view(batch_size, 1, self.hidden_dim)).view(1, batch_size, self.hidden_dim)

        # decode data step by step
        for i in range(seq_len):
            seq_rep, (h_t, c_t) = self.decoder(target_seq_rep, (h_t, c_t))
            probs = torch.softmax(self.out_tok_embed(self.out_mlp(seq_rep)), -1)
            target_data = probs.max(-1)[1]
            target_seq_rep = self.in_tok_embed(target_data).view(1, 1, embedding_size)
            self.decode_result += target_data

        self.decode_result = torch.stack(self.decode_result, -1).tolist()
        if is_display:
            print(self.decode_result); #exit()

        self.decode_result = post_process_decode_result(self.decode_result, self.tokenizer)
        return self.decode_result, target_data

def post_process_decode_result(sentences, tokenizer):
    #print(tokenizer.convert_tokens_to_ids([SpecialTokens.EOS_TOKEN]))
    new_sentences = []
    for sent_idx, sent in enumerate(sentences):
        new_sent = []
        for w in sent:
            if w in tokenizer.convert_tokens_to_ids([SpecialTokens.EOS_TOKEN]):
                break
            new_sent += [w]

        #print(tokenizer.convert_ids_to_tokens(new_sent))
        new_sentences += [new_sent]
    #print(sentences, new_sentences)
    return new_sentences


    