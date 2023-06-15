# -*- encoding: utf-8 -*-
# @Author: Yunshi Lan
# @Time: 2023/02/10 15:22
# @File: ttt.py

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import copy

import transformers
# 引入Bert的BertTokenizer与BertModel, 并单独取出BertModel中的词嵌入word_embeddings层
from transformers import BertConfig,BertModel, BertTokenizer
# 引入Bert模型的基础类BertEmbeddings, BertEncoder,BertPooler,BertPreTrainedModel
from transformers.modeling_bert import BertEmbeddings, BertEncoder, BertPooler,BertPreTrainedModel, BertLMPredictionHead



from ...utils.enum_type import SpecialTokens

def tensor_ready(batch, tokenizer, is_train=False):
    source_list_batch = batch["source_list_batch"]
    target_list_batch = batch["target_list_batch"]
    source_max_len = np.max([len(sent) for sent in source_list_batch])+2
    target_max_len = np.max([len(sent) for sent in target_list_batch])+2

    text_list_batch = []
    tag_list_batch = []
    for idx, text_list in enumerate(source_list_batch):

        #text_list = text_list
        tag_list = tokenizer.convert_tokens_to_ids([SpecialTokens.SOS_TOKEN])
        text_list = tokenizer.convert_tokens_to_ids([SpecialTokens.SOS_TOKEN])

        attention_mask = [1] * len(text_list+2) + [0] * (source_max_len - text_list - 2)
        position_ids = [id for id in range(source_max_len)]
        type_ids = [0] * source_max_len
        tag_list += target_list_batch[idx]
        tag_list += tokenizer.convert_tokens_to_ids([SpecialTokens.EOS_TOKEN])
        tag_list += tokenizer.convert_tokens_to_ids([SpecialTokens.PAD_TOKEN])*(target_max_len-len(target_list_batch[idx]))

        text_list += source_list_batch[idx]
        text_list += tokenizer.convert_tokens_to_ids([SpecialTokens.EOS_TOKEN])
        text_list += tokenizer.convert_tokens_to_ids([SpecialTokens.PAD_TOKEN])*(source_max_len-len(source_list_batch[idx]))


        text_list_batch.append(text_list)
        tag_list_batch.append(tag_list)

    batch["ready_source_batch"] = text_list_batch
    batch["ready_target_batch"] = tag_list_batch

    return batch


class RNN(BertPreTrainedModel):
    def __init__(self, config, dataset):
        super().__init__(config)
        # self.config中包含了拼写错误纠正网络Correction_Network中的Bert模型的各种配置超参数.
        self.config = config
        self.dropout = config["dropout"]
        self.device = config["device"]
        self.batch_size = config["batch_size"]
        self.embedding_size = config["embed_dim"]
        self.hidden_dim = config["hidden_dim"]
        self.gamma = config["gamma"]
        self.num_class = num_class = len(dataset.vocab)
        self.vocab = dataset
        self.padding_id = dataset.convert_tokens_to_ids(SpecialTokens.PAD_TOKEN)
        #  获取此时传入的Tokenizer类(此时的Tokenizer类为BertTokenizer类), 利用此时获得的Tokenizer类
        # self.tokenizer中的mask_token_id属性来获取遮罩特殊符[MASK]在Bert模型的嵌入层BertEmbeddings()中的
        # 词嵌入层参数矩阵word_embeddings中所对应索引的嵌入向量(embeddins vector).
        # self.tokenizer = tokenizer
        '''此时self.mask_token_id为遮罩特殊符[MASK]在Bert模型的嵌入层BertEmbeddings()中的
           词嵌入层参数矩阵word_embeddings中所对应的索引.'''
        self.mask_token_id = 103

        '''一、构建错误探查网络Detection_Network中所需的网络层'''

        # Bi-GRU网络作为错误探查网络Detection_Network的编码器
        # 此处由于BertModel中的embeddings层中所有子嵌入模块的嵌入维度都为768, 所以此处Bi-GRU网络的input_size也为768,
        # 而将Bi-GRU网络的hidden_size设为256,是为了保证Bi-GRU网络双向编码后双向隐藏层拼接到一块后隐藏层维度能保持在512.
        # 此时enc_hid_size为512.
        self.enc_bi_gru = torch.nn.GRU(input_size=768, hidden_size=256, dropout=0.2, bidirectional=True)

        # 双向GRU编码层对于输入错误探查网络Detection_Network中的input_embeddings进行双向编码,
        # 此时双向GRU编码层的输出为(seq_len, batch_size, enc_hid_size * 2),将其交换维度变形为(batch_size, seq_len, enc_hid_size * 2),
        # 再将双向GRU编码层的变形后的输出输入self.detection_network_dense_out层中,映射为形状(batch_size, seq_len, 2)的张量,
        # 这样方便后面进行判断句子序列中每一个字符是否为拼写错误字符的二分类任务的交叉熵损失值计算.
        self.detection_network_dense_out = torch.nn.Linear(512, 2)

        # 同时,将双向GRU编码层输出后经过变形的形状为(batch_size, seq_len, enc_hid_size * 2),的张量输入进soft_masking_coef_mapping层中,
        # 将其形状映射为(batch_size, seq_len, 1)的张量，此张量再在后面输入进Sigmoid()激活函数中, 将此张量的值映射至(0,1)之间，
        # 这样这个张量即变为了后面计算soft-masked embeddings时和mask_embeddings相乘的系数p (结果pi即可表示为文本序列中第i处的字符拼写错误的似然概率(likelihood)).
        self.soft_masking_coef_mapping = torch.nn.Linear(512, 1)

        '''二、构建的拼写错误纠正网络Correction_Network中BertModel中所用的个三种网络层'''

        '''
        (1): 嵌入层BertEmbeddings(),其中包含了每个character的word embedding、segment embeddings、position embedding三种嵌入函数.
        (2): Bert模型的核心,多层(12层)多头自注意力(multi-head self attention)编码层BertEncoder.
        (3): Bert模型最后的池化层BertPooler.
        '''
        # 嵌入层BertEmbeddings().
        self.embeddings = BertEmbeddings(config)
        # 多层(12层)多头自注意力(multi-head self attention)编码层BertEncoder.
        self.encoder = BertEncoder(config)
        # 池化层BertPooler。
        self.pooler = BertPooler(config)
        # 初始化权重矩阵,偏置等.
        self.init_weights()

        '''获取遮罩特殊符[MASK]在Bert模型的嵌入层BertEmbeddings()中的词嵌入层word_embeddings层中特殊符[MASK]所对应索引的嵌入向量(embeddins vector).'''
        # 在Bert模型的tokenizer类BertTokenizer()的词表中,遮罩特殊符[MASK]会被编码为索引103(只要是BertTokenizer()类,无论其from_pretrained哪种
        # 预训练的Bert模型词表,遮罩特殊符[MASK]在词表中的索引都为103; 除非换预训练模型如换成Albert模型,遮罩特殊符[MASK]在词表中的索引才会变, 否则
        # 遮罩特殊符[MASK]在同一类预训练Bert模型的词表下索引不变).
        # 在之后, 遮罩特殊符[MASK]的张量self.mask_embedding的形状要变为和Bert模型嵌入层BertEmbeddings()的输出input_embeddings张量的形状一样,
        # 此时self.mask_embeddings张量的形状要为(batch_size, seq_len, embed_size)->(batch_size, seq_len, 768).

        # 此时,self.mask_embeddings张量的形状为(768,)，将遮罩特殊符[MASK]对应的索引嵌入向量(embeddins vector)的形状降维为(768,),
        # 之后会将遮罩特殊符[MASK]对应的索引嵌入向量(embeddins vector)的形状由(768,)再扩展为(batch_size, seq_len, 768);
        # 此外，此时也对遮罩特殊符[MASK]对应的索引嵌入向量(embeddins vector)self.mask_embeddings使用detach()函数
        # 进行计算图的截断, 防止梯度信息在反向传播时传到self.mask_embeddings这里.
        # self.mask_embeddings = self.embeddings.word_embeddings.weight[ self.tokenizer.mask_token_id() ] # 老写法
        # self.mask_embeddings = self.embeddings( torch.tensor([[ self.tokenizer.mask_token_id ]]).long() ).squeeze().detach()
        self.mask_embeddings = None  # 先置为None，后续在forward()函数中进行获取遮罩特殊符[MASK]对应的索引嵌入向量(embeddins vector)的操作.

        # 注意!: 在soft_masked_embeddings输入拼写错误纠正网络correction network中的Bert模型后,其计算结果输入进最终的输出层与Softmax层之前，
        # 拼写错误纠正网络correction network的结果需通过残差连接residual connection与输入模型一开始的input embeddings相加，
        # 相加的结果才输入最终的输出层与Softmax层中做最终的正确字符预测。
        '''self.cls即为拼写错误纠正网络correction network之后的输出层, 其使用的为transformers.modeling_bert模块中预置的模型类
           BertLMPredictionHead(config)，BertLMPredictionHead(config)模型类会将经过残差连接模块residual connection之后
           的输出的维度由768投影到纠错词表的索引空间. (此处输出层self.cls的输出即可被视为Soft_Masked_BERT模型的最终输出)'''
        self.cls = BertLMPredictionHead(config)

        '''此处可不写最后的Softmax()函数, 因为若之后在训练模型时使用CrossEntropyLoss()交叉熵函数来计算损失值的话, CrossEntropyLoss()函数
           中默认会对输入进行Softmax()计算.'''

    #     def set_tokenizer(self, tokenizer):
    #         self.tokenizer = tokenizer

    '''下方三个函数为BertModel类中自带的函数,放在此处是为了和源BertModel类保持一致防止出错. '''

    def get_input_embeddings(self):
        return self.embeddings.word_embeddings

    def set_input_embeddings(self, value):
        self.embeddings.word_embeddings = value

    def _prune_heads(self, heads_to_prune):
        """ Prunes heads of the model.
            heads_to_prune: dict of {layer_num: list of heads to prune in this layer}
            See base class PreTrainedModel
        """
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)

    '''构建错误探查网络Detection_Network'''

    def Detection_Network(self, input_embeddings: torch.Tensor, attention_mask: torch.Tensor, device):
        # 此时输入错误探查网络Detection_Network中的input_embeddings张量形状为:(seq_len, batch_size, embed_size)->(seq_len, batch_size, 768)
        # attention_mask张量形状为:(batch_size, seq_len)
        # 输入模型起始处的嵌入张量input embedding由一句sentence中每个character的word embedding、position embedding、segment embeddings三者相加而成.

        # 初始化错误探查网络Detection_Network的双向GRU的初始隐藏状态h_0,
        # 将双向GRU的初始隐藏状态h_0放入当前的device中.
        h_0 = torch.zeros(2, input_embeddings.shape[1], 256).to(device)

        # 此时双向GRU层self.enc_bi_gru的输出为一个元组,元组中第一个元素为最后的隐藏层输出张量,第二个元素为最后一个时间步的隐藏状态h_n,此处仅需最后的隐藏层输出张量;
        # 此时错误探查网络Detection_Network的双向GRU编码层最后的隐藏层输出张量bi_gru_final_hidden_layer的形状为(seq_len, batch_size, enc_hid_size * 2).
        bi_gru_final_hidden_layer = self.enc_bi_gru(input_embeddings, h_0)[0]
        # 将隐藏层输出张量bi_gru_final_hidden_layer的第一第二维度互换,形状变为(batch_size, seq_len, enc_hid_size * 2)
        bi_gru_final_hidden_layer = bi_gru_final_hidden_layer.permute(1, 0, 2)

        # 双向GRU编码层对于输入错误探查网络Detection_Network中的input_embeddings进行双向编码,
        # 此时双向GRU编码层的输出为(seq_len, batch_size, enc_hid_size * 2),将其交换维度变形为(batch_size, seq_len, enc_hid_size * 2),
        # 再将双向GRU编码层的变形后的输出输入self.detection_network_dense_out层中,映射为形状(batch_size, seq_len, 2)的张量detection_network_output,
        # 这样方便后面进行判断句子序列中每一个字符是否为拼写错误字符的二分类任务的交叉熵损失值计算.
        detection_network_output = self.detection_network_dense_out(
            bi_gru_final_hidden_layer)  # 形状为(batch_size, seq_len, 2)

        # 同时,将双向GRU编码层输出后经过变形的形状为(batch_size, seq_len, enc_hid_size * 2),的张量输入进soft_masking_coef_mapping层中,
        # 将其形状映射为(batch_size, seq_len, 1)的张量，此张量再在后面输入进Sigmoid()激活函数中, 将此张量的值映射至(0,1)之间，
        # 这样这个张量即变为了后面计算soft-masked embeddings时和mask_embeddings相乘的系数p (结果pi即可表示为文本序列中第i处的字符拼写错误的似然概率(likelihood)).
        # 此时soft_masking_coefs张量可被称为：soft-masking系数张量, 其形状为(batch_size, seq_len, 1).
        soft_masking_coefs = torch.nn.functional.sigmoid(
            self.soft_masking_coef_mapping(bi_gru_final_hidden_layer))  # (batch_size, seq_len, 1)

        # 此时将attention_mask张量形状变为(batch_size, seq_len,1),即令此时attention_mask张量的形状与soft-masking系数张量soft_masking_coefs的形状保持一致.
        attention_mask = attention_mask.unsqueeze(dim=2)

        # 利用attention_mask填充符逻辑指示张量,将soft-masking系数张量soft_masking_coefs中,seq_len上为"填充特殊符[PAD]索引"的位置的
        # soft-masking系数变为0, 这样soft-masking系数张量soft_masking_coefs中"填充特殊符[PAD]索引"的位置在后面生成soft-masked embeddings时,
        # "填充特殊符[PAD]索引"位置处的mask_embeddings系数即为0, input_embeddings系数即为1,这样即令"填充特殊符[PAD]索引"位置处保持input_embeddings
        # 的值不变。
        # 由于此时attention_mask张量中, 非特殊填充符的位置指示值为1,特殊填充符的位置指示值为0,因此在此处要用反向选择操作:
        # soft_masking_coefs[~attention_mask],来让特殊填充符的位置指示值反转为1,以达到选中特殊填充符的位置并给其赋值0的目的。
        attention_mask = (attention_mask != 0)  # 将attention_mask张量从1/0变为True/False,方便进行下一步的反向选择操作.
        soft_masking_coefs[~attention_mask] = 0

        return detection_network_output, soft_masking_coefs

    '''构建Soft Masking Connection连接模块.'''

    # 在错误探查网络error detection network输出一个句子中每个位置的字符为错误拼写字符的概率之后，利用此概率作为[MASK] embeddings的权重，
    # 而1减去这个概率作为句子中每个字符character的input embeddings的权重，[MASK] embeddings乘以权重的结果再加上input embeddings乘以权重的结果后
    # 所得到的嵌入结果soft-masked embeddings即为之后的错误纠正网络error correction network的输入。
    def Soft_Masking_Connection(self, input_embeddings: torch.Tensor,
                                mask_embeddings: torch.Tensor,
                                soft_masking_coefs: torch.Tensor):

        # 此时输入Soft_Masking_Connection模块中:
        # input_embeddings张量形状为:(batch_size, seq_len, embed_size)->(batch_size, seq_len, 768);
        # mask_embeddings为只包含"遮罩特殊符[MASK]"的embedding嵌入的张量,其形状也为:(batch_size, seq_len, embed_size)->(batch_size, seq_len, 768)；
        # soft_masking_coefs张量可被称为：soft-masking系数张量, 其为计算soft-masked embeddings时和mask_embeddings相乘的系数p的张量,形状为(batch_size, seq_len, 1);
        # 输入模型起始处的嵌入张量input embedding由一句sentence中每个character的word embedding、position embedding、segment embeddings三者相加而成.

        # 得到soft-masking系数张量:soft_masking_coefs张量之后,利用soft_masking_coefs张量作为[MASK] embeddings的权重，
        # 而1减去这个概率作为句子中每个字符character的input embeddings的权重，[MASK] embeddings乘以权重的结果再加上
        # input embeddings乘以权重的结果后所得到的嵌入结果soft-masked embeddings即为之后的错误纠正网络error correction network的输入.
        # 此时soft_masked_embeddings形状也为(batch_size, seq_len, embed_size)->(batch_size, seq_len, 768),
        soft_masked_embeddings = soft_masking_coefs * mask_embeddings + (1 - soft_masking_coefs) * input_embeddings

        return soft_masked_embeddings

    '''forward函数.'''

    def forward(self,batch, dataloader, is_display, input_ids=None, attention_mask=None, token_type_ids=None, position_ids=None,
                head_mask=None, inputs_embeds=None, encoder_hidden_states=None, encoder_attention_mask=None,
                output_attentions=None, device=None, ):

        '''以下部分为transformers库中BertModel类中的forward()部门的一小部分源码, 放在此处是为了和源BertModel类保持一致防止出错.'''
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        device = input_ids.device if input_ids is not None else inputs_embeds.device

        if attention_mask is None:
            attention_mask = torch.ones(input_shape, device=device)
        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)
        if position_ids is None:
            position_ids = torch.arange(input_shape[1], dtype=torch.long, device=device).unsqueeze(dim=0).repeat(
                input_shape[0], 1)

        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        extended_attention_mask: torch.Tensor = self.get_extended_attention_mask(attention_mask, input_shape, device)

        # If a 2D ou 3D attention mask is provided for the cross-attention
        # we need to make broadcastabe to [batch_size, num_heads, seq_length, seq_length]
        if self.config.is_decoder and encoder_hidden_states is not None:
            encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states.size()
            encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)
            if encoder_attention_mask is None:
                encoder_attention_mask = torch.ones(encoder_hidden_shape, device=device)
            encoder_extended_attention_mask = self.invert_attention_mask(encoder_attention_mask)
        else:
            encoder_extended_attention_mask = None

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)
        '''以上部分为transformers库中BertModel类中的forward()部门的一小部分源码, 放在此处是为了和源BertModel类保持一致防止出错.'''

        # 若当前未传入device，则自动判断此时的device环境.
        if not device:
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # 利用张量的long()函数确保这些张量全为int型张量.
        input_ids = input_ids.long().to(device)
        attention_mask = attention_mask.long().to(device)
        token_type_ids = token_type_ids.long().to(device)
        position_ids = position_ids.long().to(device)

        '''获取遮罩特殊符[MASK]在Bert模型的嵌入层BertEmbeddings()中的词嵌入层word_embeddings层中特殊符[MASK]所对应索引的嵌入向量(embeddins vector).'''
        # 在Bert模型的tokenizer类BertTokenizer()的词表中,遮罩特殊符[MASK]会被编码为索引103(只要是BertTokenizer()类,无论其from_pretrained哪种
        # 预训练的Bert模型词表,遮罩特殊符[MASK]在词表中的索引都为103; 除非换预训练模型如换成Albert模型,遮罩特殊符[MASK]在词表中的索引才会变, 否则
        # 遮罩特殊符[MASK]在同一类预训练Bert模型的词表下索引不变).
        # 在之后, 遮罩特殊符[MASK]的张量self.mask_embedding的形状要变为和Bert模型嵌入层BertEmbeddings()的输出input_embeddings张量的形状一样,
        # 此时self.mask_embeddings张量的形状要为(batch_size, seq_len, embed_size)->(batch_size, seq_len, 768).

        # 此时,self.mask_embeddings张量的形状为(768,)，将遮罩特殊符[MASK]对应的索引嵌入向量(embeddins vector)的形状降维为(768,),
        # 之后会将遮罩特殊符[MASK]对应的索引嵌入向量(embeddins vector)的形状由(768,)再扩展为(batch_size, seq_len, 768);
        # 此外，此时也对遮罩特殊符[MASK]对应的索引嵌入向量(embeddins vector)self.mask_embeddings使用detach()函数
        # 进行计算图的截断, 防止梯度信息在反向传播时传到self.mask_embeddings这里.
        # self.mask_embeddings = self.embeddings.word_embeddings.weight[ self.tokenizer.mask_token_id() ] # 老写法
        # self.mask_embeddings = self.embeddings( torch.tensor([[ self.tokenizer.mask_token_id ]]).long() ).squeeze().detach()
        mask_token_id_tensor = torch.tensor([[self.mask_token_id]]).long().to(device)
        # 因为遮罩特殊符[MASK]对应的索引张量mask_token_id_tensor已放入当前的device中, 因此此时self.embeddings()词嵌入层
        # 输出的遮罩特殊符[MASK]对应的索引嵌入向量(embeddins vector)self.mask_embeddings也会在同一个device中.
        self.mask_embeddings = self.embeddings(mask_token_id_tensor).squeeze().detach()

        # 输入模型起始处的嵌入张量input embedding由一句sentence中每个character的word embedding、segment embeddings、position embedding三者相加而成。
        # 此时input_embeddings张量的形状为(batch_size, seq_len, embed_size)->(batch_size, seq_len, 768),
        # 应将input_embeddings张量的第一第二维度互换, 将其形状变为(seq_len, batch_size, embed_size)->(seq_len, batch_size, 768)才方便输入进
        # 后方的错误探查网络Detection_Network中的Bi-GRU网络中(双向GRU).
        input_embeddings = self.embeddings(input_ids=input_ids, position_ids=position_ids,
                                           token_type_ids=token_type_ids, inputs_embeds=inputs_embeds)
        # 形状变为(seq_len, batch_size, embed_size)->(seq_len, batch_size, 768).
        input_embeddings = input_embeddings.permute(1, 0, 2)

        # (1)错误探查网络Detection_Network中的双向GRU编码层的输出为(seq_len, batch_size, enc_hid_size * 2),
        # 将其交换维度变形为(batch_size, seq_len, enc_hid_size * 2),再将双向GRU编码层的变形后的输出输入self.detection_network_dense_out层中,
        # 映射为形状(batch_size, seq_len, 2)的张量detection_network_output, 这样方便后面进行判断句子序列中每一个字符是否为拼写错误字符的二分类任务的交叉熵损失值计算.
        # (2)此时soft_masking_coefs张量可被称为：soft-masking系数张量, 其形状为(batch_size, seq_len, 1).
        detection_network_output, soft_masking_coefs = self.Detection_Network(input_embeddings=input_embeddings,
                                                                              attention_mask=attention_mask,
                                                                              device=device)

        # 此时需再将input_embeddings张量的第一第二维度交换, 将其形状再变回(batch_size, seq_len, embed_size)->(batch_size, seq_len, 768),
        # 这样input_embeddings张量才方便输入进self.soft.masking_connection模块中计算soft_masked_embeddings.
        input_embeddings = input_embeddings.permute(1, 0, 2)

        # 遮罩特殊符[MASK]的张量self.mask_embedding的形状要变为和Bert模型嵌入层BertEmbeddings()的输出input_embeddings张量的形状一样,
        # 此时self.mask_embeddings张量的形状要为(batch_size, seq_len, embed_size)->(batch_size, seq_len, 768).
        self.mask_embeddings = self.mask_embeddings.unsqueeze(0).unsqueeze(0).repeat(1, input_embeddings.shape[1],
                                                                                     1).repeat(
            input_embeddings.shape[0], 1, 1)

        # 在错误探查网络detection network输出一个句子中每个位置的字符为错误拼写字符的概率之后，利用此概率作为[MASK] embeddings的权重，
        # 而1减去这个概率作为句子中每个字符character的input embeddings的权重，[MASK] embeddings乘以权重的结果再加上input embeddings乘以权重的结果后
        # 所得到的嵌入结果soft-masked embeddings即为之后的拼写错误纠正网络correction network的输入。
        soft_masked_embeddings = self.Soft_Masking_Connection(input_embeddings=input_embeddings,
                                                              mask_embeddings=self.mask_embeddings,
                                                              soft_masking_coefs=soft_masking_coefs)

        '''拼写错误纠正网络Correction_Network'''
        '''soft_masked_embeddings输入错误纠正网络correction network的Bert模型后的结果经过最后的输出层与Softmax层后，
        即为句子中每个位置的字符经过错误纠正网络correction network计算后预测的正确字符索引结果的概率。'''

        '''注意: 最新版本的transformers.modeling_bert中的BertEncoder()类中forward()方法所需传入的参数中不再有output_attentions这个参数.'''
        encoder_outputs = self.encoder(soft_masked_embeddings,
                                       attention_mask=extended_attention_mask,
                                       head_mask=head_mask,
                                       encoder_hidden_states=encoder_hidden_states,
                                       encoder_attention_mask=encoder_extended_attention_mask, )
        sequence_output = encoder_outputs[0]
        pooled_output = self.pooler(sequence_output)

        # add hidden_states and attentions if they are here
        # outputs为一个包含四个元素的tuple：sequence_output, pooled_output, (hidden_states), (attentions)
        outputs = (sequence_output, pooled_output,) + encoder_outputs[1:]

        # outputs[0]代表Bert模型中最后一个隐藏层的输出(此时Bert模型中的隐藏层有12层,即num_hidden_layers参数为12),
        # 注意此处和循环神经网络的输出形状不同,循环网络隐藏层状态的输出为(seq_len, batch_size, bert_hidden_size)，
        # 此时outputs[0]的张量bert_output_final_hidden_layer的形状为(batch_size, seq_len, bert_hidden_size)—>(batch_size, seq_len, 768).
        bert_output_final_hidden_layer = outputs[0]

        # 注意!: 在soft_masked_embeddings输入拼写错误纠正网络correction network中的Bert模型后,其计算结果输入进最终的输出层与Softmax层之前，
        # 拼写错误纠正网络correction network的结果需通过残差连接residual connection与输入模型一开始的input embeddings相加，
        # 相加的结果才输入最终的输出层与Softmax层中做最终的正确字符预测。
        residual_connection_outputs = bert_output_final_hidden_layer + input_embeddings

        '''self.cls即为拼写错误纠正网络correction network之后的输出层, 其使用的为transformers.modeling_bert模块中预置的模型类
           BertLMPredictionHead(config)，BertLMPredictionHead(config)模型类会将经过残差连接模块residual connection之后
           的输出的维度由768投影到纠错词表的索引空间. (此处输出层self.cls的输出即可被视为Soft_Masked_BERT模型的最终输出)'''
        final_outputs = self.cls(residual_connection_outputs)

        # 此处输出层self.cls的输出final_outputs张量即可被视为Soft_Masked_BERT模型的最终输出.
        # conmpute loss
        loss_ft_fc, g = self.fc_nll_loss(probs[:, :-1, :], target_data[:, 1:], mask_matrix[:, 1:], gamma=self.gamma)
        loss_dic = {"decode_result": self.decode_result,
                    "loss": loss_ft_fc}

        if is_display:
            print(self.decode_result[0]);  # exit()
        # print('loss_dic', loss_dic)
        return loss_dic
        #return final_outputs

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
        self.decode_result = []

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

