# -*- encoding: utf-8 -*-
# @Author: Yunshi Lan
# @Time: 2023/01/27 10:55
# @File: enum_type.py


class DatasetName:
    """dataset name
    """
    csc = "csc"


class DatasetType:
    """dataset type
    """
    Train = "train"
    Test = "test"
    Valid = "valid"

class DatasetLanguage:
    """dataset language
    """
    en="english"
    zh="chinese"


class SpecialTokens:
    """special tokens
    """
    PAD_TOKEN = "<-PAD->" # padding token
    UNK_TOKEN = "<-UNK->" # unknown token
    SOS_TOKEN = "<SOS>" # start token
    EOS_TOKEN = "<EOS>" # end token
    NON_TOKEN = "<NON>" # non-terminal token
    BRG_TOKEN = "<BRG>" # equation connecting token
    OPT_TOKEN = "<OPT>" # operator mask token
    CLS_TOKEN = "<-CLS->"
    SEP_TOKEN = "<-SEP->"
    MASK_TOKEN = "<-MASK->"
    NUM_TOKEN = '<-NUM->'
    NOT_CHINESE_TOKEN = '<-NOT_CHINESE->'


class SupervisingMode:
    """supervising mode"""
    fully_supervised="fully_supervised"


    