# -*- encoding: utf-8 -*-
# @Author: Yunshi Lan
# @Time: 2023/01/27 10:20
# @File: run_gectoolkit.py


import argparse
import sys
import os

from gectoolkit.quick_start import run_toolkit

sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), ".")))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', '-m', type=str, default='RNN', help='name of models')
    parser.add_argument('--dataset', '-d', type=str, default='csc', help='name of datasets')
    parser.add_argument('--language', '-l', type=str, default='Chinese', help='language of dataset')

    args, _ = parser.parse_known_args()
    config_dict = {}

    run_toolkit(args.model, args.dataset, args.language, config_dict)

'''
python run_gectoolkit.py    \
--model RNN \
--dataset csc
'''