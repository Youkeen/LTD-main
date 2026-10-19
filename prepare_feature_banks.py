"""Prepare frozen training features once, before running train.py."""
import os
import random

import numpy as np
import torch

from networks.feature_bank import build_bank, path_id
from networks.trainer import create_model
from options.bank_options import BankOptions


def prepare_feature_banks(opt):
    if opt.method != 'reference_flow' or opt.data_label != 'train':
        raise ValueError('Feature bank preparation requires reference_flow and the training split.')
    if opt.fake_bank_output and path_id(opt.real_bank_output) == path_id(opt.fake_bank_output):
        raise ValueError('Real and fake banks must have different output paths.')
    from data.datasets import RealFakeDataset
    random.seed(opt.bank_seed)
    np.random.seed(opt.bank_seed)
    torch.manual_seed(opt.bank_seed)
    device = torch.device(f'cuda:{opt.gpu_ids[0]}' if opt.gpu_ids else 'cpu')
    dataset = RealFakeDataset(opt)
    model = create_model(opt).to(device).eval()
    for label, output in ((0, opt.real_bank_output), (1, opt.fake_bank_output)):
        if not output:
            continue
        bank = build_bank(model, dataset, opt, labels=(label,))
        os.makedirs(os.path.dirname(os.path.abspath(output)), exist_ok=True)
        bank.save(output)
        print(f'Saved {len(bank.paths)} {"real" if label == 0 else "fake"} features to {output}')


if __name__ == '__main__':
    prepare_feature_banks(BankOptions().parse(print_options=False))
