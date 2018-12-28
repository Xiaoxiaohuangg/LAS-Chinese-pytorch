# -*- coding: utf-8 -*-
# @Author  : huangxiaoxiao
# @File    : train_all.py
# @Time    : 2018/12/25 16:26
# @Desc    : train all datasets

import os
import yaml
from util.all_datasets import create_dataloader
from util.functions import  batch_iterator
from model.las_model import Listener, Speller
import numpy as np
from torch.autograd import Variable
import torch
import time
import pdb
import matplotlib.pyplot as plt
from six.moves import cPickle
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
import python_speech_features as features
import scipy.io.wavfile as wav
torch.cuda.set_device(1)



# Load example config file for experiment
config_path = 'config/las_all_datasets_config.yaml'
conf = yaml.load(open(config_path, 'r'))

# Parameters loading
use_pretrained = conf['training_parameter']['use_pretrained']
num_epochs = conf['training_parameter']['num_epochs']
training_msg = 'epoch_{:2d}_step_{:3d}_TrLoss_{:.4f}_TrWER_{:.2f}'
epoch_end_msg = 'epoch_{:2d}_TrLoss_{:.4f}_TrWER_{:.2f}_ValLoss_{:.4f}_ValWER_{:.2f}_time_{:.2f}'
listener_model_path = conf['meta_variable']['checkpoint_dir'] + conf['meta_variable']['experiment_name'] + '.listener'
speller_model_path = conf['meta_variable']['checkpoint_dir'] + conf['meta_variable']['experiment_name'] + '.speller'
verbose_step = conf['training_parameter']['verbose_step']
tf_rate_upperbound = conf['training_parameter']['tf_rate_upperbound']
tf_rate_lowerbound = conf['training_parameter']['tf_rate_lowerbound']

def load_dataset_train(data_path,**kwargs):
    with open(data_path, 'rb') as cPickle_file:
        [X_train, y_train, X_train_len] = cPickle.load(cPickle_file)
    for data in [X_train, y_train, X_train_len]:
        assert len(data) > 0
    return X_train, y_train, X_train_len

def load_dataset(data_path,**kwargs):
    with open(data_path, 'rb') as cPickle_file:
        [X_, y_] = cPickle.load(cPickle_file)
    for data in [X_, y_]:
        assert len(data) > 0
    return X_, y_

#Load preprocessed Dataset
train_pickle_path = '/share_sdb/hxx/ASR_ZH/LAS-Chinese-pytorch/data/train_data.pkl'
X_train, y_train, X_train_len_sorted = load_dataset_train(train_pickle_path)
X_train.reverse(); y_train.reverse(); X_train_len_sorted.reverse()

test_pickle_path = '/share_sdb/hxx/ASR_ZH/LAS-Chinese-pytorch/data/test_data.pkl'
X_test, y_test = load_dataset(test_pickle_path)

val_pickle_path = '/share_sdb/hxx/ASR_ZH/LAS-Chinese-pytorch/data/val_data.pkl'
X_val, y_val = load_dataset(val_pickle_path)



train_set = create_dataloader(X_train, y_train, **conf['model_parameter'], **conf['training_parameter'], shuffle=False)
valid_set = create_dataloader(X_val, y_val, **conf['model_parameter'], **conf['training_parameter'], shuffle=False)
test_set = create_dataloader(X_test, y_test, **conf['model_parameter'], **conf['training_parameter'], shuffle=False)


if not use_pretrained:
    print('-----init model---------')
    listener = Listener(**conf['model_parameter'])
    speller = Speller(**conf['model_parameter'])
else:
    print('-----restore model------')
    listener = torch.load(conf['training_parameter']['pretrained_listener_path'])
    speller = torch.load(conf['training_parameter']['pretrained_speller_path'])

optimizer = torch.optim.Adam([{'params': listener.parameters()}, {'params': speller.parameters()}],
                             lr=conf['training_parameter']['learning_rate'])

best_ler = 1.0
traing_log = open(conf['meta_variable']['training_log_dir'] + conf['meta_variable']['experiment_name'] + '.log', 'w')
print('--------start training!---------')
for epoch in range(num_epochs):
    epoch_head = time.time()
    tr_loss = 0.0
    tr_ler = []
    val_loss = 0.0
    val_ler = []
    len_dataloader_train = len(train_set)
    batch_data_iter_train = iter(train_set)

    len_dataloader_val = len(valid_set)
    batch_data_iter_val = iter(valid_set)

    # Teacher forcing rate linearly decay
    tf_rate = tf_rate_upperbound - (tf_rate_upperbound - tf_rate_lowerbound) * (epoch / num_epochs)
    # Training
    i = 0
    while i < len_dataloader_train:
        batch_data_source = batch_data_iter_train.next()
        batch_data, batch_label = batch_data_source
        #pdb.set_trace()
        batch_loss, batch_ler = batch_iterator(batch_data, batch_label, listener, speller, optimizer,
                                               tf_rate, is_training=True, **conf['model_parameter'])
        tr_loss += batch_loss
        tr_ler.extend(batch_ler)
        if (i + 1) % verbose_step == 0:
            print(
                training_msg.format(epoch + 1, i + 1, tr_loss / (i + 1), sum(tr_ler) / len(tr_ler)),
                end='\r', flush=True)
        i += 1
    training_time = float(time.time() - epoch_head)

    # Validation
    j = 0
    while j < len_dataloader_val:
        batch_data_source = batch_data_iter_val.next()
        batch_data, batch_label = batch_data_source
        batch_loss, batch_ler = batch_iterator(batch_data, batch_label, listener, speller, optimizer,
                                               tf_rate, is_training=False, **conf['model_parameter'])
        val_loss += batch_loss
        val_ler.extend(batch_ler)
        j += 1

    # Logger
    print(epoch_end_msg.format(epoch + 1, tr_loss / (i + 1), sum(tr_ler) / len(tr_ler),
                               val_loss / len_dataloader_val, sum(val_ler) / len(val_ler), training_time), flush=True)
    print(epoch_end_msg.format(epoch + 1, tr_loss / (i + 1), sum(tr_ler) / len(tr_ler),
                               val_loss / len_dataloader_val, sum(val_ler) / len(val_ler), training_time), flush=True,
          file=traing_log)

    # Checkpoint
    if best_ler >= sum(val_ler) / len(val_ler):
        best_ler = sum(val_ler) / len(val_ler)
        torch.save(listener, listener_model_path)
        torch.save(speller, speller_model_path)




