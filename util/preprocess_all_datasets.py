# -*- coding: utf-8 -*-
# @Author  : huangxiaoxiao
# @File    : preprocess_all_datasets.py
# @Time    : 2018/12/25 15:26
# @Desc    : preprocess_all_datasets,save as pkl. wav path and txts


import os
import sys
import wave
import pdb
import json
import timeit;

program_start_time = timeit.default_timer()
import random;
import pickle

random.seed(int(timeit.default_timer()))
from six.moves import cPickle
import numpy as np
import scipy.io.wavfile as wav

import python_speech_features as features
from pathlib import Path

RIFF_wav_postfix =  '.wav'

##load dict file //dict_zh_words.py
dict_path = '/share_sdb/hxx/ASR_ZH/LAS-Chinese-pytorch/data/all_dict.pkl'
with open(dict_path, 'rb') as f:
    all_dict = pickle.load(f)
word2index = {k: v for v, k in enumerate(all_dict)}


def preprocess_dataset(source_path):
    """Preprocess data, ignoring compressed files and files starting with 'SA'"""
    i = 0
    X = []
    Y = []
    data_path = '/data_sdd/audio_data/data_thchs30/data'
    fileList = os.listdir(source_path)
    for fname in fileList:
        if not fname.endswith('.trn'):
            continue
        # trn
        trn_fname = os.path.join(data_path, fname)
        # wav
        wav_fname = os.path.join(source_path, fname[0:-8] + RIFF_wav_postfix)

        # trn
        fr = open(trn_fname)
        zh_words = fr.readlines()[0][:-1].split(" ")


        X.append(wav_fname)

        y_ = []
        for word in zh_words:
            for j in range(len(word)):
                word_num = word2index[word[j]] if word[j] in word2index else -1
                y_.append(word_num)

        fr.close()
        y_ = np.array(y_)
        if -1 in y_:
            print('WARNING: -1 detected in TARGET')
            #print(y_)
            continue

        Y.append(y_.astype('int32'))

        i += 1
        print('file No.', i, end='\r', flush=True)

    print('Done')
    return X, Y

# 1. parse thchs30
print( '------preprocessing of thchs30------ ')
X_train_thchs30 = []; X_val_thchs30 = []; X_test_thchs30 = []
y_train_thchs30 = []; y_val_thchs30 = []; y_test_thchs30 = []

thchs30_dirs_train = '/data_sdd/audio_data/data_thchs30/train/'
thchs30_dirs_val = '/data_sdd/audio_data/data_thchs30/dev/'
thchs30_dirs_test = '/data_sdd/audio_data/data_thchs30/test/'

X_train_thchs30, y_train_thchs30 = preprocess_dataset(thchs30_dirs_train)
X_val_thchs30, y_val_thchs30 = preprocess_dataset(thchs30_dirs_val)
X_test_thchs30, y_test_thchs30 = preprocess_dataset(thchs30_dirs_test)

target_path = '/share_sdb/hxx/ASR_ZH/LAS-Chinese-pytorch/data/thchs30'
with open(target_path + '.pkl', 'wb') as cPickle_file:
    cPickle.dump(
        [X_train_thchs30, y_train_thchs30, X_val_thchs30, y_val_thchs30, X_test_thchs30, y_test_thchs30],
        cPickle_file,
        protocol=cPickle.HIGHEST_PROTOCOL)

# 2. parse aishell
print('------preprocessing of aishell------ ')
X_train_aishell = []; X_val_aishell = []; X_test_aishell = []
y_train_aishell = []; y_val_aishell = []; y_test_aishell = []


def choose_aishell_path(f_name):
    sub_dir = f_name[6: -5]
    num = int(sub_dir[1:])
    mode = 'train'
    if num > 723 and num <= 763:
        mode = 'dev'
    elif num > 763:
        mode = 'test'
    return '%s/%s/%s.wav' % (mode, sub_dir, f_name), mode


aishell_txt_path = '/data_sdd/audio_data/data_aishell/transcript/aishell_transcript_v0.8.txt'
audio_path_root = '/data_sdd/audio_data/data_aishell/wav/'
i = 0
with open(aishell_txt_path) as f:
    for line in f.readlines():

        # pdb.set_trace()
        l = line.strip()
        txt = l[l.index(' ') + 1:].split(" ")
        f_name = l[:l.index(' ')]
        path, mode = choose_aishell_path(f_name)
        wav_path = audio_path_root + path

        y_ = []
        for j in range(len(txt)):
            for k in range(len(txt[j])):
                word_num = word2index[txt[j][k]] if txt[j][k] in word2index else -1
                y_.append(word_num)

        y_ = np.array(y_)
        if -1 in y_:
            print('WARNING: -1 detected in TARGET')
            #print(y_)
            continue

        if mode == 'train':
            X_train_aishell.append(wav_path)
            y_train_aishell.append(y_.astype('int32'))
        elif mode == 'dev':
            X_val_aishell.append(wav_path)
            y_val_aishell.append(y_.astype('int32'))
        else:
            X_test_aishell.append(wav_path)
            y_test_aishell.append(y_.astype('int32'))

        i += 1
        print('file No.', i, end='\r', flush=True)

target_path = '/share_sdb/hxx/ASR_ZH/LAS-Chinese-pytorch/data/aishell'
with open(target_path  +'.pkl', 'wb') as cPickle_file:
    cPickle.dump(
        [X_train_aishell, y_train_aishell, X_val_aishell, y_val_aishell, X_test_aishell, y_test_aishell ],
        cPickle_file,
        protocol=cPickle.HIGHEST_PROTOCOL)

# 3. parse primewords_md_2018_set1
print('------preprocessing of primewords_md_2018_set1------ ')
X_train_primewords = []; X_val_primewords = []; X_test_primewords = []
y_train_primewords = []; y_val_primewords = []; y_test_primewords = []


#train/val/test
if not os.path.exists('../data/primewords/'):
    os.mkdir('../data/primewords/')
    
train_file_txt = open('../data/primewords/train_primewords.txt', 'w')
val_file_txt = open('../data/primewords/val_primewords.txt', 'w')
test_file_txt = open('../data/primewords/test_primewords.txt', 'w')

primewords_txt_path = '/data_sdd/audio_data/primewords_md_2018_set1/set1_transcript.json'
with open(primewords_txt_path, 'r') as f:
    primewords_txt = json.load(f)

label_datas = list()
audio_path_root = '/data_sdd/audio_data/primewords_md_2018_set1/audio_files/'
i = 0
for f in primewords_txt:

    wav_path = os.path.join(audio_path_root, f['file'][0], f['file'][:2], f['file'])
    y_ = []
    txt = f['text'].split(" ")
    for j in range(len(txt)):
        for k in range(len(txt[j])):
            word_num = word2index[txt[j][k]] if txt[j][k] in word2index else -1
            y_.append(word_num)
    y_ = np.array(y_)
    if -1 in y_:
        print('WARNING: -1 detected in TARGET')
        #print(y_)
        continue


    if i < 38177:
        X_train_primewords.append(wav_path)
        y_train_primewords.append(y_.astype('int32'))
        train_file_txt.write(wav_path + '\n')
    elif i >= 38177 and i < 40722:
        X_val_primewords.append(wav_path)
        y_val_primewords.append(y_.astype('int32'))
        val_file_txt.write(wav_path + '\n')
    else:
        X_test_primewords.append(wav_path)
        y_test_primewords.append(y_.astype('int32'))
        test_file_txt.write(wav_path + '\n')
    i += 1

    print('file No.', i, end='\r', flush=True)
train_file_txt.close()
val_file_txt.close()
test_file_txt.close()

target_path = '/share_sdb/hxx/ASR_ZH/LAS-Chinese-pytorch/data/primewords'
with open(target_path + '.pkl', 'wb') as cPickle_file:
    cPickle.dump(
        [X_train_primewords, y_train_primewords, X_val_primewords, y_val_primewords, X_test_primewords, y_test_primewords],
        cPickle_file,
        protocol=cPickle.HIGHEST_PROTOCOL)


# 4. parse ST-CMDS-20170001_1-OS
print('------preprocessing of ST-CMDS-20170001_1-OS------ ')
st_cmds_path = Path('/data_sdd/audio_data/ST-CMDS-20170001_1-OS/')

if not os.path.exists('../data/ST-CMDS-20170001_1-OS/'):
    os.mkdir('../data/ST-CMDS-20170001_1-OS/')

train_file_txt = open('../data/ST-CMDS-20170001_1-OS/train_ST.txt', 'w')
val_file_txt = open('../data/ST-CMDS-20170001_1-OS/val_ST.txt', 'w')
test_file_txt = open('../data/ST-CMDS-20170001_1-OS/test_ST.txt', 'w')

X_train_ST = []; X_val_ST = []; X_test_ST = []
y_train_ST = []; y_val_ST = []; y_test_ST = []
i = 0
for f in st_cmds_path.glob('*.txt'):
    with open(f, 'r') as fin:

        y_ = []
        line = fin.readlines()[0].strip()
        for j in range(len(line)):
            word_num = word2index[line[j]] if line[j] in word2index else -1
            y_.append(word_num)
        y_ = np.array(y_)
        if -1 in y_:
            print('WARNING: -1 detected in TARGET')
            #print(y_)
            continue

        f_name = str(f)
        f_path = '%s.wav' % f_name[:f_name.rindex('.')]

        if i < 76950:
            X_train_ST.append(f_path)
            y_train_ST.append(y_.astype('int32'))
            train_file_txt.write(f_path + '\n')
        elif i >= 76950 and i < 82080:
            X_val_ST.append(f_path)
            y_val_ST.append(y_.astype('int32'))
            val_file_txt.write(f_path + '\n')
        else:
            X_test_ST.append(f_path)
            y_test_ST.append(y_.astype('int32'))
            test_file_txt.write(f_path + '\n')
        i += 1
        print('file No.', i, end='\r', flush=True)

train_file_txt.close()
val_file_txt.close()
test_file_txt.close()

target_path = '/share_sdb/hxx/ASR_ZH/LAS-Chinese-pytorch/data/ST'
with open(target_path +  '.pkl', 'wb') as cPickle_file:
    cPickle.dump(
        [X_train_ST, y_train_ST, X_val_ST, y_val_ST, X_test_ST, y_test_ST],
        cPickle_file,
        protocol=cPickle.HIGHEST_PROTOCOL)



