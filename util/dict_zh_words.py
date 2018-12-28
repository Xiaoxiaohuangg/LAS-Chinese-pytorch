# -*- coding: utf-8 -*-
# @Author  : huangxiaoxiao
# @File    : dict_zh_words.py
# @Time    : 2018/12/25 15:12
# @Desc    : 生成汉字字典

import os
import pdb
import numpy as np
import json
import pickle

#没用set,set.update时会打乱顺序
word_zh = []
words2index = {k: v for v, k in enumerate(word_zh)}
#四个数据库全部汉字（之前统计过，共6316个）
word_num = np.ones((6316,),dtype=int)

#1.thchs30
file_path = '/data_sdd/audio_data/data_thchs30/data'
files = os.listdir(file_path)
for file in files:
    if file.endswith('.trn'):
        f = open(os.path.join(file_path,file))
        zh = f.readlines()[0][:-1].split(" ")
        for i in range(len(zh)):
            for j in range(len(zh[i])):
                if zh[i][j] in word_zh:
                    word_num[words2index[zh[i][j]]] += 1
                else:
                    word_zh.append(zh[i][j])
                    words2index = {k: v for v, k in enumerate(word_zh)}
        f.close()


#2.ST-CMDS-20170001_1-OS
file_path1 = '/data_sdd/audio_data/ST-CMDS-20170001_1-OS'
files1 = os.listdir(file_path1)
for file in files1:
    if file.endswith('.txt'):
        f = open(os.path.join(file_path1,file))
        zh = f.readline()
        for i in range(len(zh)):
            if zh[i] in word_zh:
                word_num[words2index[zh[i]]] += 1
            else:
                word_zh.append(zh[i])
                words2index = {k: v for v, k in enumerate(word_zh)}
        f.close()


#3.aishell
file_path2 = '/data_sdd/audio_data/data_aishell/transcript/aishell_transcript_v0.8.txt'
f = open(file_path2)
zh = f.readlines()
for i in range(len(zh)):
    zh_split = zh[i][:-1].split(" ")
    for j in range(1,len(zh_split)):
        for k in range(len(zh_split[j])):
            if zh_split[j][k] in word_zh:
                word_num[words2index[zh_split[j][k]]] += 1
            else:
                word_zh.append(zh_split[j][k])
                words2index = {k: v for v, k in enumerate(word_zh)}
f.close()

#4.primewords_md_2018
primewords_txt_path = '/data_sdd/audio_data/primewords_md_2018_set1/set1_transcript.json'
with open(primewords_txt_path,'r') as f:
    primewords_txt = json.load(f)
for f1 in primewords_txt:
    zh = f1['text'].split(" ")
    for i in range(len(zh)):
        for j in range(len(zh[i])):
            if zh[i][j] in word_zh:
                word_num[words2index[zh[i][j]]] += 1
            else:
                word_zh.append(zh[i][j])
                words2index = {k: v for v, k in enumerate(word_zh)}
f.close()

index2words = {k: v for k, v in enumerate(word_zh)}

##对每个汉字出现次数倒序排序
word_num_sorted_index = np.argsort(-word_num)
word_zh_new = []
##只保留在四个数据库中出现过3次以上的汉子,共5027个汉字
for i in range(5027):
    word_zh_new.append(index2words[word_num_sorted_index[i]])

with open('all_dict.pkl', 'wb') as f:
    pickle.dump(word_zh_new, f, pickle.HIGHEST_PROTOCOL)
