from all_datasets import load_dataset
import numpy as np
import pickle
import scipy.io.wavfile as wav
from six.moves import cPickle
import python_speech_features as features

# 1.Load preprocessed thchs30 Dataset
thchs30_pickle_path = '/share_sdb/hxx/ASR_ZH/LAS-Chinese-pytorch/data/thchs30.pkl'
X_train_thchs30, y_train_thchs30, X_val_thchs30, y_val_thchs30, X_test_thchs30, y_test_thchs30 = load_dataset(
    thchs30_pickle_path)

# 2.Load preprocessed primewords Dataset
primewords_pickle_path = '/share_sdb/hxx/ASR_ZH/LAS-Chinese-pytorch/data/primewords.pkl'
X_train_primewords, y_train_primewords, X_val_primewords, y_val_primewords, X_test_primewords, y_test_primewords = load_dataset(
    primewords_pickle_path)

# 3.Load preprocessed ST Dataset
ST_pickle_path = '/share_sdb/hxx/ASR_ZH/LAS-Chinese-pytorch/data/ST.pkl'
X_train_ST, y_train_ST, X_val_ST, y_val_ST, X_test_ST, y_test_ST = load_dataset(ST_pickle_path)

# 4.Load preprocessed aishell Dataset
aishell_pickle_path = '/share_sdb/hxx/ASR_ZH/LAS-Chinese-pytorch/data/aishell.pkl'
X_train_aishell, y_train_aishell, X_val_aishell, y_val_aishell, X_test_aishell, y_test_aishell = load_dataset(
    aishell_pickle_path)

X_train = X_train_thchs30; y_train = y_train_thchs30;
X_val = X_val_thchs30; y_val = y_val_thchs30;
X_test = X_test_thchs30; y_test = y_test_thchs30

X_train.extend(X_train_primewords); y_train.extend(y_train_primewords);
X_val.extend(X_val_primewords); y_val.extend(y_val_primewords);
X_test.extend(X_test_primewords); y_test.extend(y_test_primewords)

X_train.extend(X_train_ST); y_train.extend(y_train_ST);
X_val.extend(X_val_ST); y_val.extend(y_val_ST);
X_test.extend(X_test_ST); y_test.extend(y_test_ST)

X_train.extend(X_train_aishell); y_train.extend(y_train_aishell);
X_val.extend(X_val_aishell); y_val.extend(y_val_aishell);
X_test.extend(X_test_aishell); y_test.extend(y_test_aishell)


#sort as txt_len
# y_train_len = [] 
# for i in range(len(y_train)):
#     y_train_len.append(len(y_train[i]))

# y_train_len = np.array(y_train_len)    
# y_train_sorted_index = np.argsort(-y_train_len)

# X_train_sorted = []; y_train_sorted = []
# for i in range(len(y_train)):
#     X_train_sorted.append(X_train[y_train_sorted_index[i]])
#     y_train_sorted.append(y_train[y_train_sorted_index[i]])

#sort as voice_len
X_train_len = [] 

for i in range(len(X_train)):
    (rate, sample) = wav.read(X_train[i])
    mfcc = features.mfcc(sample, rate, winlen=0.025, winstep=0.01, numcep=13, nfilt=26,
                         preemph=0.97, appendEnergy=True)
    X_train_len.append(mfcc.shape[0])
    print('file No.', i, end='\r', flush=True)
X_train_len = np.array(X_train_len)
X_train_len_sorted = sorted(X_train_len,reverse = True)
X_train_sorted_index = np.argsort(-X_train_len)

X_train_sorted = []; y_train_sorted = []
for i in range(len(X_train)):
    X_train_sorted.append(X_train[X_train_sorted_index[i]])
    y_train_sorted.append(y_train[X_train_sorted_index[i]])

with open('./data/train_data' +  '.pkl', 'wb') as cPickle_file:
    cPickle.dump(
        [X_train_sorted, y_train_sorted, X_train_len_sorted],
        cPickle_file,
        protocol=cPickle.HIGHEST_PROTOCOL)
with open('./data/test_data' +  '.pkl', 'wb') as cPickle_file:
    cPickle.dump(
        [X_test, y_test],
        cPickle_file,
        protocol=cPickle.HIGHEST_PROTOCOL)
    
with open('./data/val_data' +  '.pkl', 'wb') as cPickle_file:
    cPickle.dump(
        [X_val, y_val],
        cPickle_file,
        protocol=cPickle.HIGHEST_PROTOCOL)
