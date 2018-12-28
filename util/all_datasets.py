# -*- coding: utf-8 -*-
# @Author  : huangxiaoxiao
# @File    : all_datasets.py
# @Time    : 2018/12/25 15:26
# @Desc    : Dataset:AllDataset


from six.moves import cPickle
import numpy as np
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
import scipy.io.wavfile as wav
import python_speech_features as features
import pdb


def load_dataset(data_path,**kwargs):
    with open(data_path, 'rb') as cPickle_file:
        [X_train, y_train, X_val, y_val, X_test, y_test] = cPickle.load(cPickle_file)
    for data in [X_train, y_train, X_val, y_val, X_test, y_test]:
        assert len(data) > 0
    return X_train, y_train, X_val, y_val, X_test, y_test



def create_mfcc(filename):
    """Perform standard preprocessing, as described by Alex Graves (2012)
    http://www.cs.toronto.edu/~graves/preprint.pdf
    Output consists of 12 MFCC and 1 energy, as well as the first derivative of these.
    [1 energy, 12 MFCC, 1 diff(energy), 12 diff(MFCC)
    """

    (rate, sample) = wav.read(filename)
    mfcc = features.mfcc(sample, rate, winlen=0.025, winstep=0.01, numcep=13, nfilt=26,
                         preemph=0.97, appendEnergy=True)
    d_mfcc = features.delta(mfcc, 2)
    a_mfcc = features.delta(d_mfcc, 2)

    out = np.concatenate([mfcc, d_mfcc, a_mfcc], axis=1)

    return out, out.shape[0]

def ZeroPadding(x,pad_len):
    features = x[0].shape[-1]
    new_x = np.zeros((len(x),pad_len,features))
    for idx ,ins in enumerate(x):
        new_x[idx,:len(ins),:] = ins
    return new_x


def OneHotEncode(Y,max_len,max_idx=5027):
    new_y = np.zeros((len(Y),max_len,max_idx+2))
    for idx ,label_seq in enumerate(Y):
        last_value = -1
        cnt = 0
        for label in label_seq:
            if last_value != label:
                new_y[idx,cnt,label+2] = 1.0
                cnt += 1
                last_value = label
        new_y[idx,cnt,1] = 1.0 # <eos>
    return new_y

def normalize(X, mean_val, std_val):
	for i in range(len(X)):
		X[i] = (X[i] - mean_val)/std_val
	return X


def set_type(X, type):
	for i in range(len(X)):
		X[i] = X[i].astype(type)
	return X

##Dataset
class AllDataset(Dataset):
    def __init__(self, X, Y, max_timestep, max_label_len, batch_size, data_type, transform = None,target_transform = None):

        self.transform = transform
        self.target_transform = target_transform
        self.X = X
        self.Y = Y
        self.batch_size = batch_size
        self.max_timestep = max_timestep
        self.max_label_len = max_label_len
        self.data_type = data_type


    def __getitem__(self, index):
        if self.target_transform is not None :
            Y_ = []
            Y_.append(self.Y[index])
            label_txt = OneHotEncode(Y_,self.max_label_len)
        if self.transform is not None :
            X_ = []
            img_wav_path = self.X[index]
            img_features, total_frames = create_mfcc(img_wav_path)
            img_features = set_type(img_features, self.data_type)
            X_.append(img_features)
            img_features = ZeroPadding(X_,self.max_timestep)
        return img_features[0],label_txt[0]

    def __len__(self):
        return len(self.X)

def create_dataloader(X, Y, max_timestep, max_label_len, batch_size, data_type, transform, target_transform, shuffle,  **kwargs):

    return DataLoader(AllDataset(X,Y,max_timestep,max_label_len,batch_size,data_type,transform, target_transform),
                          batch_size=batch_size,shuffle=shuffle)


