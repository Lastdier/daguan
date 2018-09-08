from torch.utils import data
import pandas as pd
import json
import numpy as np
import random
import torch
# import math
from sklearn.model_selection import StratifiedKFold


class DC_data(data.Dataset):

    def __init__(self, max_len, augment=True):
        train_f = '../new_data/train_set.csv'
        self.train = pd.read_csv(train_f)
        f = open('/home/summer/tan/DCDNN/word2index.json', 'r')
        self.word2index = json.load(f)
        f.close()
        self.max_len = max_len
        self.augment = augment
        self.data_len = len(self.train['word_seg'])

        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=1)
        self.folds = []
        for train_index, test_index in skf.split(np.zeros(self.data_len), self.train['class']):
            self.folds.append((train_index, test_index))
        # self.fold_len = math.floor(self.data_len * 1.0 / 5)
        self.fold_index = 0
        # self.val_data_indexs = range(self.fold_index * self.fold_len, (self.fold_index+1) * self.fold_len)
        # self.train_data_indexs = list(set(range(self.data_len)) ^ set(self.val_data_indexs))
        self.current_index_set = self.folds[self.fold_index][0]
        self.trainning = True
        random.seed(19950717)
    
    def change_fold(self, fold_index):      # including change to training data
        self.fold_index = fold_index
        # self.val_data_indexs = range(self.fold_index * self.fold_len, (self.fold_index+1) * self.fold_len)
        # self.train_data_indexs = list(set(range(self.data_len)) ^ set(self.val_data_indexs))
        self.current_index_set = self.folds[self.fold_index][0]
        self.trainning = True
    
    def change2val(self):
        self.current_index_set = self.folds[self.fold_index][1]
        self.trainning = False
    
    def change2train(self):
        self.current_index_set = self.folds[self.fold_index][0]
        self.trainning = True

    def to_index(self, word):
        if self.word2index.get(word) is None:
            return 498681
        else:
            return self.word2index[word]

    def dropout(self,d,p=0.5):
        nnn = []
        for i in d:
            if random.random() > p:
                nnn.append(i)
        return nnn

    def shuffle(self,d):
        return np.random.permutation(d)

    def __getitem__(self, index):
        sen_id = self.current_index_set[index]
        sentence = self.train['word_seg'][sen_id]
        label = self.train['class'][sen_id]
        
        sentence = sentence.split(' ')

        if self.augment and self.trainning:
            temp = random.random()

            if temp < 0.4:
                sentence = self.dropout(sentence)
            elif temp < 0.5:
                sentence = self.shuffle(sentence)

        sentence = [self.word2index[word] for word in sentence if self.word2index.get(word) is not None]

        if len(sentence) > self.max_len: 
            sentence = sentence[:self.max_len]
        else:
            pad = [498681] * (self.max_len - len(sentence))
            sentence += pad
        # sentence = np.array(sentence)
        # label = np.array([label])
        # labellll = [0] * 19
        # labellll[label-1] = 1
        sentence = torch.from_numpy(np.array(sentence)).long()
        label = torch.from_numpy(np.array([label-1]))
        # label = torch.from_numpy(np.array(labellll))
        return sentence, label, sen_id

    def __len__(self):
        return len(self.current_index_set)


if __name__ == '__main__':
    dataset = DC_data(1780)
    dataloader = data.DataLoader(dataset,
                    batch_size = 5,
                    shuffle = True,
                    num_workers = 4,
                    pin_memory = True
                    )
    for i, (sentence, label) in enumerate(dataloader):
        print(sentence)

        
