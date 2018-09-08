from .BasicModule import BasicModule
import torch as t
import torch
import numpy as np
from torch import nn
from collections import OrderedDict
import json

class Inception(nn.Module):
    def __init__(self,cin,co,relu=True,norm=True):
        super(Inception, self).__init__()
        assert(co%4==0)
        cos=[int(co/4)]*4
        self.activa=nn.Sequential()
        if norm:self.activa.add_module('norm',nn.BatchNorm1d(co))
        if relu:self.activa.add_module('relu',nn.ReLU(True))

        self.branch1 =nn.Sequential(OrderedDict([
            ('conv1', nn.Conv1d(cin,cos[0], 1,stride=1)),
            ])) 
        self.branch2 =nn.Sequential(OrderedDict([
            ('conv1', nn.Conv1d(cin,cos[1], 1)),
            ('norm1', nn.BatchNorm1d(cos[1])),
            ('relu1', nn.ReLU(inplace=True)),
            ('conv3', nn.Conv1d(cos[1],cos[1], 3,stride=1,padding=1)),
            ]))
        self.branch3 =nn.Sequential(OrderedDict([
            ('conv1', nn.Conv1d(cin,cos[2], 3,padding=1)),
            ('norm1', nn.BatchNorm1d(cos[2])),
            ('relu1', nn.ReLU(inplace=True)),
            ('conv3', nn.Conv1d(cos[2],cos[2], 5,stride=1,padding=2)),
            ]))
        self.branch4 =nn.Sequential(OrderedDict([
            #('pool',nn.MaxPool1d(2)),
            ('conv3', nn.Conv1d(cin,cos[3], 3,stride=1,padding=1)),
            ]))
    def forward(self,x):
        branch1=self.branch1(x)
        branch2=self.branch2(x)
        branch3=self.branch3(x)
        branch4=self.branch4(x)
        result=self.activa(torch.cat((branch1,branch2,branch3,branch4),1))
        return result
class CNNText_inception(BasicModule):
    def __init__(self, opt ):
        super(CNNText_inception, self).__init__()
        incept_dim=opt.inception_dim
        self.model_name = 'CNNText_inception'
        self.opt=opt
        self.encoder = nn.Embedding(opt.vocab_size+1,opt.embedding_dim, padding_idx=498681)
        if opt.embedding_path:
            self.encoder.from_pretrained(self.load_embedding(MyEmbeddings(opt.embedding_path)))

        self.content_conv=nn.Sequential(
            Inception(opt.embedding_dim,incept_dim),#(batch_size,64,opt.content_seq_len)->(batch_size,64,(opt.content_seq_len)/2)
            #Inception(incept_dim,incept_dim),#(batch_size,64,opt.content_seq_len/2)->(batch_size,32,(opt.content_seq_len)/4)
            Inception(incept_dim,incept_dim),
            nn.MaxPool1d(opt.content_seq_len)
        )
        self.fc = nn.Sequential(
            nn.Linear(incept_dim,opt.linear_hidden_size),
            nn.BatchNorm1d(opt.linear_hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(opt.linear_hidden_size,opt.num_classes)
        )

    def forward(self,content):
        content=self.encoder(content)
        if self.opt.static:
            content=content.detach(0)
        content_out=self.content_conv(content.permute(0,2,1))
        out=content_out.view(content_out.size(0), -1)
        out=self.fc(out)
        return out
    
    def load_embedding(self, myembedding):
        f = open('word2index.json', 'r')
        word2index = json.load(f)
        f.close()
        
        weight = np.random.uniform(-0.1,0.1,[498681, len(myembedding)])
        weight = np.concatenate([weight, np.zeros((1,len(myembedding)))], 0)
        for line in myembedding:
            pair = line.split(' ')
            if word2index.get(pair[0]) is not None:
                weight[word2index[pair[0]]] = [float(i) for i in pair[1:]]

        weight = t.tensor(weight, dtype=t.float32)
        print('pretrain wordvec loaded!')
        return weight

class MyEmbeddings(object):
    def __init__(self, path):
        self.path = path

    def __iter__(self):
        for line in open(self.path, 'r'):
            yield line.strip()

    def __len__(self):
        length = 0
        with open(self.path, 'r') as f:
            length = f.readline().split()[1]
        return int(length)
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        