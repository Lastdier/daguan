from .BasicModule import BasicModule
import torch as t
import numpy as np
from torch import nn
import json


def kmax_pooling(x, dim, k):
    index = x.topk(k, dim = dim)[1].sort(dim = dim)[0]
    return x.gather(dim, index)

class RCNN(BasicModule): 
    def __init__(self, opt ):
        super(RCNN, self).__init__()
        self.model_name = 'RCNN'
        self.opt=opt

        kernel_size = opt.kernel_size
        self.encoder = nn.Embedding(opt.vocab_size+1,opt.embedding_dim, padding_idx=498681)

        self.content_lstm =nn.LSTM(  input_size = opt.embedding_dim,\
                            hidden_size = opt.hidden_size,
                            num_layers = opt.num_layers,
                            bias = True,
                            batch_first = False,
                            # dropout = 0.5,
                            bidirectional = True
                            )

        self.content_conv = nn.Sequential(
            nn.Conv1d(in_channels = opt.hidden_size*2 + opt.embedding_dim,
                      out_channels = opt.content_dim,
                      kernel_size =  kernel_size),
            nn.BatchNorm1d(opt.content_dim),
            nn.ReLU(inplace=True),

            nn.Conv1d(in_channels = opt.content_dim,
                      out_channels = opt.content_dim,
                      kernel_size =  kernel_size),
            nn.BatchNorm1d(opt.content_dim),
            nn.ReLU(inplace=True),

            
            # nn.MaxPool1d(kernel_size = (opt.content_seq_len - opt.kernel_size + 1))
        )
        # self.dropout = nn.Dropout()
        self.fc = nn.Sequential(
            nn.Linear(opt.kmax_pooling*(opt.content_dim),opt.linear_hidden_size),
            nn.BatchNorm1d(opt.linear_hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(opt.linear_hidden_size,opt.num_classes)
        )
        # self.fc = nn.Linear(3 * (opt.title_dim+opt.content_dim), opt.num_classes)
        if opt.embedding_path:
            self.encoder.weight.data.copy_(self.load_embedding(MyEmbeddings(opt.embedding_path)))

    def forward(self, content):
        content = self.encoder(content)
        
        if self.opt.static:
            content.detach()


        content_out = self.content_lstm(content.permute(1,0,2))[0].permute(1,2,0)
        content_em = (content).permute(0,2,1)
        content_out = t.cat((content_out,content_em),dim=1)

        content_conv_out = kmax_pooling(self.content_conv(content_out),2,self.opt.kmax_pooling)

        # conv_out = t.cat((content_conv_out),dim=1)
        reshaped = content_conv_out.view(content_conv_out.size(0), -1)
        logits = self.fc((reshaped))
        return logits
    
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

    # def get_optimizer(self):  
    #    return  t.optim.Adam([
    #             {'params': self.title_conv.parameters()},
    #             {'params': self.content_conv.parameters()},
    #             {'params': self.fc.parameters()},
    #             {'params': self.encoder.parameters(), 'lr': 5e-4}
    #         ], lr=self.opt.lr)
    # # end method forward


 
if __name__ == '__main__':
    from ..config import opt
    m = CNNText(opt)
    title = t.autograd.Variable(t.arange(0,500).view(10,50)).long()
    content = t.autograd.Variable(t.arange(0,2500).view(10,250)).long()
    o = m(title,content)
    print(o.size())
