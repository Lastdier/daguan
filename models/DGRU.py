from .BasicModule import BasicModule
import torch as t
import numpy as np
from torch import nn
import json


class DGRU(BasicModule):
    def __init__(self, opt ):
        super(DGRU, self).__init__()
        self.model_name = 'DGRU'
        self.opt=opt

        kernel_size = opt.kernel_size
        self.encoder = nn.Embedding(opt.vocab_size+1,opt.embedding_dim, padding_idx=opt.vocab_size)
        self.window_size = 15
        self.content_gru =nn.GRU(input_size = opt.embedding_dim,\
                            hidden_size = opt.hidden_size,
                            num_layers = opt.num_layers,
                            bias = True,
                            batch_first = False,
                            # dropout = 0.5,
                            bidirectional = False
                            )
        
        self.bn1 = nn.BatchNorm1d(opt.hidden_size)
        self.ln1 = nn.Linear(opt.hidden_size, opt.linear_hidden_size)
        self.mp = nn.MaxPool1d(1780)

        self.fc = nn.Sequential(
            nn.Linear(opt.linear_hidden_size,opt.num_classes),
            nn.ReLU(inplace=True)
        )
        # self.fc = nn.Linear(3 * (opt.title_dim+opt.content_dim), opt.num_classes)
        if opt.embedding_path:
            self.encoder.weight.data.copy_(self.load_embedding(MyEmbeddings(opt.embedding_path)))

    def forward(self, content):
        padd = t.full((content.size(0), self.window_size-1), self.opt.vocab_size).long().cuda()
        length_of_sent = content.size(1)
        content = t.cat((padd, content), 1)
        content = self.encoder(content)
        
        if self.opt.static:
            content.detach()


        content = content.unfold(1, self.window_size, 1)
        content_hidden = []
        for i in range(length_of_sent):
            content_hidden.append(self.content_gru(content[:,i].permute(2,0,1))[0][self.window_size-1].unsqueeze(1))
        
        content_out = t.cat(content_hidden, 1)  # zuhe

        logits = self.bn1(content_out.permute(0,2,1).contiguous())
        logits = self.ln1(logits.permute(0,2,1))
        logits = self.mp(logits.permute(0,2,1)).squeeze(2)
        logits = self.fc(logits)
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
