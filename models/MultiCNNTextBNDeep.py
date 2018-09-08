from .BasicModule import BasicModule
import torch as t
import numpy as np
import json
from torch import nn

kernel_sizes =  [1,2,3,4,5]
class MultiCNNTextBNDeep(BasicModule): 
    def __init__(self, opt ):
        super(MultiCNNTextBNDeep, self).__init__()
        self.model_name = 'MultiCNNTextBNDeep'
        self.opt=opt
        self.encoder = nn.Embedding(opt.vocab_size+1,opt.embedding_dim, padding_idx=498681)
        if opt.embedding_path:
            self.encoder.from_pretrained(self.load_embedding(MyEmbeddings(opt.embedding_path)))

        content_convs = [ nn.Sequential(
                                nn.Conv1d(in_channels = opt.embedding_dim,
                                        out_channels = opt.content_dim,
                                        kernel_size = kernel_size),
                                nn.BatchNorm1d(opt.content_dim),
                                nn.ReLU(inplace=True),
                                nn.Conv1d(in_channels = opt.content_dim,
                                        out_channels = opt.content_dim,
                                        kernel_size = kernel_size),
                                nn.BatchNorm1d(opt.content_dim),
                                nn.ReLU(inplace=True),
                                nn.MaxPool1d(kernel_size = (1780 - kernel_size*2 + 2)),
                            )
            for kernel_size in kernel_sizes ]

        self.content_convs = nn.ModuleList(content_convs)

        self.fc = nn.Sequential(
            nn.Linear(len(kernel_sizes)*(opt.content_dim),opt.linear_hidden_size),
            nn.BatchNorm1d(opt.linear_hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(opt.linear_hidden_size,opt.num_classes)
        )
        
        

    def forward(self, content):
        
        content = self.encoder(content)
        if self.opt.static:
            content.detach()
        content_out = [ content_conv(content.permute(0,2,1)) for content_conv in self.content_convs]
        conv_out = t.cat((content_out),dim=1)
        reshaped = conv_out.view(conv_out.size(0), -1)
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
    pass