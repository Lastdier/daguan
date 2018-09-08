import pandas as pd
from gensim.models import Word2Vec


class MySentences(object):
    
    def __init__(self):
        train_f = 'new_data/train_set.csv'
        test_f = 'new_data/test_set.csv'
        train = pd.read_csv(train_f)
        test = pd.read_csv(test_f)
        self.train_a = pd.concat([train['article'], test['article']], ignore_index=True)
        del train
        del test

    def __iter__(self):
        for s in self.train_a:
            yield s.split(' ')

sentences = MySentences()
model = Word2Vec(sentences, size=300, window=5, min_count=3, workers=11, sg=0, hs=0, seed=1995)
model.wv.save_word2vec_format('char_vector_sg_hs.300dim', binary=False)
