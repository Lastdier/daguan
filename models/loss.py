import torch as t
def multilabelloss():
    return  t.nn.MultiLabelSoftMarginLoss()
def multilabel_marginloss():
    return t.nn.MultiLabelMarginLoss()
def bceloss():
    return t.nn.BCELoss() 
def crossentropy():
    return t.nn.CrossEntropyLoss()
def BCElogits():
    return t.nn.BCEWithLogitsLoss()

def weight_loss():
    pass


def identityloss():
    class Loss:
        def __call__(self,x):
            return x
    return Loss()