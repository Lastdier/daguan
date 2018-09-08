from config import opt
import models
import os
import tqdm
from data.dataset import DC_data
import torch as t
import time
import fire
import torchnet as tnt
from torch.utils import data
from torch.autograd import Variable
from utils.visualize import Visualizer
from utils import get_score
vis = Visualizer(opt.env)
import pandas as pd


def val(model,dataset):
    '''
    计算模型在验证集上的分数
    '''

    dataset.change2val()

    loss_function = getattr(models,opt.loss)() 

    dataloader = data.DataLoader(dataset,
                    batch_size = opt.batch_size,
                    shuffle = False,
                    num_workers = opt.num_workers,
                    pin_memory = True
                    )
    
    a = [0] * 19
    b = [0] * 19
    c = [0] * 19
    loss = 0.
    with t.no_grad():
        for ii,(content,label,_) in tqdm.tqdm(enumerate(dataloader)):
            content,label = content.cuda(),label.cuda()
            score = model(content)
            # !TODO: 优化此处代码
            #       1. append
            #       2. for循环
            #       3. topk 代替sort

            loss += loss_function(score, label.squeeze(1))
            predict = score.data.topk(1,dim=1)[1].cpu().tolist()
            true_target = label.data.cpu().tolist()
            
        
            for jj in range(label.size(0)):
                th_predict = predict[jj][0]
                th_label = true_target[jj][0]
                if th_predict == th_label:
                    a[th_predict] += 1
                else:
                    b[th_predict] += 1
                    c[th_label] += 1
    del score

    ave_f1 = 0
    for jj in range(19):
        if a[jj] == 0:
            continue
        i_prec = float(a[jj]) / (a[jj] + b[jj])
        i_recall = float(a[jj]) / (a[jj] + c[jj])
        ave_f1 += 2 * i_prec * i_recall / (i_prec + i_recall) / 19

    dataset.change2train()
    # model.train()   #???
    return loss.item(), ave_f1


def main(**kwargs):
    '''
    训练入口
    '''

    # opt.parse(kwargs,print_=False)
    # if opt.debug:import ipdb;ipdb.set_trace()

    
    opt.parse(kwargs,print_=True)
    model = getattr(models,opt.model)(opt).cuda()
    print(model)
    vis.reinit(opt.env)
    pre_loss=1.0
    lr,lr2=opt.lr,opt.lr2
    loss_function = getattr(models,opt.loss)()  

    dataset = DC_data(1780, augment=opt.augument)
    dataloader = data.DataLoader(dataset,
                    batch_size = opt.batch_size,
                    shuffle = True,
                    num_workers = 4,
                    pin_memory = True
                    )

    optimizer = model.get_optimizer(lr,opt.lr2,opt.weight_decay)
    loss_meter = tnt.meter.AverageValueMeter()
    best_score = 0

    pred_probs = []
    for fold in range(5):
        if fold > 0:
            dataset.change_fold(fold)
            dataloader = data.DataLoader(dataset,
                        batch_size = opt.batch_size,
                        shuffle = True,
                        num_workers = 4,
                        pin_memory = True
                        )
            del model
            del optimizer
            best_score = 0
            model = getattr(models,opt.model)(opt).cuda()
            print(model)
            lr,lr2=opt.lr,opt.lr2
            optimizer = model.get_optimizer(lr,opt.lr2,opt.weight_decay)
        batch_count = 0
        notimproved_count = 0
        for epoch in range(opt.max_epoch):
            loss_meter.reset()
            a = [0] * 19
            b = [0] * 19
            c = [0] * 19
            for ii,(content,label,_) in tqdm.tqdm(enumerate(dataloader)):
                # 训练 更新参数
                content,label = content.cuda(),label.cuda()
                optimizer.zero_grad()
                score = model(content)
                loss = loss_function(score, label.squeeze(1))
                loss_meter.add(loss.data[0])
                loss.backward()
                optimizer.step()

                predict = score.data.topk(1,dim=1)[1].cpu().tolist()
                true_target = label.data.cpu().tolist()
                
                
                for jj in range(label.size(0)):
                    th_predict = predict[jj][0]
                    th_label = true_target[jj][0]
                    if th_predict == th_label:
                        a[th_predict] += 1
                    else:
                        b[th_predict] += 1
                        c[th_label] += 1
                if batch_count%opt.plot_every ==opt.plot_every-1:
                    ### 可视化
                    if os.path.exists(opt.debug_file):
                        import ipdb
                        ipdb.set_trace()

                    
                    # true_label=true_target[0][:,:5]
                    # predict_label_and_marked_label_list=[]
                    # for jj in range(label.size(0)):
                    #     true_index_=true_index[jj]
                    #     true_label_=true_label[jj]
                    #     true=true_index_[true_label_>0]
                    #     predict_label_and_marked_label_list.append((predict[jj],true.tolist()))
                    
                    # compute average f1 score
                    ave_f1 = 0
                    for jj in range(19):
                        if a[jj] == 0:
                            continue
                        i_prec = float(a[jj]) / (a[jj] + b[jj])
                        i_recall = float(a[jj]) / (a[jj] + c[jj])
                        ave_f1 += 2 * i_prec * i_recall / (i_prec + i_recall) / 19
                    prec_ = sum(a) * 1.0 / opt.batch_size / opt.plot_every
                    a = [0] * 19
                    b = [0] * 19
                    c = [0] * 19
                    vis.vis.text('prec:%s,score:%s' %(prec_,ave_f1),win='tmp')
                    # vis.plot('scores', score_meter.value()[0])
                    vis.plot('scores', ave_f1)
                    
                    #eval()
                    vis.plot('loss', loss_meter.value()[0])
                    k = t.randperm(label.size(0))[0]

                if batch_count%opt.decay_every == opt.decay_every-1:   
                    # 计算在验证集上的分数,并相对应的调整学习率
                    
                    del loss
                    val_loss, val_f1= val(model,dataset)
                    vis.log({' epoch:':epoch,' lr: ':lr,'val_loss':val_loss,'val_f1':val_f1})
                    #
                    if val_f1 <= best_score:
                        notimproved_count += 1
                        if notimproved_count == 5:
                            break
                        state = t.load('rcnnemb_'+str(fold)+'_score' +str(best_score)+'.pt')
                        model.load_state_dict(state)
                        model.cuda()
                        lr = lr * opt.lr_decay
                        lr2= 2e-4 if lr2==0 else  lr2*0.8
                        optimizer = model.get_optimizer(lr,lr2,0)  
                    if val_f1 > best_score:
                        notimproved_count = 0
                        best_score = val_f1
                        t.save(model.cpu().state_dict(), 'rcnnemb_'+str(fold)+'_score'+str(best_score)+'.pt')
                        model.cuda()                      
                    #
                    pre_loss = loss_meter.value()[0]
                    loss_meter.reset()
                batch_count += 1
            if notimproved_count == 5:
                break
        # pred_probs = val_fold(model, dataset, pred_probs)
        # t.save(model.cpu().state_dict(), 'rcnn_fold_final'+str(fold)+'.pt')
    # pred_probs.sort(key=lambda x: len(x[0]))
    # iddd, probsss = zip(*pred_probs)
    # test_prob=pd.DataFrame(probsss)
    # test_prob.columns=["class_prob_%s"%i for i in range(1,20)]
    # test_prob["id"]=iddd
    # test_prob.to_csv('result/rcnno_fold_val_probs.csv',index=None)

if __name__ == '__main__':
    fire.Fire()
