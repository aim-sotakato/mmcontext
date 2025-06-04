#coding: utf-8
import random
import numpy as np
import os
import argparse
from tqdm import tqdm

import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import torchvision.models as models
from torchvision import datasets, transforms
import torch.backends.cudnn as cudnn

from utils.network import *
from utils.mydataset import data_loader
from utils.myloss import choice_loss, class_weight
from utils.myscheduler import adjust_learning_rate
from utils.myaug import mix_data, mix_criterion
from utils.myevaluate import F1_score

try:
    import timm
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "timm"])
    import timm

#import timm 
from timm.scheduler import CosineLRScheduler

from clearml import Task, Dataset


##### train ######
def train(epoch, criterion):
    model.train()
    sum_loss = 0
    correct = 0
    total = 0

    for batch_idx, (inputs, targets) in enumerate(tqdm(train_loader, leave=False)):
        inputs = inputs.cuda(device)
        targets = targets.cuda(device)
        targets = targets.long()

        ### hard augmentation ###
        if 'Mix' in args.augmentation:
            inputs, targets_a, targets_b, lam = mix_data(inputs, targets, args.augmentation)
            inputs, targets_a, targets_b = map(Variable, (inputs, targets_a, targets_b))


        output, features = model(inputs)

        ### loss ###
        if 'Mix' in args.augmentation:
            if 'IB' in args.loss:
                loss = mix_criterion(criterion, output, targets_a, targets_b, lam, features, epoch)
            else:
                loss = mix_criterion(criterion, output, targets_a, targets_b, lam, None, None)

        else:
            if 'IB' in args.loss:
                loss = criterion(output, targets, features, epoch)
            else:
                loss = criterion(output, targets)


        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        output = F.softmax(output, dim=1)
        _, predicted = output.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        sum_loss += loss.item()
 
    return sum_loss/(batch_idx+1), float(correct)/float(total)


###### test #######
def test(epoch, criterion):
    model.eval()
    sum_loss = 0
    correct = 0
    total = 0
    predict = []
    answer = []
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(tqdm(val_loader, leave=False)):
            inputs = inputs.cuda(device)
            targets = targets.cuda(device)
            targets = targets.long()

            output, features = model(inputs)

            ### loss ###
            if 'IB' in args.loss:
                loss = criterion(output, targets, features, epoch)
            else:
                loss = criterion(output, targets)
  

            output = F.softmax(output, dim=1)
            _, predicted = output.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            sum_loss += loss.item()
    
            for j in range(predicted.shape[0]):
                predict.append(predicted[j].cpu())
                answer.append(targets[j].cpu())
        
    results = F1_score(answer, predict, 5, args)

    return sum_loss/(batch_idx+1), float(correct)/float(total), round(results["macro_f1"],3), round(results["weighted_f1"],3)



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='IMLoss')
    parser.add_argument('--project', type=str, default='screening_mmcontext')
    parser.add_argument('--task_name', type=str, default='screening_classification_sample')
    parser.add_argument('--dataset', type=str, default='CIFAR10')
    parser.add_argument('--datatype', type=str, default='exp', help='exp or step')
    parser.add_argument('--data_id', type=str, default='bdbaccd54ac7414f886a7766e703e0fc')
    parser.add_argument('--batchsize', type=int, default=32)
    parser.add_argument('--Tbatchsize', type=int, default=4)
    parser.add_argument('--num_epochs', type=int, default=200)
    parser.add_argument('--out', type=str, default='result')
    parser.add_argument('--cv', type=int, default=5)
    parser.add_argument('--fold', type=int, default=0)
    parser.add_argument('--gpu', type=int, default=-1)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--lr', type=float, default=0.1)
    parser.add_argument('--loss', type=str, default='CE', choices=['CE','Focal','CBW','GR','BS','LADE','LDAM','LA',
                                                                   'VS','IB','IBFL','ELM','FCE','LAFCE'], help='loss name')
    parser.add_argument('--norm', action='store_true', help='use norm?')
    parser.add_argument('--weight_rule', type=str, default='None', help='CBReweight or IBReweight')
    parser.add_argument('--weight_scheduler', type=str, default='None', help='DRW')
    parser.add_argument('--augmentation', type=str, default='None', help='Mixup or CutMix')
    #args = parser.parse_args()
    args, unknown = parser.parse_known_args()


    ####### ClearML task initialization ########
    task = Task.init(project_name='screening_mmcontext', 
                     task_name='screening_classification_sample',
                     )
    task.set_base_docker("harbor.dev.ai-ms.com/screening_mmcontext/mmcontext_image:latest")
    task.execute_remotely(queue_name="a100x1a", exit_process=True)

    args = task.connect(args)


    gpu_flag = args.gpu

    ### plot ###
    print("[Experimental conditions]")
    print(" Cross Val      : {}".format(args.cv))
    print(" Fold           : {}".format(args.fold))
    print(" GPU ID         : {}".format(args.gpu))
    print(" Dataset        : {}".format(args.dataset))
    print(" Loss function  : {}".format(args.loss))
    print(" Class weight   : {}".format(args.weight_rule))
    print(" CW scheduler   : {}".format(args.weight_scheduler))
    print(" Augmentation   : {}".format(args.augmentation))
    print("")


    dataset = Dataset.get(dataset_id=args.data_id)
    dataset_path = dataset.get_local_copy()
    args.data_path = dataset_path

    ### save dir ###
    if not os.path.exists("{}".format(args.out)):
      	os.mkdir("{}".format(args.out))
    if not os.path.exists(os.path.join("{}".format(args.out), "fold_{}".format(args.fold))):
      	os.mkdir(os.path.join("{}".format(args.out), "fold_{}".format(args.fold)))
    if not os.path.exists(os.path.join("{}".format(args.out), "fold_{}".format(args.fold), "model")):
      	os.mkdir(os.path.join("{}".format(args.out), "fold_{}".format(args.fold), "model"))

    PATH_1 = "{}/fold_{}/trainloss.txt".format(args.out, args.fold)
    PATH_2 = "{}/fold_{}/testloss.txt".format(args.out, args.fold)
    PATH_3 = "{}/fold_{}/trainaccuracy.txt".format(args.out, args.fold)
    PATH_4 = "{}/fold_{}/testaccuracy.txt".format(args.out, args.fold)

    with open(PATH_1, mode = 'w') as f:
        pass
    with open(PATH_2, mode = 'w') as f:
        pass
    with open(PATH_3, mode = 'w') as f:
        pass
    with open(PATH_4, mode = 'w') as f:
        pass


    ### seed ###
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


    ### device ###
    device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() else 'cpu')


    ### dataset ###
    train_loader, val_loader, cls_num_list, n_classes = data_loader(args)

    cls_num_list = torch.Tensor(cls_num_list).cuda(device)

    ### model ###
    #model = resnet32(num_classes=n_classes, use_norm=args.norm).cuda(device)
    model = Swin_s(num_classes=n_classes).cuda(device)


    ### optimizer ###
    #optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=2e-4)
    optimizer = torch.optim.AdamW(model.parameters())
    scheduler = CosineLRScheduler(optimizer, t_initial=40, 
                                  lr_min=1e-4, warmup_t=5, 
                                  warmup_lr_init=5e-5, warmup_prefix=True)

    sample_acc = 0
    sample_loss = 1000000
    ##### training & test #####
    for epoch in range(args.num_epochs):
        ### learning rate scheduler ###
        #adjust_learning_rate(optimizer, epoch, args.lr)

        ### class balancing weight ###
        per_class_weights = class_weight(args, cls_num_list, epoch)

        ### loss function ###
        criterion = choice_loss(args, cls_num_list, per_class_weights, device)

        ### training ###
        loss_train, acc_train = train(epoch, criterion)
        
        scheduler.step(epoch+1)

        ### test ###
        loss_test, acc_test, macro_f1, w_f1 = test(epoch, criterion)

        print("Epoch{:3d}/{:3d}  TrainLoss={:.4f}  TestAccuracy={:.2f}%  MacroF1={:.4f}  WeightedF1={:.4f}".format(epoch+1, args.num_epochs, loss_train, acc_test*100, macro_f1, w_f1))

        with open(PATH_1, mode = 'a') as f:
            f.write("\t%d\t%f\n" % (epoch+1, loss_train))
        with open(PATH_2, mode = 'a') as f:
            f.write("\t%d\t%f\n" % (epoch+1, loss_test))
        with open(PATH_3, mode = 'a') as f:
            f.write("\t%d\t%f\n" % (epoch+1, acc_train))
        with open(PATH_4, mode = 'a') as f:
            f.write("\t%d\t%f\t%f\t%f\n" % (epoch+1, acc_test, macro_f1, w_f1))

        if macro_f1 >= sample_acc:
           sample_acc = macro_f1
           PATH_best ="{}/fold_{}/model/model_bestacc.pth".format(args.out, args.fold)
           torch.save(model.state_dict(), PATH_best)

        if loss_train <= sample_loss:
           sample_loss = loss_train
           PATH_best ="{}/fold_{}/model/model_bestloss.pth".format(args.out, args.fold)
           torch.save(model.state_dict(), PATH_best)


