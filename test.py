#coding: utf-8
import random
import numpy as np
import os
import argparse
from tqdm import tqdm
import seaborn  as sns
from matplotlib import pyplot as plt
from PIL import Image

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
from utils.myevaluate import topk_accuracy, confusionmatrix, shot_acc
from utils.mydataset import data_loader
from utils.myevaluate import F1_score

#category = [
        # device-0, scope-2, treatment-5 : 3
        # washing-1 : 0
        # indigocarmine-3 : 1
        # bleeding-4 : 2

# test関数
def test():
    model.eval()
    sum_acc1 = 0
    sum_acc2 = 0
    sum_acc3 = 0
    count = 0
    predict = []
    answer = []
    mean = np.expand_dims(np.expand_dims(np.array([0.4914, 0.4822, 0.4465]), axis=0), axis=0)
    std = np.expand_dims(np.expand_dims(np.array([0.2023, 0.1994, 0.2010]), axis=0), axis=0)
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(tqdm(test_loader, leave=False)):
            inputs = inputs.cuda(device)
            targets = targets.cuda(device)
            targets = targets.long()


            output, _ = model(inputs)


            acc1, acc2, acc3 = topk_accuracy(output, targets, topk=(1, 2, 3))
            sum_acc1 += acc1[0]
            sum_acc2 += acc2[0]
            sum_acc3 += acc3[0]

            output = F.softmax(output, dim=1)
            _, predicted = output.max(1)

            for j in range(predicted.shape[0]):
                predict.append(predicted[j].cpu())
                answer.append(targets[j].cpu())
                if predicted[j].cpu() != targets[j].cpu():
                    img = np.transpose(inputs[j].cpu().numpy(), (1, 2, 0))
                    Image.fromarray(np.uint8((img*std + mean) * 255.0)).save("{}/fold_{}/debug_image/img{}_pre-{}_gt-{}.png".format(args.out, args.fold, count, predicted[j].cpu().item(), targets[j].cpu().item()), quality=95)
                    count+= 1
    results = F1_score(answer, predict, 5, args)
    cm = confusionmatrix(answer, predict, 5, args)
    # many, medium, few = shot_acc(predict, answer, np.array(cls_num_list), many_shot_thr=100, low_shot_thr=20)

    return sum_acc1/(batch_idx+1), round(results["macro_f1"],3), round(results["weighted_f1"],3), cm


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='IMLoss')
    parser.add_argument('--dataset', type=str, default='CIFAR10')
    parser.add_argument('--datatype', type=str, default='exp', help='exp or step')
    parser.add_argument('--batchsize', type=int, default=128)
    parser.add_argument('--Tbatchsize', type=int, default=1)
    parser.add_argument('--out', type=str, default='result')
    parser.add_argument('--gpu', type=int, default=-1)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--cv', type=int, default=5)
    parser.add_argument('--fold', type=int, default=0)
    parser.add_argument('--norm', action='store_true', help='use norm?')
    args = parser.parse_args()
    gpu_flag = args.gpu


    ### device ###
    device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() else 'cpu')

    acc_all = 0
    macro_all = 0
    w_f1_all = 0
    cm_all = 0
    for f in range(args.cv):
        args.fold = f
        
        ### save dir ###
        PATH = "{}/fold_{}/prediction.txt".format(args.out, args.fold)
        with open(PATH, mode = 'w') as f:
            pass
        if not os.path.exists(os.path.join("{}".format(args.out), "fold_{}".format(args.fold), "debug_image")):
      	    os.mkdir(os.path.join("{}".format(args.out), "fold_{}".format(args.fold), "debug_image"))

        ### dataset ###
        _, test_loader, _, n_classes = data_loader(args)
        
        ### model ###
        model = resnet32(num_classes=n_classes, use_norm=args.norm).cuda(device)
        
        ### model load ###
        model_path = "{}/fold_{}/model/model_bestacc.pth".format(args.out, args.fold)
        model.load_state_dict(torch.load(model_path))

        acc, macro_f1, weighted_f1, cm = test()

        print("Fold-{} : Accuracy={:.2f}%  Macro-F1={:.4f}  Weighted-F1={:.4f}".format(args.fold, acc, macro_f1, weighted_f1))

        with open(PATH, mode = 'a') as f:
            f.write("%.4f\t%.4f\t%.4f\n" % (acc, macro_f1, weighted_f1))
        

        acc_all += acc
        macro_all += macro_f1
        w_f1_all += weighted_f1
        cm_all += cm

print("=======================================")
print("Average acc={:.2f}%  Average macro-F1={:.4f}  Average weighted-F1={:.4f}".format((acc_all/5.), (macro_all/5.), (w_f1_all/5.)))

sns.heatmap(cm_all, annot=True, fmt='.0f', cmap='jet')
plt.xticks(fontsize=10) 
plt.yticks(fontsize=10) 
plt.ylabel("Ground truth", fontsize=14)
plt.xlabel("Predicted label", fontsize=14)
#plt.title("AIM mini (Ratio={:2d})".format(int(1/args.ratio)), fontsize=14)
plt.title("AIM mini dataset", fontsize=14)
plt.savefig("{}/ConfusionMatrix.png".format(args.out))
plt.close()