#coding: utf-8
import random
import numpy as np
import os
import argparse
from tqdm import tqdm
import seaborn  as sns
from matplotlib import pyplot as plt
from PIL import Image
import cv2

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
    model1.eval()
    #model2.eval()
    #model3.eval()
    #model4.eval()
    #model5.eval()
    count = 0
    predict = []
    answer = []
    mean = np.expand_dims(np.expand_dims(np.array([0.4914, 0.4822, 0.4465]), axis=0), axis=0)
    std = np.expand_dims(np.expand_dims(np.array([0.2023, 0.1994, 0.2010]), axis=0), axis=0)
    with torch.no_grad():
        for batch_idx, (inputs, targets, frame) in enumerate(tqdm(test_loader, leave=False)):
            inputs = inputs.cuda(device)

            output1, _, att, _ = model1(inputs)
            #output2, _ = model2(inputs)
            #output3, _ = model3(inputs)
            #output4, _ = model4(inputs)
            #output5, _ = model5(inputs)
            
            output = output1# + output2 + output3 + output4 + output5

            output = F.softmax(output, dim=1)
            _, predicted = output.max(1)

            for j in range(predicted.shape[0]):
                predict.append(predicted[j].cpu())
                att_np = att[j, 0].cpu().numpy()
                att_np = (att_np - att_np.min())/(att_np.max() - att_np.min())
                jet_colored = cv2.applyColorMap(np.uint8(att_np*255.0), cv2.COLORMAP_JET)
                jet_colored = cv2.resize(jet_colored, (inputs[j].shape[2], inputs[j].shape[1]))
                img = np.transpose(inputs[j].cpu().numpy(), (1, 2, 0))
                img = np.uint8((img*std + mean) * 255.0)
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                img_color = cv2.addWeighted(img, 0.7, jet_colored, 0.3, 0)
                combined = np.hstack((img, img_color))
                cv2.imwrite("{}/{}/att/{}_att.png".format(args.out, args.video_name, frame[j][:-4]), combined)

                with open(PATH, mode = 'a') as f:
                    f.write("%s\t%d\n" % (frame[j][:-4], predicted[j].item()))
                #img = np.transpose(inputs[j].cpu().numpy(), (1, 2, 0))
                #Image.fromarray(np.uint8((img*std + mean) * 255.0)).save("sampe.png", quality=95)
                #print(zzzzz)    
    predict = np.array(predict)
    print(np.bincount(np.int64(predict).flatten()))
    return predict


def make_model(n_classes, device, fold, args):
    ### model ###
    model = resnet32_abn(num_classes=n_classes, use_norm=args.norm).cuda(device)
    ### model load ###
    model_path = "{}/fold_{}/model/model_bestacc.pth".format(args.out, fold)
    model.load_state_dict(torch.load(model_path))
    return model

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='IMLoss')
    parser.add_argument('--dataset', type=str, default='CIFAR10')
    parser.add_argument('--batchsize', type=int, default=1)
    parser.add_argument('--Tbatchsize', type=int, default=1)
    parser.add_argument('--out', type=str, default='result')
    parser.add_argument('--gpu', type=int, default=-1)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--norm', action='store_true', help='use norm?')
    parser.add_argument('--video_name', type=str, default='juntendo-room12-20210517-122015')
    args = parser.parse_args()

    gpu_flag = args.gpu


    ### device ###
    device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() else 'cpu')

    if not os.path.exists(os.path.join("{}".format(args.out), "{}".format(args.video_name))):
      	os.mkdir(os.path.join("{}".format(args.out), "{}".format(args.video_name)))
    if not os.path.exists(os.path.join("{}".format(args.out), "{}".format(args.video_name), "att")):
      	os.mkdir(os.path.join("{}".format(args.out), "{}".format(args.video_name), "att"))
    
    PATH = "{}/{}/prediction.txt".format(args.out, args.video_name)
    with open(PATH, mode = 'w') as f:
        pass

    _, test_loader, _, n_classes = data_loader(args)
    
    ### model ###
    model1 = make_model(n_classes, device, 0, args)
    #model2 = make_model(n_classes, device, 1, args)
    #model3 = make_model(n_classes, device, 2, args)
    #model4 = make_model(n_classes, device, 3, args)
    #model5 = make_model(n_classes, device, 4, args)

    results = test()
    