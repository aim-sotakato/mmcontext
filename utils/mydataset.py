import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import torchvision.models as models
from torchvision import datasets, transforms

from utils.dataset import IMBALANCECIFAR10, IMBALANCECIFAR100, AIMdataset, AIM_Video_dataset


##### data loader #####
def data_loader(args):
    transform_train = transforms.Compose([transforms.RandomCrop(32, padding=4),
                                          transforms.RandomHorizontalFlip(),
                                          transforms.ToTensor(),
                                          transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])

    transform_val = transforms.Compose([transforms.ToTensor(),
                                        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])


    if args.dataset=='CIFAR10':
        n_classes = 10
        train_dataset = IMBALANCECIFAR10(root='./data', imb_type=args.datatype, imb_factor=args.ratio, rand_number=args.seed, train=True, download=True, transform=transform_train)
        val_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_val)
    elif args.dataset=='CIFAR100':
        n_classes = 100
        train_dataset = IMBALANCECIFAR100(root='./data', imb_type=args.datatype, imb_factor=args.ratio, rand_number=args.seed, train=True, download=True, transform=transform_train)
        val_dataset = datasets.CIFAR100(root='./data', train=False, download=True, transform=transform_val)
    elif args.dataset=='aim_mini':
        transform_train = transforms.Compose([#transforms.RandomCrop(128),
                                              transforms.RandomAffine(degrees=(-10, 10), 
                                                                      translate=(0.1, 0.3), 
                                                                      scale=(0.8, 1.2)), 
                                              transforms.RandomHorizontalFlip(),
                                              transforms.ColorJitter(brightness=0.5, 
                                                                     contrast=0.5, 
                                                                     saturation=0.5, 
                                                                     hue=0.5),
                                              transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
                                              transforms.ToTensor(),
                                              transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])

        transform_val = transforms.Compose([transforms.ToTensor(),
                                            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])

        n_classes = 5
        train_dataset = AIMdataset(root='/home/skato/work/WSC/data', train=True, transform=transform_train, cv=args.cv, fold=args.fold)
        val_dataset = AIMdataset(root='/home/skato/work/WSC/data', train=False, transform=transform_val, cv=args.cv, fold=args.fold)


    elif args.dataset=='aim_video':
        transform_val = transforms.Compose([transforms.ToTensor(),
                                            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])

        n_classes = 5
        val_dataset = AIM_Video_dataset(root='/home/skato/work/WSC/data/Avi_data/Crop', transform=transform_val, video_name=args.video_name)
        train_dataset = val_dataset

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batchsize, shuffle=True, drop_last=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.Tbatchsize, shuffle=False, drop_last=False)

    if args.dataset!='aim_video':
        cls_num_lists = train_dataset.get_cls_num_list()
    else:
        cls_num_lists = []

    return train_loader, val_loader, cls_num_lists, n_classes

