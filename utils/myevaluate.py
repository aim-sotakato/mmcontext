import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
from sklearn.metrics import confusion_matrix
import seaborn  as sns
from matplotlib import pyplot as plt


def confusionmatrix(answer, predict, n_classes, args):
    label = list(range(n_classes))

    mat = confusion_matrix(answer, predict, labels=label)

    sns.heatmap(mat, annot=True, fmt='.0f', cmap='jet')
    plt.xticks(fontsize=10) 
    plt.yticks(fontsize=10) 
    plt.ylabel("Ground truth", fontsize=14)
    plt.xlabel("Predicted label", fontsize=14)
    #plt.title("AIM mini (Ratio={:2d})".format(int(1/args.ratio)), fontsize=14)
    plt.title("AIM mini dataset", fontsize=14)
    plt.savefig("{}/fold_{}/ConfusionMatrix.png".format(args.out, args.fold))
    plt.close()

    return mat

def f1_from_multiclass_confusion_matrix(cm):
    cm = np.array(cm)
    n_classes = cm.shape[0]
    
    precision_list = []
    recall_list = []
    f1_list = []
    support_list = []  # 各クラスの正解数（= 実際の件数）

    for i in range(n_classes):
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp
        support = cm[i, :].sum()

        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

        precision_list.append(precision)
        recall_list.append(recall)
        f1_list.append(f1)
        support_list.append(support)

    support_sum = sum(support_list)

    macro_f1 = np.mean(f1_list)
    weighted_f1 = np.average(f1_list, weights=support_list)

    return {"per_class_f1": f1_list,
            "macro_f1": macro_f1,
            "weighted_f1": weighted_f1,
            "per_class_precision": precision_list,
            "per_class_recall": recall_list}

def F1_score(answer, predict, n_classes, args):
    label = list(range(n_classes))

    mat = confusion_matrix(answer, predict, labels=label)
    
    results = f1_from_multiclass_confusion_matrix(mat)
    
    return results



def topk_accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)
        output = F.softmax(output, dim=1)
        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))

        return res


def shot_acc(preds, labels, train_data, many_shot_thr=100, low_shot_thr=20, acc_per_cls=False):
    if isinstance(train_data, np.ndarray):
        training_labels = np.array(train_data).astype(int)
    else:
        training_labels = np.array(train_data.dataset.labels).astype(int)

    if isinstance(preds, torch.Tensor):
        preds = preds.detach().cpu().numpy()
        labels = labels.detach().cpu().numpy()
    elif isinstance(preds, np.ndarray):
        pass
    else:
        raise TypeError('Type ({}) of preds not supported'.format(type(preds)))
    train_class_count = []
    test_class_count = []
    class_correct = []
    for l in np.unique(labels):
        train_class_count.append(len(training_labels[training_labels == l]))
        test_class_count.append(len(labels[labels == l]))
        class_correct.append((preds[labels == l] == labels[labels == l]).sum())
    print(class_correct)
    many_shot = []
    median_shot = []
    low_shot = []
    for i in range(len(train_data)):
        if train_data[i] > many_shot_thr:
            many_shot.append((class_correct[i] / test_class_count[i]))
        elif train_data[i] < low_shot_thr:
            low_shot.append((class_correct[i] / test_class_count[i]))
        else:
            median_shot.append((class_correct[i] / test_class_count[i]))    
 
    if len(many_shot) == 0:
        many_shot.append(0)
    if len(median_shot) == 0:
        median_shot.append(0)
    if len(low_shot) == 0:
        low_shot.append(0)

    if acc_per_cls:
        class_accs = [c / cnt for c, cnt in zip(class_correct, test_class_count)] 
        return np.mean(many_shot), np.mean(median_shot), np.mean(low_shot), class_accs
    else:
        return np.mean(many_shot), np.mean(median_shot), np.mean(low_shot)
