import numpy as np
import torch
import torch.utils.data as data
from torchvision import datasets, transforms
import os
from PIL import Image, ImageOps
import json
from collections import Counter
from collections import defaultdict
import random
import glob
from natsort import natsorted
from pathlib import Path

class AIMdataset(data.Dataset):
    cls_num = 5
    np.random.seed(42)
    random.seed(42)
    def __init__(self, root=None, train=True, transform=None, cv=5, fold=0):
        class_names = ["device", "washing", "scope", "indigocarmine", "bleeding", "treatment"]
        # device-0,scope-2,treatment-5 : 0
        # washing-1 : 1
        # indigocarmine-3 : 2
        # bleeding-4 : 3
        
        self.root = root
        self.train = train
        self.transform = transform
        self.cv = cv
        self.fold = fold
        self.item_image = []
        self.item_gt = []

        with open(self.root + '/MMContext_situation_washing0414_coco_caseid.json', 'r', encoding='utf-8') as f:
            coco_data = json.load(f)
        with open(self.root + '/MMContext_situation_washing0414_coco.json', 'r', encoding='utf-8') as f:
            coco_data_mini = json.load(f)
        with open(self.root + '/noncan_data.json', 'r', encoding='utf-8') as f:
            noncan_data_mini = json.load(f)
        
        ### cross validation ###
        case_list = []
        for ann in coco_data['images']:
            case = ann.get('case_id')
            if case is None:
                continue
            if isinstance(case, list):
                if len(case) > 0:
                    case_list.append(case[0])
            elif isinstance(case, str):
                case_list.append(case)
        case_counts = Counter(case_list)

        n_samples = len(case_counts)
        indices = np.arange(n_samples)
        np.random.shuffle(indices)
        fold_sizes = np.full(self.cv, n_samples // self.cv, dtype=int)
        fold_sizes[:n_samples % self.cv] += 1

        current = 0
        folds = []
        for fold_size in fold_sizes:
            start, stop = current, current + fold_size
            folds.append(indices[start:stop])
            current = stop

        test_idx = folds[fold]
        train_idx = np.hstack([folds[i] for i in range(self.cv) if i != self.fold])
        
        case_counts_list = list(case_counts.items())
        train_data = [case_counts_list[i][0] for i in train_idx]
        test_data = [case_counts_list[i][0] for i in test_idx]


        #######################################################
        name_to_paths = defaultdict(list)
        for non_path in noncan_data_mini:
            name = os.path.basename(os.path.dirname(non_path))
            name_to_paths[name].append(non_path)

        unique_names = list(name_to_paths.keys())
        random.shuffle(unique_names)
        non_folds = [[] for _ in range(self.cv)]
        for i, name in enumerate(unique_names):
            non_folds[i % self.cv].append(name)
        
        non_val_names = set(non_folds[self.fold])
        non_train_names = set(name for i, f in enumerate(non_folds) if i != self.fold for name in f)

        non_val_paths = [p for name in non_val_names for p in name_to_paths[name]]
        non_train_paths = [p for name in non_train_names for p in name_to_paths[name]]

        ############################################################
        if self.train:
            label_json = train_data
            label_non = non_train_paths
            print("Training data : {} case".format(len(train_data)+len(label_non)))
        else:
            label_json = test_data
            label_non = non_val_paths
            print("Test data : {} case".format(len(test_data)+len(label_non)))

        for case in label_json:
            id_list = []
            for img in coco_data['images']:
                if img['case_id'] == case:
                    id_list.append(img['id'])
            matched_annotations = [ann for ann in coco_data_mini['annotations'] if ann['image_id'] in id_list]
            
            for ann in matched_annotations:
                target_image_id = ann.get('id')
                situation = ann.get('situation')
                image_path = next((im.get('file_name') for im in coco_data['images'] if str(im.get('id')) == str(target_image_id)), None)    
                self.item_image.append("{}/250411_MMcontext_situation/{}".format(self.root, image_path))
                if situation[0]=='washing':
                    label = 0
                elif situation[0]=='indigocarmine':
                    label = 1
                elif situation[0]=='bleeding':
                    label = 2
                else:
                    label = 3
                #label = class_names.index(situation[0])
                self.item_gt.append(label)                
                #Image.open("250411_MMcontext_situation/{}".format(image_path)).save(save_path+"/{}".format(os.path.basename(image_path)))

        for case_non in label_non:
            self.item_image.append("{}/".format(self.root) + case_non)
            self.item_gt.append(4)  

        count_gt = np.bincount(np.int64(self.item_gt).flatten())
        for ccc in range(len(count_gt)):
            print("Class {} : {} samples".format(ccc, count_gt[ccc]))


    def __getitem__(self, index):
        label = self.item_gt[index]
        image = Image.open("{}".format(self.item_image[index])).convert("RGB")
        image = image.resize((224, 224))
        
        if self.transform:
            image = self.transform(image)

        return image, label

    def __len__(self):
        return len(self.item_image)


    def get_cls_num_list(self):
        cls_num_list = np.bincount(np.int64(self.item_gt).flatten()) 
        return cls_num_list


###################################################################33
class AIM_Video_dataset(data.Dataset):
    cls_num = 5
    np.random.seed(42)
    random.seed(42)
    def __init__(self, root=None, transform=None, video_name=None):
        class_names = ["device", "washing", "scope", "indigocarmine", "bleeding", "treatment"]
        # device-0,scope-2,treatment-5 : 0
        # washing-1 : 1
        # indigocarmine-3 : 2
        # bleeding-4 : 3
        # normal : 4

        self.root = root
        self.transform = transform

        frame_list = natsorted(glob.glob(self.root + "/" + video_name + "/*.png"))

        self.item_image = frame_list

        print("Video {} : {} samples".format(video_name, len(self.item_image)))


    def __getitem__(self, index):
        label = 0
        image = Image.open("{}".format(self.item_image[index])).convert("RGB")
        image = image.crop((640, 0, 1920, 1080))
        image = image.resize((224, 224))

        parts = Path(self.item_image[index]).parts
        last = parts[-1]
        
        if self.transform:
            image = self.transform(image)

        return image, label, last

    def __len__(self):
        return len(self.item_image)


