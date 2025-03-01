import torch
import torchvision.transforms as tr
from torch.utils.data import Dataset
import os
from PIL import Image
import pandas as pd
import random
import numpy as np

def random_shuffle(generated_image, label):
    randnum = 1
    np.random.seed(randnum)
    np.random.shuffle(generated_image)
    np.random.seed(randnum)
    np.random.shuffle(label)
    return generated_image, label


# AGIQA-1k

def load_image_label(image_path, label_path):
    data = pd.read_excel(label_path)
    labels = data['MOS']
    image_files_name = data['Image']

    image_list = []
    for name in image_files_name:
        file = os.path.join(image_path, str('{}'.format(name)))
        image = Image.open(file).convert('RGB')
        image_list.append(image)
    label_list = []
    for label in labels:
        label_list.append(label)
    return image_list, label_list


class AGIQA1kDataset(Dataset):
    def __init__(self, image_path, label_path, transforms, split):
        self.transforms = transforms
        self.split_train_img, self.split_train_label, self.split_test_img, self.split_test_label = self.load_data(image_path, label_path)
        self.split = split
        if self.split == 'train':
            self.image = self.split_train_img
            self.label = self.split_train_label
        else:
            self.image = self.split_test_img
            self.label = self.split_test_label
        #print(len(self.image))

    def load_data(self, image_path, label_path):
        image, label = load_image_label(image_path, label_path)
        image, label = random_shuffle(image, label)
        percent = int(len(image) * 0.8)
        train_image = image[:percent]
        train_label = label[:percent]
        test_image = image[percent:]
        test_label = label[percent:]
        '''train_image = []
        train_label = []
        test_image = []
        test_label = []
        for i in range(len(image)):
            if i % 4 == 3:
                test_image.append(image[i])
                test_label.append(label[i])
            else:
                train_image.append(image[i])
                train_label.append(label[i])'''
        return train_image, train_label, test_image, test_label

    def __len__(self):
        return len(self.label)

    def __getitem__(self, idx):
        data = {}
        data['img'] = self.transforms(self.image[idx])
        data['label'] = self.label[idx]
        if self.split == 'train':
            target_image_list = self.split_train_img.copy()
            target_label_list = self.split_train_label.copy()
            # exclude self
            if len(target_image_list) > 1:
                target_image_list.pop(idx)
                target_label_list.pop(idx)

            # choosing one out
            target = {}
            tmp_idx = random.randint(0, len(target_image_list) - 1)
            target['img'] = self.transforms(target_image_list[tmp_idx])
            target['label'] = target_label_list[tmp_idx]
            #print(type(target['img']))

            return data, target

        else:
            target_image_list = self.split_train_img.copy()
            target_label_list = self.split_train_label.copy()
            random_numbers_list = random.sample(range(len(target_image_list)), 10)
            target_list = []
            for i in random_numbers_list:
                target = {}
                target['img'] = self.transforms(target_image_list[i])
                target['label'] = target_label_list[i]
                target_list.append(target)
            return data, target_list


def get_AGIQA1K_dataloaders(args):
    label_path = "./Dataset/AGIQA-1k/AIGC_MOS_Zscore.xlsx"
    image_path = "./Dataset/AGIQA-1k/file"

    if args.backbone == 'inceptionv4':
        resize_img_size = 320
        crop_img_size = 299
    else:
        resize_img_size = 256
        crop_img_size = 224

    if args.PS:
        train_transforms = tr.Compose([
            tr.RandomHorizontalFlip(),
            tr.ToTensor(),
            #tr.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            tr.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        test_transforms = tr.Compose([
            tr.ToTensor(),
            #tr.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            tr.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    else:
        train_transforms = tr.Compose([
            tr.Resize(resize_img_size),
            tr.RandomCrop(crop_img_size),
            tr.RandomHorizontalFlip(),
            tr.ToTensor(),
            #tr.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            tr.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        test_transforms = tr.Compose([
            tr.Resize(resize_img_size),
            tr.CenterCrop(crop_img_size),
            tr.ToTensor(),
            #tr.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            tr.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])


    dataloaders = {}

    dataloaders['train'] = torch.utils.data.DataLoader(AGIQA1kDataset(image_path, label_path, train_transforms, split='train'),
                                                       batch_size=args.train_batch_size,
                                                       shuffle=True,
                                                       pin_memory=True)

    dataloaders['test'] = torch.utils.data.DataLoader(AGIQA1kDataset(image_path, label_path, test_transforms, split='test'),
                                                      batch_size=args.test_batch_size,
                                                      shuffle=False,
                                                      pin_memory=True)
    return dataloaders

