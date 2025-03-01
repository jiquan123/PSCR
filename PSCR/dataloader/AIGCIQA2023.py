import torch
import torchvision.transforms as tr
from torch.utils.data import Dataset
import os
from PIL import Image
import scipy.io
import random
import numpy as np

def random_shuffle(generated_image, label):
    randnum = 1
    np.random.seed(randnum)
    np.random.shuffle(generated_image)
    np.random.seed(randnum)
    np.random.shuffle(label)
    return generated_image, label


# AIGCIQA-2023
def load_image(root_path):
    # 获取数据集目录中所有图片文件的文件名并按照命名顺序排序
    image_files = [f for f in os.listdir(root_path) if f.endswith('.jpg') or f.endswith('.png')]
    image_files = sorted(image_files, key=lambda x: int(x.split('.')[0]))  # AIGCIQA-2023

    image_list = []
    for name in image_files:
        file = os.path.join(root_path, name)
        image = Image.open(file).convert('RGB')
        image_list.append(image)
    return image_list


def load_label(path):
    mat_data = scipy.io.loadmat(path)
    label = mat_data['MOSz']
    label_list = []
    for i in range(len(label)):
        label_list.append(label[i][0])
    return label_list


#select exemplar images from the whole training set
class AIGCIQA2023Dataset(Dataset):
    def __init__(self, root_path, label_path, transforms, split):
        self.transforms = transforms
        self.split_train_img, self.split_train_label, self.split_test_img, self.split_test_label = self.load_data(root_path, label_path)
        self.split = split
        if self.split == 'train':
            self.image = self.split_train_img
            self.label = self.split_train_label
        else:
            self.image = self.split_test_img
            self.label = self.split_test_label
        #print(len(self.image))

    def load_data(self, root_path, label_path):
        image = load_image(root_path)
        label = load_label(label_path)
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

def get_AIGCIQA2023q_dataloaders(args):
    image_path = './Dataset/AIGCIQA2023/Image/allimg'
    label_path = './Dataset/AIGCIQA2023/DATA/MOS/mosz1.mat'

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

    dataloaders['train'] = torch.utils.data.DataLoader(AIGCIQA2023Dataset(image_path, label_path, train_transforms, split='train'),
                                                       batch_size=args.train_batch_size,
                                                       shuffle=True,
                                                       pin_memory=True)

    dataloaders['test'] = torch.utils.data.DataLoader(AIGCIQA2023Dataset(image_path, label_path, test_transforms, split='test'),
                                                      batch_size=args.test_batch_size,
                                                      shuffle=False,
                                                      pin_memory=True)
    return dataloaders


def get_AIGCIQA2023a_dataloaders(args):
    image_path = './Dataset/AIGCIQA2023/Image/allimg'
    label_path = './Dataset/AIGCIQA2023/DATA/MOS/mosz2.mat'

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

    dataloaders['train'] = torch.utils.data.DataLoader(AIGCIQA2023Dataset(image_path, label_path, train_transforms, split='train'),
                                                       batch_size=args.train_batch_size,
                                                       shuffle=True,
                                                       pin_memory=True)

    dataloaders['test'] = torch.utils.data.DataLoader(AIGCIQA2023Dataset(image_path, label_path, test_transforms, split='test'),
                                                      batch_size=args.test_batch_size,
                                                      shuffle=False,
                                                      pin_memory=True)
    return dataloaders


def get_AIGCIQA2023c_dataloaders(args):
    image_path = './Dataset/AIGCIQA2023/Image/allimg'
    label_path = './Dataset/AIGCIQA2023/DATA/MOS/mosz3.mat'

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

    dataloaders['train'] = torch.utils.data.DataLoader(AIGCIQA2023Dataset(image_path, label_path, train_transforms, split='train'),
                                                       batch_size=args.train_batch_size,
                                                       shuffle=True,
                                                       pin_memory=True)

    dataloaders['test'] = torch.utils.data.DataLoader(AIGCIQA2023Dataset(image_path, label_path, test_transforms, split='test'),
                                                      batch_size=args.test_batch_size,
                                                      shuffle=False,
                                                      pin_memory=True)
    return dataloaders







