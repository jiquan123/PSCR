# -*- coding: utf-8 -*-


import torch
import logging
import math
import torch.nn as nn
import torchvision.transforms as tr
import random



def get_logger(filepath, log_info):
    logger = logging.getLogger(filepath)
    logger.setLevel(logging.INFO)
    fh = logging.FileHandler(filepath)
    fh.setLevel(logging.INFO)
    logger.addHandler(fh)
    logger.info('-' * 30 + log_info + '-' * 30)
    return logger


def log_and_print(logger, msg):
    logger.info(msg)
    print(msg)


# Overlapping Patches Sampling
def OPS(image, image_size, start_idx):
    '''input_img_size: 3,512,512
    resnet:start_idx[0, 150, 288]
    inceptionv4:start_idx[0, 100, 213]
    '''
    image_list = []
    for i in start_idx:
        for j in start_idx:
            image_patch = image[:, :, i:i + image_size, j:j + image_size]
            image_list.append(image_patch)
    patch_image = torch.cat(image_list, dim=0)
    return patch_image

def NOPS(image, image_size):
    image_list = []
    for i in range(2):
        for j in range(2):
            image_patch = image[:, :, i * image_size:(i + 1) * image_size, j * image_size:(j + 1) * image_size]
            image_list.append(image_patch)
    patch_image = torch.cat(image_list, dim=1)
    return patch_image

# Random Patches Sampling
def RPS(image, image_size):
    image_list = []
    B, C, H, W = image.shape
    random_h = random.sample(range(H - image_size + 1), 9)
    random_w = random.sample(range(W - image_size + 1), 9)
    for i in range (9):
        image_list.append(image[:, :, random_h[i] : random_h[i] + image_size, random_w[i] : random_w[i] + image_size])
    patch_image = torch.cat(image_list, dim=0)
    return patch_image



