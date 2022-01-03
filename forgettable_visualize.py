import pdb
import argparse
import numpy as np
import numpy.random as npr
import time
import os
import sys
import pickle
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
from torch.optim.lr_scheduler import MultiStepLR

from torchvision.utils import make_grid
from torchvision import datasets, transforms

import xlsxwriter
import csv

# Format time for printing purposes
def get_hms(seconds):
    m, s = divmod(seconds, 60)
    h, m = divmod(m, 60)

    return h, m, s


# Evaluate model predictions on heldout test data
#
# example_stats: dictionary containing statistics accumulated over every presentation of example
#
def save_cifar_image(array, path):
    # array is 3x32x32. cv2 needs 32x32x3i
    print('array.shape before',array.shape)
    #array = array.transpose(1,2,0)
    print('array.shape after',array.shape)
    # array is RGB. cv2 needs BGR
    array = cv2.cvtColor(array, cv2.COLOR_RGB2BGR)
    # save to PNG file
    return cv2.imwrite(path, array)


model_options = ['resnet32', 'resnet_grasp']
dataset_options = ['cifar10', 'cifar100']


# Image Preprocessing
normalize = transforms.Normalize(
    mean=[x / 255.0 for x in [125.3, 123.0, 113.9]],
    std=[x / 255.0 for x in [63.0, 62.1, 66.7]])

# Setup train transforms
train_transform = transforms.Compose([])
train_transform.transforms.append(transforms.ToTensor())
train_transform.transforms.append(normalize)

# Setup test transforms
test_transform = transforms.Compose([transforms.ToTensor(), normalize])

dataset = "cifar10"

# Load the appropriate train and test datasets
if dataset == 'cifar10':
    num_classes = 10
    train_dataset = datasets.CIFAR10(
        root='./data.cifar10',
        train=True,
        transform=train_transform,
        download=True)

    test_dataset = datasets.CIFAR10(
        root='./data.cifar10',
        train=False,
        transform=test_transform,
        download=True)
elif dataset == 'cifar100':
    num_classes = 100
    train_dataset = datasets.CIFAR100(
        root='./data.cifar100',
        train=True,
        transform=train_transform,
        download=True)

    test_dataset = datasets.CIFAR100(
        root='./data.cifar100',
        train=False,
        transform=test_transform,
        download=True)


def visualization(pkl_model):
    ######visualize image########

    indices = pkl_model['indices']  # [15000:]
    train_dataset.data = train_dataset.data[indices, :, :, :]
    train_dataset.targets = np.array(
        train_dataset.targets)[indices].tolist()
    print('len(train_dataset.data)', len(train_dataset.data))

    output_dir = "./figs/MBST/visualization/"
    os.makedirs(output_dir, exist_ok=True)

    for i in range(49850, 49999):
        print(i)
        print('target', train_dataset.targets[i])
        # save_cifar_image(train_dataset.data[i], os.path.join(output_dir, 'most_unforget_{}.png'.format(i)))
        save_cifar_image(train_dataset.data[i], os.path.join(output_dir, 'most_forget_{}_target'.format(i) + str(
            train_dataset.targets[i]) + '.png'))


def ordered_idx_and_counts(pkl_model):
    # make list for index and counts

    print('ordered_indx', pkl_model['indices'])
    print('ordered_counts', pkl_model['forgetting counts'])

    idx = np.array(pkl_model['indices'])
    counts = np.array(pkl_model['forgetting counts'])

    print(idx.shape)
    print(counts.shape)

    Nth = 0
    num_unforget = np.sum(counts <= Nth)

    print("unforgetable (N <= {}): {}".format(num_unforget, Nth))


def forgetting_trend(pkl_model):
    # make list for index and counts

    epoch_all = np.array(pkl_model['epoch'])
    num_forget_all = np.array(pkl_model['forgetting counts'])

    print(num_forget_all)




##########################################################################################

input_file = "pkl/irr_0.6_MwE_final_unforget_15254.pkl"

with open(os.path.join(input_file), 'rb') as fin:
    pkl_model = pickle.load(fin)

visualization(pkl_model)

# ordered_idx_and_counts(pkl_model)

# forgetting_trend(pkl_model)


#np.savetxt("foo.csv", indx_and_counts, delimiter=",")
# with open('list.csv', 'w') as f:
#     writer = csv.writer(f)
#     writer.writerows(zip(indx_and_counts['indices'], indx_and_counts['forgetting counts']))


# ordered_indx {'indices': array([48848, 40348, 34448, ..., 35126,  7397, 22755]), 'forgetting counts': array([ 0,  0,  0, ..., 35, 36, 40])}




