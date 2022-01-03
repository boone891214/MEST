from __future__ import print_function
import argparse
import torch
import torch.nn.functional as F
from torchvision import datasets, transforms, models

import os
from torch.autograd import Variable
import matplotlib.pyplot as plt
from testers import *
import random


parser = argparse.ArgumentParser(description='PyTorch CIFAR training')
parser.add_argument('--s', type=float, default=0.0001,
                    help='scale sparse rate (default: 0.0001)')
parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--test-batch-size', type=int, default=256, metavar='N',
                    help='input batch size for testing (default: 256)')
parser.add_argument('--epochs', type=int, default=1, metavar='N',
                    help='number of epochs to train (default: 160)')
parser.add_argument('--lr', type=float, default=0.1, metavar='LR',
                    help='learning rate (default: 0.1)')
parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                    help='SGD momentum (default: 0.9)')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--save', default='./logs', type=str, metavar='PATH',
                    help='path to save prune model (default: current directory)')
parser.add_argument('--depth', default=16, type=int,
                    help='depth of the neural network')
parser.add_argument('--sparsity_type', type=str, default='column',
                    help="define sparsity_type: [irregular,column,filter]")
parser.add_argument('--config_file', type=str, default='config_vgg16',
                    help="define sparsity_type: [irregular,column,filter]")

args = parser.parse_args()

args.cuda = not args.no_cuda and torch.cuda.is_available()


torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

from time import time
print(time())
random.seed(int(time()))
torch.manual_seed(int(time()))
torch.cuda.manual_seed(int(time()))

if not os.path.exists(args.save):
    os.makedirs(args.save)

kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}
use_cuda = not args.no_cuda and torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")


test_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10('./data.cifar10', train=False, download=True, transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
                       ])),
        batch_size=args.test_batch_size, shuffle=True, **kwargs)


def test(args, model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.max(1, keepdim=True)[1]  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    accuracy = 100. * float(correct) / float(len(test_loader.dataset))

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.4f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset), accuracy))

    return accuracy


def main():

    print("\n------------------------------\n")

    from models.resnet32_cifar10_grasp import resnet32
    from models.vgg_grasp import vgg19, vgg16
    from models.resnet20_cifar import resnet20

    model = resnet20(dataset="cifar10")
    # model = resnet32()
    # model = vgg16()
    # model = vgg19()

    model.load_state_dict(torch.load("./checkpoints/resnet20/prune_iter/cifar10/round_11_sp0.914/seed914_64_lr_0.01_resnet20_cifar10_acc_88.240_sgd_lr0.01_default_sp0.913_epoch156.pt"))

    model = model.cuda()

    for name, weight in model.named_parameters():
        if (len(weight.size()) == 4): # and "shortcut" not in name):
            print(name, weight.shape)

    test_sparsity(model, column=False, channel=True, filter=True, kernel=False)

    accu = test(args, model, device, test_loader)


if __name__ == '__main__':
    main()
