import os
import argparse
import shutil
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable

import time

from models.resnet32_cifar10_grasp import resnet32
from models.vgg_grasp import vgg19, vgg16
from models.resnet20_cifar import resnet20
from torch.optim.lr_scheduler import _LRScheduler

from testers import *

import numpy as np

import random
from prune_utils import *
# from prune_utils.admm import ADMM


# Training settings
parser = argparse.ArgumentParser(description='PyTorch CIFAR training')
parser.add_argument('--arch', type=str, default=None,
                    help='[vgg, resnet, convnet, alexnet]')
parser.add_argument('--depth', default=None, type=int,
                    help='depth of the neural network, 16,19 for vgg; 18, 50 for resnet')
parser.add_argument('--dataset', type=str, default="cifar10",
                    help='[cifar10, cifar100]')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--multi-gpu', action='store_true', default=False,
                    help='for multi-gpu training')
parser.add_argument('--s', type=float, default=0.0001,
                    help='scale sparse rate (default: 0.0001)')
parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--test-batch-size', type=int, default=256, metavar='N',
                    help='input batch size for testing (default: 256)')
parser.add_argument('--epochs', type=int, default=160, metavar='N',
                    help='number of epochs to train (default: 160)')
parser.add_argument('--optmzr', type=str, default='adam', metavar='OPTMZR',
                    help='optimizer used (default: adam)')
parser.add_argument('--lr', type=float, default=0.1, metavar='LR',
                    help='learning rate (default: 0.001)')
parser.add_argument('--lr-decay', type=int, default=60, metavar='LR_decay',
                    help='how many every epoch before lr drop (default: 30)')
parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                    help='SGD momentum (default: 0.9)')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')

parser.add_argument('--lr-scheduler', type=str, default='default',
                    help='define lr scheduler')
parser.add_argument('--warmup', action='store_true', default=False,
                    help='warm-up scheduler')
parser.add_argument('--warmup-lr', type=float, default=0.0001, metavar='M',
                    help='warmup-lr, smaller than original lr')
parser.add_argument('--warmup-epochs', type=int, default=0, metavar='M',
                    help='number of epochs for lr warmup')
parser.add_argument('--mixup', action='store_true', default=False,
                    help='ce mixup')
parser.add_argument('--alpha', type=float, default=0.3, metavar='M',
                    help='for mixup training, lambda = Beta(alpha, alpha) distribution. Set to 0.0 to disable')
parser.add_argument('--smooth', action='store_true', default=False,
                    help='lable smooth')
parser.add_argument('--smooth-eps', type=float, default=0.0, metavar='M',
                    help='smoothing rate [0.0, 1.0], set to 0.0 to disable')

parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--rho', type=float, default = 0.0001,
                    help ="Just for initialization")
parser.add_argument('--pretrain-epochs', type=int, default=0, metavar='M',
                    help='number of epochs for pretrain')
parser.add_argument('--pruning-epochs', type=int, default=0, metavar='M',
                    help='number of epochs for pruning')
parser.add_argument('--remark', type=str, default=None,
                    help='optimizer used (default: adam)')
parser.add_argument('--save-model', type=str, default='model/',
                    help='optimizer used (default: adam)')
parser.add_argument('--sparsity-type', type=str, default='random-pattern',
                    help="define sparsity_type: [irregular,column,filter,pattern]")
parser.add_argument('--config-file', type=str, default='config_vgg16',
                    help="config file name")

parser.add_argument('--patternNum', type=int, default=8, metavar='M',
                    help='number of epochs for lr warmup')
parser.add_argument('--rand-seed', action='store_true', default=False,
                    help='use random seed')
parser.add_argument("--log-filename", default=None, type=str, help='log filename, will override self naming')
parser.add_argument("--resume", default=None, type=str, metavar="PATH", help="path to latest checkpoint (default: none)")
parser.add_argument('--save-mask-model', action='store_true', default=False, help='save a sparse model indicating pruning mask')
parser.add_argument('--mask-sparsity', type=str, default=None, help='dir and file name for mask models')

prune_parse_arguments(parser)
args = parser.parse_args()

args.cuda = not args.no_cuda and torch.cuda.is_available()

if args.rand_seed:
    seed = random.randint(1, 999)
    print("Using random seed:", seed)
else:
    seed = args.seed
    print("Using manual seed:", seed)

torch.manual_seed(seed)
np.random.seed(seed)
if args.cuda:
    torch.cuda.manual_seed(seed)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

if not os.path.exists(args.save_model):
    os.makedirs(args.save_model)

if args.save_model[-1] != '/':
    args.save_model += '/'

kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}

if args.dataset == "cifar10":
    train_loader = torch.utils.data.DataLoader(
            datasets.CIFAR10('./data.cifar10', train=True, download=True,
                           transform=transforms.Compose([
                               transforms.Pad(4),
                               transforms.RandomCrop(32),
                               transforms.RandomHorizontalFlip(),
                               transforms.ToTensor(),
                               transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
                           ])),
            batch_size=args.batch_size, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(
            datasets.CIFAR10('./data.cifar10', train=False, transform=transforms.Compose([
                               transforms.ToTensor(),
                               transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
                           ])),
            batch_size=args.test_batch_size, shuffle=True, **kwargs)

elif args.dataset == "cifar100":
    train_loader = torch.utils.data.DataLoader(
            datasets.CIFAR100('./data.cifar100', train=True, download=True,
                            transform=transforms.Compose([
                            transforms.Pad(4),
                            transforms.RandomCrop(32),
                            transforms.RandomHorizontalFlip(),
                            transforms.ToTensor(),
                            transforms.Normalize((0.5070751592371323, 0.48654887331495095, 0.4409178433670343),
                                                 (0.2673342858792401, 0.2564384629170883, 0.27615047132568404))
                            ])),
            batch_size=args.batch_size, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(
            datasets.CIFAR100('./data.cifar100', train=False, transform=transforms.Compose([
                            transforms.ToTensor(),
                            transforms.Normalize((0.5070751592371323, 0.48654887331495095, 0.4409178433670343),
                                                 (0.2673342858792401, 0.2564384629170883, 0.27615047132568404))
                            ])),
            batch_size=args.test_batch_size, shuffle=True, **kwargs)


class CrossEntropyLossMaybeSmooth(nn.CrossEntropyLoss):
    ''' Calculate cross entropy loss, apply label smoothing if needed. '''

    def __init__(self, smooth_eps=0.0):
        super(CrossEntropyLossMaybeSmooth, self).__init__()
        self.smooth_eps = smooth_eps

    def forward(self, output, target, smooth=False):
        if not smooth:
            return F.cross_entropy(output, target)

        target = target.contiguous().view(-1)
        n_class = output.size(1)
        one_hot = torch.zeros_like(output).scatter(1, target.view(-1, 1), 1)
        smooth_one_hot = one_hot * (1 - self.smooth_eps) + (1 - one_hot) * self.smooth_eps / (n_class - 1)
        log_prb = F.log_softmax(output, dim=1)
        loss = -(smooth_one_hot * log_prb).sum(dim=1).mean()
        return loss


def mixup_data(x, y, alpha=1.0):

    '''Compute the mixup data. Return mixed inputs, pairs of targets, and lambda'''
    if alpha > 0.0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.0

    batch_size = x.size()[0]
    index = torch.randperm(batch_size).cuda()

    mixed_x = lam * x + (1 - lam) * x[index,:]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam, smooth):
    return lam * criterion(pred, y_a, smooth=smooth) + \
           (1 - lam) * criterion(pred, y_b, smooth=smooth)


class GradualWarmupScheduler(_LRScheduler):
    """ Gradually warm-up(increasing) learning rate in optimizer.
    Proposed in 'Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour'.
    Args:
        optimizer (Optimizer): Wrapped optimizer.
        multiplier: target learning rate = base lr * multiplier
        total_iter: target learning rate is reached at total_iter, gradually
        after_scheduler: after target_epoch, use this scheduler(eg. ReduceLROnPlateau)
    """

    def __init__(self, optimizer, multiplier, total_iter, after_scheduler=None):
        self.multiplier = multiplier
        if self.multiplier <= 1.:
            raise ValueError('multiplier should be greater than 1.')
        self.total_iter = total_iter
        self.after_scheduler = after_scheduler
        self.finished = False
        super().__init__(optimizer)

    def get_lr(self):
        if self.last_epoch > self.total_iter:
            if self.after_scheduler:
                if not self.finished:
                    self.after_scheduler.base_lrs = [base_lr * self.multiplier for base_lr in self.base_lrs]
                    self.finished = True
                return self.after_scheduler.get_lr()
            return [base_lr * self.multiplier for base_lr in self.base_lrs]

        return [base_lr * ((self.multiplier - 1.) * self.last_epoch / self.total_iter + 1.) for base_lr in self.base_lrs]

    def step(self, epoch=None):
        if self.finished and self.after_scheduler:
            return self.after_scheduler.step(epoch)
        else:
            return super(GradualWarmupScheduler, self).step(epoch)


def train(model, train_loader, criterion, scheduler, optimizer, epoch, maskretrain, masks):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    for i, (input, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        # prune_update_learning_rate(optimizer, epoch, args)

        input = input.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)

        if args.mixup:
            input, target_a, target_b, lam = mixup_data(input, target, args.alpha)

        # compute output
        output = model(input)

        ce_loss = criterion(output, target, smooth=args.smooth)

        # measure accuracy and record loss
        acc1,_ = accuracy(output, target, topk=(1,5))

        losses.update(ce_loss.item(), input.size(0))
        top1.update(acc1[0], input.size(0))

        # compute gradient and do SGD step
        ce_loss.backward()

        prune_apply_masks_on_grads()

        optimizer.step()

        optimizer.zero_grad()

        prune_apply_masks()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % 10 == 0:
            for param_group in optimizer.param_groups:
                current_lr = param_group['lr']

            print('Epoch: [{0}][{1}/{2}]\t'
                  'LR: {3:.5f}\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  .format(
                   epoch, i, len(train_loader),
                   current_lr,
                   batch_time=batch_time,
                   data_time=data_time, loss=losses, top1=top1))


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def test(model):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            if args.cuda:
                data, target = data.cuda(), target.cuda()
            data, target = Variable(data, volatile=True), Variable(target)
            output = model(data)
            criterion = nn.CrossEntropyLoss()
            test_loss = criterion(output, target)
            # test_loss += F.cross_entropy(output, target, size_average=False).data[0] # sum up batch loss
            pred = output.data.max(1, keepdim=True)[1]  # get the index of the max log-probability
            correct += pred.eq(target.data.view_as(pred)).cpu().sum()

    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.3f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        float(100. * correct) / float(len(test_loader.dataset))))
    return (float(100 * correct) / float(len(test_loader.dataset)))


def main():

    if args.cuda:
        if args.arch == "vgg":
            if args.depth == 19:
                model = vgg19(dataset=args.dataset)
            elif args.depth == 16:
                model = vgg16(dataset=args.dataset)
            else:
                sys.exit("vgg doesn't have those depth!")
        elif args.arch == "resnet":
            if args.depth == 20:
                model = resnet20(dataset=args.dataset)
            elif args.depth == 32:
                model = resnet32(depth=32, dataset=args.dataset)
            else:
                sys.exit("resnet doesn't implement those depth!")
        else:
            sys.exit("wrong arch!")

        if args.multi_gpu:
            model = torch.nn.DataParallel(model)
        model.cuda()

    criterion = CrossEntropyLossMaybeSmooth(smooth_eps=args.smooth_eps).cuda()

    optimizer_init_lr = args.warmup_lr if args.warmup else args.lr

    optimizer = None
    if (args.optmzr == 'sgd'):
        optimizer = torch.optim.SGD(model.parameters(), optimizer_init_lr, momentum=0.9, weight_decay=1e-4)
    elif (args.optmzr == 'adam'):
        optimizer = torch.optim.Adam(model.parameters(), optimizer_init_lr)

    scheduler = None
    if args.lr_scheduler == 'cosine':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=4e-08)
    elif args.lr_scheduler == 'default':
        if not args.warmup:
            epoch_milestones = [80, 120]
        else:
            epoch_milestones = [80-args.warmup_epochs, 120-args.warmup_epochs]

        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=epoch_milestones, gamma=0.1)
    else:
        raise Exception("unknown lr scheduler")

    if args.warmup:
        scheduler = GradualWarmupScheduler(optimizer, multiplier=args.lr / args.warmup_lr, total_iter=args.warmup_epochs, after_scheduler=scheduler)

    # ----------- load checkpoint ---------------------
    model_state = None
    current_epoch = 0
    if args.resume is not None:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            if 'state_dict' in checkpoint:
                optimizer.load_state_dict(checkpoint['optimizer'])
                scheduler.load_state_dict(checkpoint['scheduler'])
                model_state = checkpoint['state_dict']
                current_epoch = checkpoint['current_epoch']
            else:
                model_state = checkpoint
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))
            model_state = None
        time.sleep(2)

    if not model_state is None:
        model.load_state_dict(model_state)

    log_filename = args.log_filename
    print(log_filename)

    log_filename_dir_str = log_filename.split('/')
    log_filename_dir = "/".join(log_filename_dir_str[:-1])
    if not os.path.exists(log_filename_dir):
        os.system('mkdir -p ' + log_filename_dir)
        print("New folder {} created...".format(log_filename_dir))

    with open(log_filename, 'a') as f:
        for arg in sorted(vars(args)):
            f.write("{}:".format(arg))
            f.write("{}".format(getattr(args, arg)))
            f.write("\n")

    # ------------- pre training ---------------------
    print("==============pre training=================")

    prune_init(args, model)
    prune_apply_masks()  # if wanted to make sure the mask is applied in retrain
    prune_print_sparsity(model)

    _, total_sparsity = test_sparsity(model, column=False, channel=True, filter=True, kernel=True)

    all_acc = [0.000]
    epoch_cnt = current_epoch
    best_file_name = []
    for epoch in range(current_epoch, args.epochs):
        prune_update(epoch)
        train(model, train_loader, criterion, scheduler, optimizer, epoch, maskretrain=False, masks={})
        prec1 = test(model)

        scheduler.step()

        epoch_cnt += 1
        prune_print_sparsity(model)

        log_line = "Epoch:{}, Test_accu:[{}] LR:{}\n".format(epoch, prec1, optimizer.param_groups[0]['lr'])

        with open(log_filename, 'a') as f:
            f.write(log_line)
            f.write("\n")

        if prec1 > max(all_acc):
            print("\n>_ Got better accuracy, saving model with accuracy {:.3f}% now...\n".format(prec1))
            filename = "./{}seed{}_{}_{}{}_{}_acc_{:.3f}_{}_lr{}_{}_sp{:.3f}_epoch{}.pt".format(args.save_model,
                                                                                            seed, args.remark,
                                                                                            args.arch,
                                                                                            args.depth,
                                                                                            args.dataset,
                                                                                            prec1,
                                                                                            args.optmzr, args.lr,
                                                                                            args.lr_scheduler,
                                                                                            total_sparsity,
                                                                                            epoch_cnt)
            torch.save(model.state_dict(), filename)
            best_file_name.append(filename)
            print("\n>_ Deleting previous model file with accuracy {:.3f}% now...\n".format(max(all_acc)))
            if len(all_acc) > 1:
                os.remove(
                    "./{}seed{}_{}_{}{}_{}_acc_{:.3f}_{}_lr{}_{}_sp{:.3f}_epoch{}.pt".format(args.save_model,
                                                                                         seed, args.remark,
                                                                                         args.arch,
                                                                                         args.depth,
                                                                                         args.dataset,
                                                                                         max(all_acc),
                                                                                         args.optmzr, args.lr,
                                                                                         args.lr_scheduler,
                                                                                         total_sparsity,
                                                                                         best_epoch))
            best_epoch = epoch_cnt

        all_acc.append(prec1)



if __name__ == '__main__':
    main()