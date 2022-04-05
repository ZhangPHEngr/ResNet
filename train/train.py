# -*- coding: UTF-8 -*-
"""
@Project ：  ResNet
@Author:     Zhang P.H
@Date ：     4/4/22 4:59 PM
"""
import argparse
import random
import shutil
import time
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
from torch.optim.lr_scheduler import CosineAnnealingLR

from model import ResNet18
from data_provider import train_dataset, val_dataset

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
# 训练配置
parser.add_argument('--epochs', default=10, type=int, metavar='N', help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N', help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=16, type=int, metavar='N', help='mini-batch size (default: 256), this is the total '
                                                                                 'batch size of all GPUs on the current node when '
                                                                                 'using Data Parallel or Distributed Data Parallel')
# 优化器设置
parser.add_argument('--lr', '--learning-rate', default=0.01, type=float, metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M', help='momentum')
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float, metavar='W', help='weight decay (default: 1e-4)', dest='weight_decay')
# 其他设置
parser.add_argument('-p', '--print-freq', default=1, type=int, metavar='N', help='print frequency (default: 10)')
parser.add_argument('--seed', default=0, type=int, metavar='N', help='print frequency (default: 10)')
parser.add_argument('--model_path', default=None, type=str, help='loading pretrained model')

best_acc1 = 0

SAMPLES = 100


def main():
    # ------------------------------------------训练设置------------------------------
    args = parser.parse_args()
    # 确定化随机值， 便于复现
    if args.seed is not None:
        random.seed(args.seed)  # 设置random随机种子
        torch.manual_seed(args.seed)  # 设置pytorch CPU随机种子
        torch.cuda.manual_seed(args.seed)  # 设置当前GPU的随机种子
        torch.cuda.manual_seed_all(args.seed)  # 设置所有GPU的随机种子
        cudnn.deterministic = True  # CuDNN卷积使用确定性算法, 每次卷积都一种

    # 判断GPU能否使用，可以加速训练
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ------------------------------------------网络设置------------------------------
    # 配置数据
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=8, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=8, pin_memory=True)

    # 网络载入
    model = ResNet18(in_channel=1, num_classes=10)
    if args.model_path is not None:
        model.load_state_dict(torch.load(args.model_path))
    model = model.to(device)

    # 设置loss
    criterion = nn.CrossEntropyLoss().to(device)

    # 设置优化器
    optimizer = torch.optim.SGD(model.parameters(), args.lr, momentum=args.momentum)
    lr_scheduler = CosineAnnealingLR(optimizer=optimizer, T_max=args.epochs)

    # # optionally resume from a checkpoint
    # if args.resume:
    #     if os.path.isfile(args.resume):
    #         print("=> loading checkpoint '{}'".format(args.resume))
    #         if args.gpu is None:
    #             checkpoint = torch.load(args.resume)
    #         else:
    #             # Map model to be loaded to specified single gpu.
    #             loc = 'cuda:{}'.format(args.gpu)
    #             checkpoint = torch.load(args.resume, map_location=loc)
    #         args.start_epoch = checkpoint['epoch']
    #         best_acc1 = checkpoint['best_acc1']
    #         if args.gpu is not None:
    #             # best_acc1 may be from a checkpoint from a different GPU
    #             best_acc1 = best_acc1.to(args.gpu)
    #         model.load_state_dict(checkpoint['state_dict'])
    #         optimizer.load_state_dict(checkpoint['optimizer'])
    #         lr_scheduler.load_state_dict(checkpoint['scheduler'])
    #         print("=> loaded checkpoint '{}' (epoch {})"
    #               .format(args.resume, checkpoint['epoch']))
    #     else:
    #         print("=> no checkpoint found at '{}'".format(args.resume))

    # cudnn.benchmark = True

    for epoch in range(args.start_epoch, args.epochs):
        print("正在进行第{}个epoch".format(epoch))
        # train for one epoch
        train(train_loader, model, criterion, optimizer, lr_scheduler, device)
        # evaluate on validation set
        validate(val_loader, model, criterion, device)
        # remember best acc@1 and save checkpoint
        torch.save(model.state_dict(), "pth/{}.pth".format(epoch))


def train(train_loader, model, criterion, optimizer, lr_scheduler, device):
    # switch to train mode
    model.train()
    bar = tqdm(train_loader)
    for i, (images, target) in enumerate(bar):
        # copy data
        if torch.cuda.is_available():
            images = images.cuda(device)
            target = target.cuda(device)

        # compute output
        output = model(images)
        loss = criterion(output, target)

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        lr_scheduler.step()

        if i % SAMPLES == 0:
            # 计算精度
            prediction = torch.argmax(output, 1)
            acc = (prediction == target).sum().float()/target.shape[0]
            # print(output, target, prediction,acc)
            # 打印输出
            bar.set_description("完成训练：{}".format(i))
            print("loss:", float(loss.to("cpu")), "acc:",float(acc.to("cpu")))


def validate(val_loader, model, criterion, device):
    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        avg_loss = 0
        avg_acc = 0
        bar = tqdm(val_loader)
        for i, (images, target) in enumerate(bar):
            bar.set_description("正在验证：{}".format(i))
            if torch.cuda.is_available():
                images = images.cuda(device)
                target = target.cuda(device)

            # compute output
            output = model(images)
            prediction = torch.argmax(output, 1)
            acc = (prediction == target).sum().float() / target.shape[0]

            avg_loss += criterion(output, target)
            avg_acc += acc

        avg_loss /= i
        avg_acc /= i
        # 打印输出
        print("验证结果： avg_loss:", avg_loss, "avg_acc:", avg_acc)


def save_checkpoint(state, is_best, filename='repvgg_model_best.pth.tar'):
    torch.save(state, filename)
    if is_best:
        # shutil.copyfile(filename, 'resnet18_model_best.pth.tar')
        shutil.copyfile(filename, 'model_best.pth.tar')


if __name__ == '__main__':
    main()
