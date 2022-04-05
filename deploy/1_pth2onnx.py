# -*- coding: UTF-8 -*-
"""
@Project ：  ResNet 
@Author:     Zhang P.H
@Date ：     4/4/22 7:47 PM
"""
import torch
from train.model import ResNet18
pth_path = "../train/pth/4.pth"

# 构造输入， 和训练时一样
x = torch.randn(16, 1, 224, 224)
# 载入空白模型
net = ResNet18(in_channel=1, num_classes=10)
# 载入预训练权重, 这里可能会涉及CPU模型和GPU模型参数的问题
net.load_state_dict(torch.load(pth_path))
# 输入模型
torch.onnx.export(net, x, "resnet_18.onnx", export_params=True, verbose=True, opset_version=12)


if __name__ == '__main__':
    pass
