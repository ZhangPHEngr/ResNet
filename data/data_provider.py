# -*- coding: UTF-8 -*-
"""
@Project ：  ResNet 
@Author:     Zhang P.H
@Date ：     4/4/22 5:07 PM
"""
import cv2
import torchvision.datasets as datasets
import torchvision.transforms as transforms

train_dataset = datasets.MNIST("/media/zhangph/Elements1/dataset/", train=True,
                               transform=transforms.Compose([
                                   transforms.Resize(256),
                                   transforms.CenterCrop(224),
                                   transforms.ToTensor(),
                               ]))

val_dataset = datasets.MNIST("/media/zhangph/Elements1/dataset/", train=False,
                               transform=transforms.Compose([
                                   transforms.Resize(256),
                                   transforms.CenterCrop(224),
                                   transforms.ToTensor(),
                               ]))

if __name__ == '__main__':
    print(train_dataset[0][0].shape)
    img = train_dataset[0][0].numpy().transpose(1,2,0)
    cv2.imshow("", img)
    cv2.waitKey(10)
    cv2.imwrite("sample/test.jpg", img)  # 0 1 二值图像，所以保存也看不出来 是数字5
