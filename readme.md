# 1.Train Model
```shell
# 直接运行train.py程序即可完成Resnet训练任务，会把训练结果保存在data/pth中
cd train
python3 train.py
```
训练代码是参考https://www.bilibili.com/video/BV1Xw411f7FW 中讲解的ResNet框架。

这里没用ImageNet训练，而是使用了最简单的MNIST手写数字体数据集做RestNet18的演示demo，训练4个epoch基本就能达到98%的正确率。
# 2.Deploy Model
部署代码采用的思路是： 

pytorch  --> pth file --> onnx file --> trt file --> TensorRT Engine(Python/C++)
```shell
# onnx 转 tensorRT file指令
trtexec --onnx=path-to-onnx-model/xxx.onnx --saveEngine=path-to-save_trt_model/xxx.trt
```
这里面最难的是：配置环境 和 TensorRT使用

配置环境请参考：
- ubuntu 安装CUDA和CuDNN https://zhuanlan.zhihu.com/p/72298520
- ubuntu 安装 TensorRT https://blog.csdn.net/zong596568821xp/article/details/86077553
- 以上三个NVIDIA相关模块安装建议多搜网上同类型显卡安装配置教程，且尽可能不用deb包安装坑比较多
- ubuntu 安装 TensorRT 官方教程最全面 【强烈推荐】https://docs.nvidia.com/deeplearning/tensorrt/archives/tensorrt-840-ea/install-guide/index.html
- cmakelists.txt调用CUDA使用建议 https://www.cnblogs.com/binbinjx/p/5626916.html https://blog.csdn.net/afei__/article/details/81201039
- TensorRT安装成功后，其安装目录下有若干pdf和sample，可以帮助学习TensorRT

# 3.进展更新

- 2022.04.06 ubuntu18.04+RTX3080 配置好以上NVIDIA环境，并完成训练代码和Python部署推理代码，C++学了下TensorRT样例