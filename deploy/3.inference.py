# -*- coding: UTF-8 -*-
"""
@Project ：  resnet
@Author:     Zhang P.H
@Date ：     4/5/22 10:40 PM
"""
import cv2
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np

VIS = False

# 0.预先定义，与训练时一样
BATCH_SIZE = 16
img = cv2.imread("../data/sample/test.jpg")
print("origin img size:", img.shape)
if VIS:
    img_show = img / img.max() * 255
    cv2.imshow(" ", img_show)
    cv2.waitKey(-1)

# 创建输入输出
img = img.transpose(2, 0, 1)
h_input = np.array(np.repeat(np.expand_dims(img, axis=0), BATCH_SIZE, axis=0), dtype=np.float32)
print("batch input imgs size:", h_input.shape)
h_output = np.empty([BATCH_SIZE, 10], dtype=np.float32)
print("batch output class size:", h_output.shape)

# 2.载入trt创建trt engine
f = open("../data/trt/resnet_18.trt", "rb")
runtime = trt.Runtime(trt.Logger(trt.Logger.WARNING))
engine = runtime.deserialize_cuda_engine(f.read())
context = engine.create_execution_context()

# 3.allocate device memory
print("分配显存： input:{} bytes  output:{} bytes".format(h_input.nbytes, h_output.nbytes))
d_input = cuda.mem_alloc(1 * h_input.nbytes)
d_output = cuda.mem_alloc(1 * h_output.nbytes)
bindings = [int(d_input), int(d_output)]  # 获取显存地址

# 4.完成推理
stream = cuda.Stream()
# transfer input data to device
cuda.memcpy_htod_async(d_input, h_input, stream)
# execute model
context.execute_async_v2(bindings, stream.handle, None)
# transfer predictions back
cuda.memcpy_dtoh_async(h_output, d_output, stream)
# syncronize threads
stream.synchronize()

# 5.输出结果
print(np.argmax(h_output, axis=1))

if __name__ == '__main__':
    pass
