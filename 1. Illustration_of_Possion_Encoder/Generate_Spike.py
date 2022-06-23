from ast import increment_lineno
from math import floor
import torch
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from PIL import Image
from spikingjelly.clock_driven import encoding
from spikingjelly import visualizing

# 选择Lamborghini_egoista图片
lena_img = np.array(Image.open('Lamborghini_egoista.jpg')) / 255
x = torch.from_numpy(lena_img)

PE = encoding.PoissonEncoder()

# 仿真50个时间步长，将图像编码为脉冲矩阵并输出
w, h, c= x.shape
out_spike = torch.full((50, w, h, c), 0, dtype=torch.bool)
T = 50
for t in range(T):
    out_spike[t] = PE(x)

# 每隔10个步长选择一个脉冲矩阵并输出
N = floor(T/10)
ind = torch.Tensor(N)
for i in range(N):
    ind[i] = i
ind = torch.tensor(ind,dtype=torch.int)

plt_out_spike = torch.index_select(out_spike, 0, ind)

plt.figure()
plt.imshow(x, cmap='gray')
plt.axis('off')

visualizing.plot_2d_spiking_feature_map(plt_out_spike[:,:,:,0].float().numpy(), 1, 5, 30, 'Red-PoissonEncoder')
plt.axis('off')
plt.show()

visualizing.plot_2d_spiking_feature_map(plt_out_spike[:,:,:,1].float().numpy(), 1, 5, 30, 'Green-PoissonEncoder')
plt.axis('off')
plt.show()

visualizing.plot_2d_spiking_feature_map(plt_out_spike[:,:,:,2].float().numpy(), 1, 5, 30, 'Blue-PoissonEncoder')
plt.axis('off')
plt.show()