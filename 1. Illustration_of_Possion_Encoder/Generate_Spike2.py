from ast import increment_lineno
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
w, h, c= x.shape

superposition = torch.full((w, h, c), 0, dtype=torch.float)
superposition_ = torch.full((5, w, h, c), 0, dtype=torch.float)
T = 512
for t in range(T):
    superposition += PE(x).float()
    if t == 0 or t == 127 or t == 255 or t == 383 or t == 511:
        superposition_[int((t + 1) / 128)] = superposition

# 归一化
for i in range(5):
    min_ = superposition_[i].min()
    max_ = superposition_[i].max()
    superposition_[i] = (superposition_[i] - min_) / (max_ - min_)
Encoder_Matics = superposition_[:,:,:,0]
    
# 画图
visualizing.plot_2d_spiking_feature_map(Encoder_Matics.numpy(), 1, 5, 30,'PossionEncoder')
plt.axis('off')

plt.show()