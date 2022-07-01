import torch
import numpy as np
from spikingjelly.clock_driven import neuron
from spikingjelly import visualizing
from matplotlib import pyplot as plt

lif = neuron.LIFNode(tau=50.0, decay_input = True, v_threshold = 1.5, v_reset = -0.5)
T = 100
s_list = []
v_list = []
N = 1024
n = int(N ** 0.5)
for t in range(T):
    if t % 3 == 0:
        x = t/3*(torch.rand(size=[N]))
    elif t % 2 == 1:
        x = (100-t)/4*(torch.rand(size=[N]))
    else:
        x = 7*(torch.rand(size=[N]))
    s_list.append(lif(x).unsqueeze(0))
    v_list.append(lif.v.unsqueeze(0))
    

s_list = torch.cat(s_list)
v_list = torch.cat(v_list)

visualizing.plot_2d_heatmap(array=np.asarray(s_list), title='Spike', xlabel='Time',
                            ylabel='Neuron Index', int_x_ticks=True, x_max=T, dpi=1024)
plt.show()

Spike = torch.Tensor(n,n)
s_list = s_list.view([100,n,n])
for t in range(T):
    Spike += s_list[t,:,:]
min_ = Spike.min()
max_ = Spike.max()
S = (Spike - min_) / (max_ - min_)
S = S.view([1,n,n])

visualizing.plot_2d_spiking_feature_map(S.numpy(), 1, 1, 10 , title='Spiking Feature Maps', dpi=100)
plt.show()