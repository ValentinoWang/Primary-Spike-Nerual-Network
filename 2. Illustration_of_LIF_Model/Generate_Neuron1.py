import torch
import torch.nn as nn
import numpy as np
from spikingjelly.clock_driven import neuron
from spikingjelly import visualizing
from matplotlib import pyplot as plt

lif = neuron.LIFNode(tau=50.0, decay_input = True, v_threshold = 1.5, v_reset = -0.5)
T = 1000
s_list = []
v_list = []
for t in range(T):
    x = 6.5*(torch.rand(size=[1]))
    s_list.append(lif(x))
    v_list.append(lif.v)

visualizing.plot_one_neuron_v_s(np.asarray(v_list), np.asarray(s_list), 
v_threshold=lif.v_threshold, v_reset=lif.v_reset, dpi=1024)
plt.show()