# %%
from numpy import random
from kdezero import Variable, Function, using_config, no_grad
import kdezero.functions as F
from kdezero.utils import _dot_func, _dot_var, plot_dot_graph
import numpy as np
import matplotlib.pyplot as plt

x = Variable(np.array(1.0))
y = F.tanh(x)
x.name = 'x'
y.name = 'y'
y.backward(create_graph=True)

iters = 3
for i in range(iters):
    gx = x.grad
    x.cleargrad()
    gx.backward(create_graph=True)


gx = x.grad
gx.name = 'gx' + str(iters + 1)
plot_dot_graph(gx, verbose=False, to_file='tanh.png')
