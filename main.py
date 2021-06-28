# %%
from numpy import random
from kdezero import Variable, Function, using_config, no_grad
import kdezero.functions as F
from kdezero.utils import _dot_func, _dot_var, plot_dot_graph
import numpy as np
import matplotlib.pyplot as plt

x = Variable(np.array([1, 2, 3, 4, 5, 6]))
y = F.sum(x)
y.backward()
print(y)
print(x.grad)
