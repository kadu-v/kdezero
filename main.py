# %%
from numpy import random
from kdezero import Variable, Function, using_config, no_grad
import kdezero.functions as F
from kdezero.utils import _dot_func, _dot_var, plot_dot_graph
import numpy as np


x = Variable(np.array(np.pi / 4))
y = F.sin(x)
y.backward()

print(y)
print(x.grad)
