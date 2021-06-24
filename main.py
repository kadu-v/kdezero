# %%
from numpy import random
from kdezero import Variable, Function, using_config, no_grad
import kdezero.functions as F
from kdezero.utils import _dot_func, _dot_var, plot_dot_graph
import numpy as np


def f(x):
    y = x ** 4 - 2 * x ** 2
    return y


x = Variable(np.array(2.0))
y = f(x)
y.backward(create_graph=True)
print(x.grad)


gx = x.grad
x.cleargrad()
gx.backward()
print(x.grad)
# %%
iters = 10

for i in range(iters):
    print(i, x)

    y = f(x)
    x.cleargrad()
    y.backward(create_graph=True)

    gx = x.grad
    x.cleargrad()
    gx.backward()
    gx2 = x.grad

    x.data -= gx.data / gx2.data
