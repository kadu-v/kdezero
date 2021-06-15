# %%
from numpy import random
from kdezero import Variable, using_config, no_grad
from kdezero.utils import _dot_func, _dot_var, plot_dot_graph
import numpy as np


def sphere(x, y):
    z = x ** 2 + y ** 2
    return z


def matyas(x, y):
    z = 0.26 * (x ** 2 + y ** 2) - 0.48 * x * y
    return z


def goldstein(x, y):
    z = (1 + (x + y + 1)**2 * (19 - 14 * x + 3 * x**2 - 14 * y + 6 * x * y + 3 * y ** 2)) * \
        (30 + (2 * x - 3 * y)**2 * (18 - 32 * x + 12 *
                                    x ** 2 + 48 * y - 36 * x * y + 27 * y ** 2))
    return z


x = Variable(np.array(1.0))
y = Variable(np.array(1.0))
z = sphere(x, y)
z.backward()
print(z)
print(x.grad)
print(y.grad)


x.cleargrad()
y.cleargrad()
z = matyas(x, y)
z.backward()
print(z)
print(x.grad)
print(y.grad)

x.cleargrad()
y.cleargrad()
z = goldstein(x, y)
z.backward()
print(z)
print(x.grad)
print(y.grad)

# %%
x = Variable(np.random.randn(2))
y = Variable(np.random.randn(2))
x.name = 'x'
print(_dot_var(x))
print(_dot_var(x, verbose=True))

z = x + y
txt = _dot_func(z.creator)
print(txt)
# %%
x = Variable(np.array(1.0))
y = Variable(np.array(2))
z = goldstein(x, y)
z.backward()
x.name = 'x'
y.name = 'y'
z.name = 'z'

plot_dot_graph(z, verbose=False, to_file='goldstein.png')

# %%
