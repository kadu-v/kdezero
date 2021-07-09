# %%
from kdezero import transforms
from kdezero.dataloders import DataLoader
from matplotlib import colors
from numpy import random
from numpy.core.fromnumeric import argmax
import kdezero
from kdezero import Variable, Function, using_config, no_grad
import kdezero.functions as F
import kdezero.layers as L
from kdezero.utils import _dot_func, _dot_var, plot_dot_graph
from kdezero.models import Model, MLP
from kdezero import optimizers
import numpy as np
import matplotlib.pyplot as plt
import math
# %%

train_set = kdezero.datasets.MNIST(train=True, transforms=None)
test_set = kdezero.datasets.MNIST(train=True, transforms=None)

print(len(train_set))
print(len(test_set))

# %%
x, t = train_set[0]
print(type(x), x.shape)
print(t)

# %%
# データの表示
plt.imshow(x.reshape(28, 28), cmap='gray')
plt.axis('off')
plt.show()
print('lable:', t)

# %%
