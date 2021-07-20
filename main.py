# %%
from kdezero.cuda import get_array_module
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
import os
# %%

train_set = kdezero.datasets.MNIST(train=True, transforms=None)
test_set = kdezero.datasets.MNIST(train=True, transforms=None)

print(len(train_set))
print(len(test_set))


# %%
def get_conv_outsuze(input_size, kernel_size, stride, pad):
    return (input_size + pad * 2 - kernel_size) // stride + 1


H, W = 4, 4
KH, KW = 3, 3
SH, SW = 1, 1
PH, PW = 1, 1
OH = get_conv_outsuze(H, KH, SH, PH)
OW = get_conv_outsuze(W, KW, SW, PW)
print(OH, OW)

# %%
