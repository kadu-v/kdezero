# %%
from kdezero.cuda import get_array_module
from kdezero import transforms
from kdezero.dataloders import DataLoader
from matplotlib import colors
from numpy import random
from numpy.core.fromnumeric import argmax, reshape
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
from kdezero.transforms import Compose, Flatten, Normalize, ToFloat


# %%


class CNN(Model):
    def __init__(self):
        super().__init__()
        self.conv1 = L.Conv2d(1, 3)
        self.conv2 = L.Conv2d(1, 3)
        self.fc1 = L.Linear(10)
        self.fc2 = L.Linear(10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = F.pooling(x, 3)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.pooling(x, 3)
        x = F.reshape(x, (x.shape[0], -1))
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        return x


# %%
max_epoch = 3
batch_size = 100

train_set = kdezero.datasets.MNIST(train=True, transform=Compose([
    ToFloat(), Normalize(0., 255.)]))
train_loader = DataLoader(train_set, batch_size)
model = CNN()
optimizer = optimizers.SGD().setup(model)


for epoch in range(max_epoch):
    sum_loss = 0

    for x, t in train_loader:
        # print(x.shape)
        y = model(x)
        loss = F.softmax_cross_entropy(y, t)
        model.cleargrads()
        loss.backward()
        optimizer.update()
        sum_loss += float(loss.data) * len(t)

    print('epoch: {}, loss: {:.4f}'.format(
        epoch + 1, sum_loss / len(train_set)))

model.save_weights('my_mlp.npz')

# %%
