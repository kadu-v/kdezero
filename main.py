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
import os
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
max_epoch = 5
batch_size = 100
hidden_size = 1000

train_set = kdezero.datasets.MNIST(train=True)
test_set = kdezero.datasets.MNIST(train=True)
train_loader = DataLoader(train_set, batch_size)
test_loader = DataLoader(test_set, batch_size)


model = MLP((hidden_size, hidden_size, 10), activation=F.relu)
optimizer = optimizers.SGD().setup(model)

if os.path.exists('my_mlp.npz'):
    model.load_weights('my_mlp.npz')

if kdezero.cuda.gpu_enable:
    train_loader.to_gpu()
    model.to_gpu()


train_loss_history = []
train_acc_history = []
test_loss_history = []
test_acc_history = []


for epoch in range(max_epoch):
    sum_loss, sum_acc = 0, 0
    for x, t in train_loader:
        y = model(x)
        loss = F.softmax_cross_entropy(y, t)
        acc = F.accuracy(y, t)
        model.cleargrads()
        loss.backward()
        optimizer.update()

        sum_loss += float(loss.data) * len(t)
        sum_acc += float(acc.data) * len(t)
    train_loss_history.append(sum_loss / len(train_set))
    train_acc_history.append(sum_acc / len(train_set))
    print('epoch: {}'.format(epoch + 1))
    print('train loss: {:.4f}, ecc: {:.4f}'.format(
        sum_loss / len(train_set), sum_acc / len(train_set)))

    sum_loss, sum_acc = 0, 0
    with no_grad():
        for x, t in test_loader:
            y = model(x)
            loss = F.softmax_cross_entropy(y, t)
            acc = F.accuracy(y, t)
            sum_loss += float(loss.data) * len(t)
            sum_acc += float(acc.data) * len(t)
    test_loss_history.append(sum_loss / len(test_set))
    test_acc_history.append(sum_acc / len(test_set))
    print('test loss: {:.4f}, acc: {:.4f}'.format(
        sum_loss / len(test_set), sum_acc / len(test_set)))

model.save_weights('my_mlp.npz')

# %%
# 結果の可視化
plt.plot(np.arange(max_epoch), train_loss_history)
plt.plot(np.arange(max_epoch), test_loss_history)
plt.show()

plt.plot(np.arange(max_epoch), train_acc_history)
plt.plot(np.arange(max_epoch), test_acc_history)
plt.show()
