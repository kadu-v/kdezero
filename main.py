# %%
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


np.random.seed(0)

# ハイパラメータの設定
max_epoch = 300
batch_size = 30
hidden_size = 10
lr = 1.0


train_set = kdezero.datasets.Spiral(train=True)
test_set = kdezero.datasets.Spiral(train=False)

train_loader = DataLoader(train_set, batch_size)
test_loader = DataLoader(test_set, batch_size, shuffle=False)
# %%


# モデルの作成
model = MLP((hidden_size, 3))
optimizer = optimizers.SGD(lr).setup(model)

data_size = len(train_set)
max_iter = math.ceil(data_size / batch_size)
train_loss_history = []
train_acc_history = []
test_loss_history = []
test_acc_history = []

for epoch in range(max_epoch):
    index = np.random.permutation(data_size)
    sum_loss = 0
    sum_acc = 0

    for x, t in train_loader:

        y = model(x)
        loss = F.softmax_cross_entropy(y, t)
        acc = F.accuracy(y, t)
        model.cleargrads()
        loss.backward()
        optimizer.update()

        sum_loss += float(loss.data) * len(t)
        sum_acc += float(acc.data) * len(t)

    avg_loss = sum_loss / len(train_set)
    avg_acc = sum_acc / len(train_set)
    train_loss_history.append(avg_loss)
    train_acc_history.append(avg_acc)
    print('epoch {}'.format(epoch + 1))
    print('train loss: {:.4f}, acc: {:.4f}'.format(
        avg_loss, avg_acc))

    sum_loss, sum_acc = 0, 0
    with kdezero.no_grad():
        for x, t in test_loader:
            y = model(x)
            loss = F.softmax_cross_entropy(y, t)
            acc = F.accuracy(y, t)
            sum_loss += float(loss.data) * len(t)
            sum_acc += float(acc.data) * len(t)

        avg_loss = sum_loss / len(test_set)
        avg_acc = sum_acc / len(test_set)
        test_loss_history.append(avg_loss)
        test_acc_history.append(avg_acc)
        print('test loss: {:.4f}, acc: {:.4f}'.format(
            avg_loss, avg_acc))

# %%
plt.plot(np.array(range(max_epoch)), train_loss_history)
plt.plot(np.array(range(max_epoch)), test_loss_history, color='r')
plt.show()

# %%
plt.plot(np.array(range(max_epoch)), train_acc_history)
plt.plot(np.array(range(max_epoch)), test_acc_history, color='r')
plt.show()

# %%
# 予測結果の可視化
x, t = kdezero.datasets.get_spiral(train=True)
print(x.shape)
print(t.shape)
print(x[10], t[10])
print(x[110], t[110])


h = 0.001
x_min, x_max = x[:, 0].min() - .1, x[:, 0].max() + .1
y_min, y_max = x[:, 1].min() - .1, x[:, 1].max() + .1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
X = np.c_[xx.ravel(), yy.ravel()]

with kdezero.no_grad():
    score = model(X)
predict_cls = np.argmax(score.data, axis=1)
Z = predict_cls.reshape(xx.shape)
plt.contourf(xx, yy, Z)

# Plot data points of the dataset
N, CLS_NUM = 100, 3
markers = ['o', 'x', '^']
colors = ['orange', 'blue', 'green']
for i in range(len(x)):
    c = t[i]
    plt.scatter(x[i][0], x[i][1], s=40, marker=markers[c], c=colors[c])
plt.show()
# %%
