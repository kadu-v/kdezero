# %%
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
train_set = kdezero.datasets.Spiral()
# %%
# ハイパラメータの設定
max_epoch = 300
batch_size = 30
hidden_size = 10
lr = 1.0

# モデルの作成
model = MLP((hidden_size, 3))
optimizer = optimizers.SGD(lr).setup(model)

data_size = len(train_set)
max_iter = math.ceil(data_size / batch_size)
history = []

for epoch in range(max_epoch):
    index = np.random.permutation(data_size)
    sum_loss = 0

    for i in range(max_iter):
        batch_index = index[i * batch_size: (i + 1) * batch_size]
        batch = [train_set[i] for i in batch_index]
        batch_x = np.array([example[0] for example in batch])
        batch_t = np.array([example[1] for example in batch])

        y = model(batch_x)
        loss = F.softmax_cross_entropy(y, batch_t)
        model.cleargrads()
        loss.backward()
        optimizer.update()

        sum_loss += float(loss.data) * len(batch_t)

    avg_loss = sum_loss / data_size
    history.append(avg_loss)
    print('epoch %d, loss %.5f' % (epoch + 1, avg_loss))

# %%
plt.plot(np.array(range(max_epoch)), history)
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
