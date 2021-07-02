# %%
from numpy import random
from kdezero import Variable, Function, using_config, no_grad
import kdezero.functions as F
import kdezero.layers as L
from kdezero.utils import _dot_func, _dot_var, plot_dot_graph
import numpy as np
import matplotlib.pyplot as plt


np.random.seed(0)
X_train = np.random.rand(100, 1)
y_train = np.sin(2 * np.pi * X_train) + np.random.rand(100, 1)

l1 = L.Linear(10)
l2 = L.Linear(1)


def predict(x):
    y = l1(x)
    y = F.sigmoid(y)
    y = l2(y)
    return y


lr = 0.2
iters = 10000

for i in range(iters):
    y_pred = predict(X_train)
    loss = F.mean_squared_error(y_pred, y_train)

    l1.cleargrads()
    l2.cleargrads()
    loss.backward()

    for l in [l1, l2]:
        for p in l.params():
            p.data -= lr * p.grad.data

    if i % 1000 == 0:
        print(loss)

# %%
t = np.arange(0, 1, .01)[:, np.newaxis]
y_pred = predict(t)

plt.scatter(X_train, y_train, marker='+')
plt.plot(t, y_pred.data.flatten(), color='r')
plt.show()

# %%
