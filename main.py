# %%
from numpy import random
from kdezero import Variable, Function, using_config, no_grad
import kdezero.functions as F
from kdezero.utils import _dot_func, _dot_var, plot_dot_graph
import numpy as np
import matplotlib.pyplot as plt


np.random.seed(0)
X_train = np.random.rand(100, 1)
y_train = 5 + 2 * X_train + np.random.rand(100, 1)
x, y = Variable(X_train), Variable(y_train)

W = Variable(np.zeros((1, 1)))
b = Variable(np.zeros(1))


def predict(x):
    y = F.matmul(x, W) + b
    return y


def mean_squared_error(x0, x1):
    diff = x0 - x1
    return F.sum(diff ** 2) / len(diff)


lr = 0.1
iters = 100
for i in range(iters):
    y_pred = predict(x)
    loss = F.mean_squared_error(y, y_pred)

    W.cleargrad()
    b.cleargrad()
    loss.backward()

    W.data -= lr * W.grad.data
    b.data -= lr * b.grad.data
    print(W, b)

y_pred = predict(x)

plt.scatter(X_train, y_train, marker='+')
plt.plot(X_train, y_pred.data.flatten(), color='r')
plt.show()
