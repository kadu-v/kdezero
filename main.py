# %%
from numpy import random
from kdezero import Variable, Function, using_config, no_grad
import kdezero.functions as F
from kdezero.utils import _dot_func, _dot_var, plot_dot_graph
import numpy as np
import matplotlib.pyplot as plt


np.random.seed(0)
X_train = np.random.rand(100, 1)
y_train = np.sin(2 * np.pi * X_train) + np.random.rand(100, 1)

I, H, O = 1, 10, 1
W1 = Variable(0.01 * np.random.rand(I, H))
b1 = Variable(np.zeros(H))
W2 = Variable(0.01 * np.random.rand(H, O))
b2 = Variable(np.zeros(O))


def predict(x):
    y = F.linear(x, W1, b1)
    y = F.sigmoid(y)
    y = F.linear(y, W2, b2)
    return y


lr = 0.2
iters = 10000

for i in range(iters):
    y_pred = predict(X_train)
    loss = F.mean_squared_error(y_pred, y_train)

    W1.cleargrad()
    b1.cleargrad()
    W2.cleargrad()
    b2.cleargrad()
    loss.backward()

    W1.data -= lr * W1.grad.data
    b1.data -= lr * b1.grad.data
    W2.data -= lr * W2.grad.data
    b2.data -= lr * b2.grad.data

    if i % 100 == 0:
        print(loss)

# %%
t = np.arange(0, 1, .01)[:, np.newaxis]
y_pred = predict(t)

plt.scatter(X_train, y_train, marker='+')
plt.plot(t, y_pred.data.flatten(), color='r')
plt.show()

# %%
