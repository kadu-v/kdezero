# %%
from numpy import random
from kdezero import Variable, Function, using_config, no_grad
import kdezero.functions as F
import kdezero.layers as L
from kdezero.utils import _dot_func, _dot_var, plot_dot_graph
from kdezero.models import Model, MLP
from kdezero import optimizers
import numpy as np
import matplotlib.pyplot as plt


np.random.seed(0)
X_train = np.random.rand(100, 1)
y_train = np.sin(2 * np.pi * X_train) + np.random.rand(100, 1)

lr = 0.2
iters = 10000
hidden_size = 10


model = MLP((hidden_size, 1))
optimizer = optimizers.MomentumSGD(lr)
optimizer.setup(model)


for i in range(iters):
    y_pred = model(X_train)
    loss = F.mean_squared_error(y_pred, y_train)

    model.cleargrads()
    loss.backward()

    optimizer.update()

    if i % 1000 == 0:
        print(loss)

# %%
t = np.arange(0, 1, .01)[:, np.newaxis]
y_pred = model(t)

plt.scatter(X_train, y_train, marker='+')
plt.plot(t, y_pred.data.flatten(), color='r')
plt.show()

# %%
