# %%
import kdezero
from kdezero.core_simple import Variable
from kdezero.functions import square, exp, Add
import numpy as np


x0 = Variable(np.array(0.5))
x1 = Variable(np.array(0.5))
y = Add()(x0, x1)
print(y.data)

# C = y.creator
# b = C.input
# b.grad = C.backward(y.grad)

# B = b.creator
# a = B.input
# a.grad = B.backward(b.grad)

# A = a.creator
# x = A.input
# x.grad = A.backward(a.grad)

# print(x.grad)

# %%
