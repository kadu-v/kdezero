# %%
import kdezero
from kdezero.core_simple import Variable
from kdezero.functions import square, exp, add
import numpy as np


x = Variable(np.array(2.0))
y = Variable(np.array(3.0))
z = add(square(x), square(y))
z.backward()
print(z.data)
print(x.grad)
print(y.grad)
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
