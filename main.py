import kdezero
from kdezero.core_simple import Variable
from kdezero.functions import square, exp
import numpy as np


x = Variable(np.array(0.5))
a = square(x)
b = exp(a)
y = square(b)

y.backward()
print(x.grad)
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
