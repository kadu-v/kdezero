# %%
import kdezero
from kdezero.core_simple import Variable
from kdezero.functions import square, exp, add
import numpy as np


x = Variable(np.array(3.0))
y = add(x, x)

print(y.data)

y.backward()
print(x.grad)

x.cleargrad()
y = add(x, add(x, x))
y.backward()
print(x.grad)

# %%
