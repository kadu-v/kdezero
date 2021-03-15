# %%
import kdezero
from kdezero.core_simple import Variable
from kdezero.functions import square, exp, add
import numpy as np


x = Variable(np.array(2.0))
a = square(x)
y = add(square(a), square(a))

print(y.data)

y.backward()
print(x.grad)

# %%
