# %%
import kdezero
from kdezero.core_simple import Variable, using_config, no_grad
from kdezero.functions import square, exp, add
import numpy as np


with using_config('enable_backprop', False):
    x = Variable(np.array(2.0))
    y = square(x)
    print(y.data)


with no_grad():
    x = Variable(np.array(2.0))
    y = square(x)
    print(y.data)
# %%
