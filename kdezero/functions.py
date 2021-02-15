import numpy as np
from kdezero.core_simple import Function

# ==================================================================================================
# Basic Functions: Exp
# ==================================================================================================


class Exp(Function):
    def forward(self, x):
        return np.exp(x)
