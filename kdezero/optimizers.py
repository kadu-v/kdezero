import math
from kdezero import Parameter


# =================================================================================================
# Optimizer (base class)
# =================================================================================================

class Optimizer:
    def __init__(self) -> None:
        self.targets = None
        self.hooks = []

    def setup(self, target):
        self.targets = target
        return self

    def update(self):
        # None 以外のパラメータをリストにまとめる
        params = [p for p in self.targets.params() if p.grad is not None]

        # 前処理
        for f in self.hooks:
            f(params)
        # パラメータの更新
        for param in params:
            self.update_one(param)

    def update_one(self, param):
        raise NotImplementedError()

    def add_hook(self, f):
        self.hooks.append(f)


# =================================================================================================
# Optimizer (base class)
# =================================================================================================


class SGD(Optimizer):
    def __init__(self, lr=0.01):
        super().__init__()
        self.lr = lr

    def update_one(self, param):
        param.data -= self.lr * param.grad.data
