from .layer import Layer


class DropoutLayer(Layer):
    def __init__(self, rate, srng):
        self.srng = srng
        self.rate = rate
        params = []
        non_trainable_weights = []
        super(DropoutLayer, self).__init__(params=params, non_trainable_weights=non_trainable_weights)

    def call(self, x):
        rnd = self.srng.binomial(n=1, p=1. - self.rate, size=x.shape, dtype='float32')
        out = (x * rnd) / (1. - self.rate)
        return out, []

    def call_validation(self, x):
        return x
