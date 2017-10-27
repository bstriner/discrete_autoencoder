from .layer import Layer


class DropoutLayer(Layer):
    def __init__(self, rate, srng):
        self.srng = srng
        self.rate = rate
        params = []
        non_trainable_weights = []
        super(DropoutLayer, self).__init__(params=params, non_trainable_weights=non_trainable_weights)

    def __str__(self):
        return "{} rate={}".format(self.__class__.__name__, self.rate)

    def call(self, x):
        if self.rate > 0:
            rnd = self.srng.binomial(n=1, p=1. - self.rate, size=x.shape, dtype='float32')
            out = (x * rnd) / (1. - self.rate)
        else:
            out = x
        return out, []

    def call_validation(self, x):
        return x
