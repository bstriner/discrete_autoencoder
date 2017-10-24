from .layer import Layer


class ActivationLayer(Layer):
    def __init__(self, activation=None):
        self.activation = activation
        params = []
        non_trainable_weights = []
        super(ActivationLayer, self).__init__(params=params, non_trainable_weights=non_trainable_weights)

    def call(self, x):
        out = self.activation(x)
        return out, []
