import keras.backend as K
import theano.tensor as T

from .layer import Layer
from ..initializers import uniform_initializer


class DenseLayer(Layer):
    def __init__(self, input_units, units, initializer=uniform_initializer(0.05), activation=None):
        self.w = K.variable(initializer((input_units, units)))
        self.b = K.variable(initializer((units,)))
        self.units = units
        self.activation = activation
        params = [self.w, self.b]
        non_trainable_weights = []
        super(DenseLayer, self).__init__(params=params, non_trainable_weights=non_trainable_weights)

    def __str__(self):
        return "{} units={}".format(self.__class__.__name__, self.units)

    def call(self, x):
        out = T.dot(x, self.w) + self.b
        if self.activation:
            out = self.activation(out)
        return out, []
