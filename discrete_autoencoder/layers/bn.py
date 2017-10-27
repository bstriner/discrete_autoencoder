import keras.backend as K
import numpy as np
import theano.tensor as T
import theano.tensor.nnet.bn as BN

from .layer import Layer


class BNLayer(Layer):
    def __init__(self, units):
        self.units = units
        self.gamma = K.variable(0.1 * np.ones((units,), dtype=np.float32), dtype='float32')
        self.beta = K.variable(np.zeros((units,), dtype=np.float32), dtype='float32')
        self.mean = K.variable(np.zeros((units,), dtype=np.float32), dtype='float32')
        self.var = K.variable(0.1 * np.ones((units,), dtype=np.float32), dtype='float32')
        params = [self.gamma, self.beta]
        non_trainable_weights = [self.mean, self.var]
        super(BNLayer, self).__init__(params=params, non_trainable_weights=non_trainable_weights)

    def __str__(self):
        return "{} units={}".format(self.__class__.__name__, self.units)

    def call(self, x):
        out, mean, std, newmean, newvar = BN.batch_normalization_train(inputs=x,
                                                                       gamma=self.gamma,
                                                                       beta=self.beta,
                                                                       axes='per-activation',
                                                                       running_mean=self.mean,
                                                                       running_var=self.var)
        updates = [(self.mean, T.cast(newmean, 'float32')), (self.var, T.cast(newvar, 'float32'))]
        return out, updates

    def call_validation(self, x):
        out = BN.batch_normalization_test(inputs=x,
                                          gamma=self.gamma,
                                          beta=self.beta,
                                          mean=self.mean,
                                          var=self.var,
                                          axes='per-activation')
        return out
