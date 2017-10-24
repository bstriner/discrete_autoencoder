import keras.backend as K
import theano.tensor as T

from .initializers import uniform_initializer

import numpy as np

class MLP(object):
    def __init__(self,
                 input_units,
                 units,
                 initializer=uniform_initializer(0.05),
                 hidden_activation=None,
                 output_activation=None,
                 use_bn=False):
        self.units = units
        self.hidden_activation = hidden_activation
        self.output_activation = output_activation
        self.params = []
        self.use_bn = use_bn

        ws = []
        bs = []

        u0 = input_units
        for u1 in units:
            ws.append(K.variable(initializer((u0, u1))))
            bs.append(K.variable(initializer((u1,))))
            u0 = u1
        self.ws = ws
        self.bs = bs
        self.params = ws + bs
        self.other_weights = []
        if self.use_bn:
            gammas = []
            betas = []
            means = []
            stds = []
            for i in range(len(self.units)-1):
                gammas.append(K.variable(0.1 * np.ones((self.units[i],), dtype=np.float32), dtype='float32'))
                betas.append(K.variable(np.zeros((self.units[i],), dtype=np.float32), dtype='float32'))
                means.append(K.variable( np.zeros((self.units[i],), dtype=np.float32), dtype='float32'))
                stds.append(K.variable(0.1 * np.ones((self.units[i],), dtype=np.float32), dtype='float32'))
            self.params += gammas
            self.params += betas
            self.other_weights += means
            self.other_weights += stds

    def call(self, x):
        return self.call_on_params(x, self.params)

    def call_on_params(self, x, params):
        # Parse params
        k = len(self.units)
        idx = 0
        ws = params[idx:idx+k]
        idx += k
        bs = params[idx:idx+k]
        idx += k
        if self.use_bn:
            gammas = params[idx:idx+k-1]
            idx+=k-1
            betas = params[idx:idx+k-1]
            idx+=k-1
        assert idx == len(params)

        # Call
        h = x
        for i in range(k):
            h = T.dot(h, ws[i]) + bs[i]
            if i < k -1: # not last layer
                if self.use_bn:

            if i < k - 1 and self.hidden_activation:
                h = self.hidden_activation(h)
        if self.output_activation:
            h = self.output_activation(h)
        return h
