import keras.backend as K
import theano
import theano.tensor as T

from .layer import Layer
from ..initializers import uniform_initializer


class LSTMLayer(Layer):
    def __init__(self, input_units, units, initializer=uniform_initializer(0.05)):
        self.fw = K.variable(initializer((input_units, units)))
        self.fu = K.variable(initializer((units, units)))
        self.fb = K.variable(initializer((units,)))
        self.iw = K.variable(initializer((input_units, units)))
        self.iu = K.variable(initializer((units, units)))
        self.ib = K.variable(initializer((units,)))
        self.cw = K.variable(initializer((input_units, units,)))
        self.cu = K.variable(initializer((units, units)))
        self.cb = K.variable(initializer((units,)))
        self.ow = K.variable(initializer((input_units, units)))
        self.ou = K.variable(initializer((units, units,)))
        self.ob = K.variable(initializer((units,)))
        self.non_sequences = [
            self.fw, self.fu, self.fb,
            self.iw, self.iu, self.ib,
            self.cw, self.cu, self.cb,
            self.ow, self.ou, self.ob,
        ]
        self.h0 = K.variable(initializer((1, units)))
        params = self.non_sequences + [self.h0]
        non_trainable_weights = []
        super(LSTMLayer, self).__init__(params=params, non_trainable_weights=non_trainable_weights)

    def call(self, x):
        assert x.ndim == 3
        assert x.dtype == 'float32'
        n = x.shape[0]
        sequences = [T.transpose(x, (1, 0, 2))]  # (depth, n, units)
        outputs_info = [T.repeat(self.h0, repeats=n, axis=0), None]
        (h1, y1r), _ = theano.scan(self.scan,
                                   sequences=sequences,
                                   outputs_info=outputs_info,
                                   non_sequences=self.non_sequences)
        y1 = T.transpose(y1r, (1, 0, 2))  # (n, depth, units)
        return y1, []

    def scan(self, x, h0, *params):
        idx = 0
        vars = []
        for i in range(4):
            w = params[idx + 0]
            u = params[idx + 1]
            b = params[idx + 2]
            idx += 3
            var = T.dot(x, w) + T.dot(h0, u) + b
            vars.append(var)
        assert idx == len(params)
        f, i, c, o = vars

        h1 = T.nnet.sigmoid(f) * h0 + T.nnet.sigmoid(i) * T.tanh(c)
        y1 = T.nnet.sigmoid(o) * T.tanh(h1)
        return h1, y1
