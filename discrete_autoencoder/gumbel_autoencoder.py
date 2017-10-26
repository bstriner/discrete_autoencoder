import keras.backend as K
import theano
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams

from .discrete_autoencoder import DiscreteAutoencoder
from .gumbel import gumbel_softmax, sample_one_hot
from .initializers import uniform_initializer
from .layers import Stack, DenseLayer, LSTMLayer
from .tensor_util import softmax_nd, tensor_one_hot


class GumbelAutoencoder(DiscreteAutoencoder):
    def __init__(self,
                 z_n,
                 z_k,
                 encoder_net,
                 decoder_net,
                 opt,
                 regularizer=None,
                 initializer=uniform_initializer(0.05),
                 hard=True,
                 pz_units=512,
                 tau0=5.,
                 tau_min=0.25,
                 tau_decay=1e-6,
                 kl_weight=1.,
                 recurrent_pz=False,
                 srng=RandomStreams(123),
                 eps=1e-9):
        self.z_n = z_n
        self.z_k = z_k
        self.encoder_net = encoder_net
        self.decoder_net = decoder_net
        self.srng = srng
        self.hard = hard
        self.recurrent_pz = recurrent_pz
        self.ceps = T.constant(eps, name='epsilon', dtype='float32')

        # Temperature
        self.iteration = K.variable(0, dtype='int32', name='iteration')
        iter_updates = [(self.iteration, self.iteration + 1)]
        tau = T.constant(tau0, dtype='float32', name='tau0')
        if tau_decay > 0:
            tau_decay = T.constant(tau_decay, name='tau_decay', dtype='float32')
            tau_min = T.constant(tau_min, name='tau_min', dtype='float32')
            tau = tau / (1. + (tau_decay * self.iteration))
            tau = T.nnet.relu(tau - tau_min) + tau_min
        self.tau = tau

        # Prior
        if recurrent_pz:
            self.pz_net = Stack([
                DenseLayer(z_k + 1, pz_units, initializer=initializer),
                LSTMLayer(pz_units, pz_units, initializer=initializer),
                LSTMLayer(pz_units, pz_units, initializer=initializer),
                DenseLayer(pz_units, z_k)
            ])
            pz_params = self.pz_net.params
            pz_non_trainable_weights = self.pz_net.non_trainable_weights
        else:
            self.z_prior_weight = K.variable(initializer((z_n, z_k)), name='z_prior_weight')
            self.z_prior = T.nnet.softmax(self.z_prior_weight)  # (z_n, z_k)
            pz_params = [self.z_prior_weight]
            pz_non_trainable_weights = []

        # Input
        input_x = T.fmatrix(name='input_x')  # (n, input_units)
        n = input_x.shape[0]
        rnd = srng.uniform(size=input_x.shape, low=0., high=1., dtype='float32')
        input_x_binary = T.gt(input_x, rnd)  # (n, input_units)

        # Encode
        pz, z, encode_updates = self.encode(input_x_binary)  # (n, z_n, z_k)

        # Decode
        xpred, decode_updates = self.decode(z)  # (n, input_units)

        # NLL
        nll_z = self.calc_nll_z(z)  # (n,)
        nll_x = self.calc_nll_x(input_x_binary, xpred)  # (n,)
        nll_part = nll_x + nll_z
        nll = T.mean(nll_part)
        mean_nll_z = T.mean(nll_z)
        mean_nll_x = T.mean(nll_x)

        # Validation NLL
        _, val_z, _ = self.encode(input_x_binary, validation=True)
        val_xpred, _ = self.decode(T.reshape(val_z, (n, z_n * z_k)), validation=True)
        val_nll_z = self.calc_nll_z(val_z)  # (n,)
        val_nll_x = self.calc_nll_x(input_x_binary, val_xpred)  # (n,)
        val_nll_part = val_nll_z + val_nll_x
        val_nll = T.mean(val_nll_part)
        val_mean_nll_z = T.mean(val_nll_z)
        val_mean_nll_x = T.mean(val_nll_x)
        val_function = theano.function([input_x], [val_mean_nll_z, val_mean_nll_x, val_nll])
        val_headers = ['Val NLL Z', 'Val NLL X', 'Val NLL']

        # Regularization
        self.params = pz_params + encoder_net.params + decoder_net.params
        reg_loss = T.constant(0.)
        if regularizer:
            for p in self.params:
                reg_loss += regularizer(p)

        # KL
        kl_part = T.sum(pz * (T.log(eps + pz) - T.log(1. / z_k)), axis=(1, 2))  # (n,)
        kl = kl_weight * T.mean(kl_part)

        # Training
        loss = nll + reg_loss + kl
        train_updates = opt.get_updates(loss, self.params)
        train_function = theano.function([input_x], [mean_nll_z, mean_nll_x, nll, reg_loss, kl, loss, self.tau],
                                         updates=train_updates + iter_updates + decode_updates + encode_updates)
        train_headers = ['NLL Z', 'NLL X', 'NLL', 'Reg', 'KL', 'Loss', 'Tau']
        weights = (self.params +
                   opt.weights +
                   [self.iteration] +
                   pz_non_trainable_weights +
                   encoder_net.non_trainable_weights +
                   decoder_net.non_trainable_weights)

        # Generation
        if recurrent_pz:
            generate_function = None
            z_prior_function = None
        else:
            input_n = T.iscalar()
            logitrep = T.repeat(T.reshape(self.z_prior_weight, (1, z_n, z_k)), repeats=input_n, axis=0)
            zsamp = sample_one_hot(logits=logitrep, srng=srng)
            xgen, _ = self.decode(T.reshape(zsamp, (input_n, -1)), validation=True)
            rnd = srng.uniform(size=xgen.shape, low=0., high=1., dtype='float32')
            xsamp = T.cast(T.gt(xgen, rnd), 'int32')
            generate_function = theano.function([input_n], xsamp)
            z_prior_function = theano.function([], self.z_prior)

        # Autoencode
        rnd = srng.uniform(low=0., high=1., dtype='float32', size=val_xpred.shape)
        xout = T.cast(T.gt(val_xpred, rnd), dtype='float32')
        autoencode_function = theano.function([input_x], [input_x_binary, xout])

        super(GumbelAutoencoder, self).__init__(
            train_headers=train_headers,
            val_headers=val_headers,
            train_function=train_function,
            generate_function=generate_function,
            val_function=val_function,
            z_prior_function=z_prior_function,
            autoencode_function=autoencode_function,
            weights=weights
        )

    def __str__(self):
        return "{} z_n={}, z_k={}, hard={}".format(self.__class__.__name__, self.z_n, self.z_k, self.hard)

    def encode(self, x, validation=False):
        assert x is not None
        n = x.shape[0]
        if validation:
            h = self.encoder_net.call_validation(x)  # (n, z_n*z_k)
            updates = []
        else:
            h, updates = self.encoder_net.call(x)  # (n, z_n*z_k)
        logits = T.reshape(h, (n, self.z_n, self.z_k))
        pz = softmax_nd(logits)
        if validation:
            z = tensor_one_hot(T.argmax(logits, axis=-1), logits.shape[-1])
        else:
            z = gumbel_softmax(logits=logits, temperature=self.tau, srng=self.srng, hard=self.hard)  # (n, z_n, z_k)
        return pz, z, updates

    def calc_nll_z(self, z):
        assert z.ndim == 3
        # z: (n, z_n, z_k)
        # z = theano.gradient.zero_grad(z)
        if self.recurrent_pz:
            n = z.shape[0]
            zs = T.concatenate((T.zeros((n, 1, self.z_k), dtype='float32'), z[:, :-1, :]), axis=1)  # (n, z_n, z_k)
            zs = T.concatenate((T.zeros((n, self.z_n, 1), dtype='float32'), zs), axis=2)  # (n, z_n, z_k)
            zs = T.set_subtensor(zs[:, 0, 0], 1)
            # assert zs.dtype=='float32'
            zs = T.cast(zs, 'float32')
            # zs = theano.gradient.zero_grad(zs)
            logits, pz_updates = self.pz_net.call(zs)
            # logits: (n, z_n, z_k)
            pz = softmax_nd(logits)
            # v1
            pzt = T.sum(pz * z, axis=2)  # (n, z_k)
            nll_z = -T.sum(T.log(self.ceps + pzt), axis=1)  # (n,)
            # v2
            #nll_z = -T.sum(z*T.log(self.ceps+pz), axis=(1,2))
            return nll_z  # (n,)
        else:
            # Old version
            # nll_z_prior = -T.log(self.ceps + self.z_prior)  # (z_n, z_k)
            # nll_z = T.tensordot(z, nll_z_prior, axes=((1, 2), (0, 1)))  # (n,)
            h = T.sum(z * (self.z_prior.dimshuffle(('x', 0, 1))), axis=2)  # (n, z_n)
            nll_z = -T.sum(T.log(self.ceps + h), axis=1)  # (n,)
            return nll_z  # (n,)

    def decode(self, z, validation=False):
        n = z.shape[0]
        z_flat = T.reshape(z, (n, self.z_n * self.z_k))
        if validation:
            h = self.decoder_net.call_validation(z_flat)  # (n, input_units)
            updates = []
        else:
            h, updates = self.decoder_net.call(z_flat)  # (n, input_units)
        xpred = T.nnet.sigmoid(h)
        return xpred, updates

    def calc_nll_x(self, x, xpred):
        xp = T.switch(x, xpred, 1. - xpred)  # (n, input_units)
        nll_x = -T.sum(T.log(self.ceps + xp), axis=1)  # (n,)
        return nll_x
