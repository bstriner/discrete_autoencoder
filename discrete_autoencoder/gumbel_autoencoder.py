import keras.backend as K
import theano
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams

from .discrete_autoencoder import DiscreteAutoencoder
from .gumbel import gumbel_softmax, sample_gumbel
from .initializers import uniform_initializer


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
                 tau0=5.,
                 tau_min=0.25,
                 tau_decay=1e-6,
                 srng=RandomStreams(123),
                 eps=1e-9):
        self.z_n = z_n
        self.z_k = z_k
        self.encoder_net = encoder_net
        self.decoder_net = decoder_net
        self.srng = srng
        self.hard = hard
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
        self.z_prior_weight = K.variable(initializer((z_n, z_k)), name='z_prior_weight')
        self.z_prior = T.nnet.softmax(self.z_prior_weight)  # (z_n, z_k)

        # Input
        input_x = T.fmatrix(name='input_x')  # (n, input_units)
        n = input_x.shape[0]
        rnd = srng.uniform(size=input_x.shape, low=0., high=1., dtype='float32')
        input_x_binary = T.gt(input_x, rnd)  # (n, input_units)

        # Encode
        z = self.encode(input_x_binary)  # (n, z_n, z_k)

        # Decode
        xpred = self.decode(T.reshape(z, (n, z_n * z_k)))  # (n, input_units)

        # NLL
        nll_z = self.calc_nll_z(z)  # (n,)
        nll_x = self.calc_nll_x(input_x_binary, xpred)  # (n,)
        nll_part = nll_x + nll_z
        nll = T.mean(nll_part)

        # Regularization
        self.params = [self.z_prior_weight] + encoder_net.params + decoder_net.params
        reg_loss = T.constant(0.)
        if regularizer:
            for p in self.params:
                reg_loss += regularizer(p)

        # Training
        loss = nll + reg_loss
        train_updates = opt.get_updates(loss, self.params)
        train_function = theano.function([input_x], [nll, reg_loss, loss, self.tau],
                                         updates=train_updates + iter_updates)
        train_headers = ['NLL', 'Reg', 'Loss', 'Tau']
        weights = self.params + opt.weights + [self.iteration]

        # Validation
        val_function = theano.function([input_x], nll)

        # Generation
        input_n = T.iscalar()
        logitrep = T.repeat(T.reshape(self.z_prior_weight, (1, z_n, z_k)), repeats=input_n, axis=0)
        g = logitrep + sample_gumbel(shape=logitrep.shape, srng=srng)
        zsamp = T.eq(g, T.max(g, axis=-1, keepdims=True))  # (n, z_n, z_k)
        xgen = self.decode(T.reshape(zsamp, (input_n, -1)))
        rnd = srng.uniform(size=xgen.shape, low=0., high=1., dtype='float32')
        xsamp = T.cast(T.gt(xgen, rnd), 'int32')
        generate_function = theano.function([input_n], xsamp)

        super(GumbelAutoencoder, self).__init__(
            train_headers=train_headers,
            train_function=train_function,
            generate_function=generate_function,
            val_function=val_function,
            weights=weights
        )

    def encode(self, x):
        n = x.shape[0]
        h = self.encoder_net.call(x)  # (n, z_n*z_k)
        logits = T.reshape(h, (n, self.z_n, self.z_k))
        z = gumbel_softmax(logits=logits, temperature=self.tau, srng=self.srng, hard=self.hard)  # (n, z_n, z_k)
        return z

    def calc_nll_z(self, z):
        nll_z_prior = -T.log(self.ceps + self.z_prior)  # (z_n, z_k)
        nll_z = T.tensordot(z, nll_z_prior, axes=((1, 2), (0, 1)))  # (n,)
        return nll_z  # (n,)

    def decode(self, z):
        n = z.shape[0]
        z_flat = T.reshape(z, (n, self.z_n * self.z_k))
        xpred = T.nnet.sigmoid(self.decoder_net.call(z_flat))  # (n, input_units)
        return xpred

    def calc_nll_x(self, x, xpred):
        xp = T.switch(x, xpred, 1. - xpred)  # (n, input_units)
        nll_x = -T.sum(T.log(self.ceps + xp), axis=1)  # (n,)
        return nll_x
