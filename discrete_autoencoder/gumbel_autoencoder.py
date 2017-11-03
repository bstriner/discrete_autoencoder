import keras.backend as K
import theano
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams

from .autoencoder import Autoencoder
from .gumbel import gumbel_softmax, sample_one_hot
from .initializers import uniform_initializer
from .tensor_util import softmax_nd, tensor_one_hot


class GumbelAutoencoder(Autoencoder):
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
                 learn_pz=False,
                 use_mean=False,
                 tau_min=0.25,
                 tau_decay=1e-6,
                 srng=RandomStreams(123),
                 eps=1e-7):
        self.z_n = z_n
        self.z_k = z_k
        self.encoder_net = encoder_net
        self.decoder_net = decoder_net
        self.srng = srng
        self.hard = hard
        self.learn_pz = learn_pz
        self.use_mean = use_mean
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
        if learn_pz:
            self.z_prior_weight = K.variable(initializer((z_n, z_k)), name='z_prior_weight')
            self.z_prior = T.nnet.softmax(self.z_prior_weight)  # (z_n, z_k)
            pz_params = [self.z_prior_weight]
        else:
            self.z_prior = T.ones((z_n, z_k), dtype='float32') / z_k
            pz_params = []

        # Input
        input_x = T.fmatrix(name='input_x')  # (n, input_units)
        n = input_x.shape[0]
        rnd = srng.uniform(size=input_x.shape, low=0., high=1., dtype='float32')
        input_x_binary = T.gt(input_x, rnd)  # (n, input_units)

        # Encode
        pz, z, encode_updates = self.encode(input_x_binary)  # (n, z_n, z_k)

        # Decode
        xpred, decode_updates = self.decode(z)  # (n, input_units)

        # NLL X
        nll_x = self.calc_nll_x(input_x_binary, xpred)  # (n,)
        mean_nll_x = T.mean(nll_x)  # scalar

        # KL
        kl = self.calc_kl(pz)  # scalar

        # NLL Z
        nll_z = self.calc_nll_z(z)  # (n,)
        mean_nll_z = T.mean(nll_z)

        # Validation NLL X
        val_pz, val_z, _ = self.encode(input_x_binary, validation=True)
        val_xpred, _ = self.decode(val_z, validation=True)
        val_nll_x = self.calc_nll_x(input_x_binary, val_xpred)  # (n,)
        val_mean_nll_x = T.mean(val_nll_x)

        # Validation KL
        val_kl = self.calc_kl(val_pz)  # scalar

        # Validation NLL Z
        val_nll_z = self.calc_nll_z(val_z)  # (n,)
        val_mean_nll_z = T.mean(val_nll_z)  # scalar

        val_loss = val_mean_nll_x + val_kl

        # Validation function
        val_function = theano.function([input_x], [val_mean_nll_z, val_mean_nll_x, val_kl, val_loss])
        val_headers = ['Val NLL Z', 'Val NLL X', 'KL', 'Val NLL']

        # Regularization
        self.params = pz_params + encoder_net.params + decoder_net.params
        reg_loss = T.constant(0.)
        if regularizer:
            for p in self.params:
                reg_loss += regularizer(p)

        # Training
        loss = mean_nll_x + kl + reg_loss
        train_updates = opt.get_updates(loss, self.params)
        train_function = theano.function([input_x], [mean_nll_z, mean_nll_x, kl, reg_loss, loss, self.tau],
                                         updates=train_updates + iter_updates + decode_updates + encode_updates)
        train_headers = ['NLL Z', 'NLL X', 'KL', 'Reg', 'Loss', 'Tau']
        weights = (self.params +
                   opt.weights +
                   [self.iteration] +
                   encoder_net.non_trainable_weights +
                   decoder_net.non_trainable_weights)

        # Generation
        input_n = T.iscalar()
        logitrep = T.log(self.ceps+T.repeat(T.reshape(self.z_prior, (1, z_n, z_k)), repeats=input_n, axis=0))
        zsamp = sample_one_hot(logits=logitrep, srng=srng)
        xgen, _ = self.decode(zsamp, validation=True)
        # rnd = srng.uniform(size=xgen.shape, low=0., high=1., dtype='float32')
        # xsamp = T.cast(T.gt(xgen, rnd), 'int32')
        generate_function = theano.function([input_n], xgen)  # xsamp for binarized
        z_prior_function = theano.function([], self.z_prior)

        # Autoencode
        # rnd = srng.uniform(low=0., high=1., dtype='float32', size=val_xpred.shape)
        # xout = T.cast(T.gt(val_xpred, rnd), dtype='float32')
        autoencode_function = theano.function([input_x], [input_x_binary, val_xpred])  # xout for binarized

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
        ret = "{} z_n={}, z_k={}, hard={}, learn_pz={}, use_mean={}".format(self.__class__.__name__,
                                                                            self.z_n,
                                                                            self.z_k,
                                                                            self.hard,
                                                                            self.learn_pz,
                                                                            self.use_mean)
        ret += "\nencoder_net: [\n{}\n]".format(str(self.encoder_net))
        ret += "\ndecoder_net: [\n{}\n]".format(str(self.decoder_net))
        return ret

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

    def calc_kl(self, pz):
        # KL
        if self.use_mean:
            mpz = T.mean(pz, axis=0)  # (z_n, z_k)
            kl = T.sum(mpz * (T.log(self.ceps + mpz) - T.log(self.ceps + self.z_prior)), axis=(0, 1))  # scalar
        else:
            kl_part = T.sum(pz * (T.log(self.ceps + pz) - (T.log(self.ceps + self.z_prior).dimshuffle(('x', 0, 1)))),
                            axis=(1, 2))  # (n,)
            kl = T.mean(kl_part)  # scalar
        return kl
