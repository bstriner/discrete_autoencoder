import keras.backend as K
import theano
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams

from .discrete_autoencoder import DiscreteAutoencoder
from .gumbel import sample_one_hot
from .initializers import uniform_initializer
from .tensor_util import softmax_nd
from .tensor_util import tensor_one_hot


class ReinforceAutoencoder(DiscreteAutoencoder):
    def __init__(self,
                 z_n,
                 z_k,
                 encoder_net,
                 decoder_net,
                 opt,
                 regularizer=None,
                 entropy_weight=1.,
                 initializer=uniform_initializer(0.05),
                 srng=RandomStreams(123),
                 beta=0.01,
                 eps=1e-6):
        self.z_n = z_n
        self.z_k = z_k
        self.encoder_net = encoder_net
        self.decoder_net = decoder_net
        self.srng = srng
        self.ceps = T.constant(eps, name='epsilon', dtype='float32')
        self.entropy_weight = entropy_weight

        # Prior
        self.z_prior_weight = K.variable(initializer((z_n, z_k)), name='z_prior_weight')
        self.z_prior = T.nnet.softmax(self.z_prior_weight)  # (z_n, z_k)

        # Input
        input_x = T.fmatrix(name='input_x')  # (n, input_units)
        n = input_x.shape[0]
        rnd = srng.uniform(size=input_x.shape, low=0., high=1., dtype='float32')
        input_x_binary = T.gt(input_x, rnd)  # (n, input_units)

        # Encode
        pz, z, encode_updates = self.encode(input_x_binary)
        # pz: (n, z_n, z_k)
        # z: (n, z_n, z_k) one-hot
        pzt = T.sum(pz * z, axis=2)  # (n, z_n)
        logpz = T.sum(T.log(self.ceps + pzt), axis=1)  # (n,)

        # Decode
        xpred, decode_updates = self.decode(z)  # (n, input_units)

        # NLL
        nll_z = self.calc_nll_z(z)  # (n,)
        nll_x = self.calc_nll_x(input_x_binary, xpred)  # (n,)
        nll_part = nll_x + nll_z  # (n,)
        nll = T.mean(nll_part)
        mean_nll_z = T.mean(nll_z)
        mean_nll_x = T.mean(nll_x)

        # Reinforce
        mean_nll = K.variable(0., dtype='float32', name='mean_nll')
        new_mean_nll = (beta * nll) + ((1. - beta) * mean_nll)
        mean_updates = [(mean_nll, new_mean_nll)]
        reinforce_loss = T.mean(logpz * theano.gradient.zero_grad(nll_part - mean_nll))

        # Regularization
        self.params = [self.z_prior_weight] + encoder_net.params + decoder_net.params
        reg_loss = T.constant(0.)
        if regularizer:
            for p in self.params:
                reg_loss += regularizer(p)

        # Entropy regularizer
        # entropy = T.sum(pz * (T.log(self.ceps+pz)-T.log(1./z_k)), axis=(1, 2))  # (n,)
        # entropy_reg = entropy_weight * T.mean(entropy)
        pzm = T.mean(pz, axis=0)  # (z_n, z_k)
        entropy = T.sum(pzm * (T.log(self.ceps + pzm) - T.log(1. / z_k)), axis=(0, 1))  # scalar
        entropy_reg = entropy_weight * entropy

        # Training
        loss = nll + reinforce_loss + reg_loss + entropy_reg
        train_updates = opt.get_updates(loss, self.params)
        train_function = theano.function([input_x], [mean_nll_z,
                                                     mean_nll_x,
                                                     nll,
                                                     reinforce_loss,
                                                     reg_loss,
                                                     entropy_reg,
                                                     loss],
                                         updates=train_updates + decode_updates + encode_updates + mean_updates)
        train_headers = ['NLL Z', 'NLL X', 'NLL', 'Reinforce', 'Reg', 'Entropy', 'Loss']
        weights = (self.params +
                   opt.weights +
                   encoder_net.non_trainable_weights +
                   decoder_net.non_trainable_weights +
                   [mean_nll])

        # Validation
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

        # Generation
        input_n = T.iscalar()
        logitrep = T.repeat(T.reshape(self.z_prior_weight, (1, z_n, z_k)), repeats=input_n, axis=0)
        zsamp = sample_one_hot(logitrep, srng=srng)
        xgen, _ = self.decode(zsamp, validation=True)
        rnd = srng.uniform(size=xgen.shape, low=0., high=1., dtype='float32')
        xsamp = T.cast(T.gt(xgen, rnd), 'int32')
        generate_function = theano.function([input_n], xsamp)

        z_prior_function = theano.function([], self.z_prior)

        super(ReinforceAutoencoder, self).__init__(
            train_headers=train_headers,
            val_headers=val_headers,
            train_function=train_function,
            generate_function=generate_function,
            val_function=val_function,
            z_prior_function=z_prior_function,
            weights=weights
        )

    def __str__(self):
        return "{} z_n={}, z_k={}, entropy_weight={}".format(self.__class__.__name__,
                                                             self.z_n,
                                                             self.z_k,
                                                             self.entropy_weight)

    def encode(self, x, validation=False):
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
            z = sample_one_hot(logits=logits, srng=self.srng)  # (n, z_n, z_k)
        return pz, z, updates

    def calc_nll_z(self, z):
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
