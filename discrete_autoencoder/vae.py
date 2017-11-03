import theano
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams

from .autoencoder import Autoencoder
from .initializers import uniform_initializer


class VAE(Autoencoder):
    def __init__(self,
                 z_k,
                 encoder_net,
                 decoder_net,
                 opt,
                 regularizer=None,
                 initializer=uniform_initializer(0.05),
                 srng=RandomStreams(123),
                 eps=1e-7):
        self.z_k = z_k
        self.encoder_net = encoder_net
        self.decoder_net = decoder_net
        self.srng = srng
        self.ceps = T.constant(eps, name='epsilon', dtype='float32')

        # Input
        input_x = T.fmatrix(name='input_x')  # (n, input_units)
        n = input_x.shape[0]
        rnd = srng.uniform(size=input_x.shape, low=0., high=1., dtype='float32')
        input_x_binary = T.gt(input_x, rnd)  # (n, input_units)

        train_loss, train_nll_x, train_kl, calc_updates = self.calc_loss(input_x_binary, validation=False)
        val_loss, val_nll_x, val_kl, _ = self.calc_loss(input_x_binary, validation=True)

        # Validation function
        val_function = theano.function([input_x], [val_nll_x, val_kl, val_loss])
        val_headers = ['Val NLL X', 'Val KL', 'Val Loss']

        # Regularization
        self.params = encoder_net.params + decoder_net.params
        reg_loss = T.constant(0.)
        if regularizer:
            for p in self.params:
                reg_loss += regularizer(p)

        # Training
        loss = train_loss + reg_loss
        train_updates = opt.get_updates(loss, self.params)
        train_function = theano.function([input_x], [train_nll_x, train_kl, reg_loss, train_loss],
                                         updates=train_updates + calc_updates)
        train_headers = ['NLL X', 'KL', 'Reg', 'Loss']
        weights = (self.params +
                   opt.weights +
                   encoder_net.non_trainable_weights +
                   decoder_net.non_trainable_weights)

        # Generation
        input_n = T.iscalar()
        zsamp = self.srng.normal(size=(input_n, self.z_k), avg=0., std=1., dtype='float32')
        xgen, _ = self.decode(zsamp, validation=True)
        generate_function = theano.function([input_n], xgen)  # xsamp for binarized

        # Autoencode
        z, _, _, _ = self.encode(input_x_binary, validation=True)
        x, _ = self.decode(z, validation=True)
        autoencode_function = theano.function([input_x], [input_x_binary, x])  # xout for binarized

        super(VAE, self).__init__(
            train_headers=train_headers,
            val_headers=val_headers,
            train_function=train_function,
            generate_function=generate_function,
            val_function=val_function,
            autoencode_function=autoencode_function,
            weights=weights
        )

    def __str__(self):
        ret = "{} z_k={}".format(self.__class__.__name__, self.z_k)
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
        means = h[:, :self.z_k]
        vars = T.exp(0.5 * h[:, self.z_k:])
        noise = self.srng.normal(avg=0, std=1, size=(n, self.z_k), dtype='float32')
        if validation:
            z = means
        else:
            z = means + (vars * noise)
        return z, means, vars, updates

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

    def calc_kl(self, mean, var):
        logvar = T.log(self.ceps + var)
        kl = -0.5 * T.sum(1 + logvar - T.square(mean) - T.exp(logvar), axis=1)
        return kl  # (n,)

    def calc_loss(self, input_x_binary, validation=False):
        # Encode
        z, mean, var, encode_updates = self.encode(input_x_binary, validation=validation)  # (n, z_n, z_k)

        # Decode
        xpred, decode_updates = self.decode(z, validation=validation)  # (n, input_units)

        # NLL X
        nll_x = self.calc_nll_x(input_x_binary, xpred)  # (n,)
        mean_nll_x = T.mean(nll_x)  # scalar

        # KL
        kl = self.calc_kl(mean, var)  # (n,)
        mean_kl = T.mean(kl)

        mean_loss = mean_kl + mean_nll_x
        return mean_loss, mean_nll_x, mean_kl, encode_updates + decode_updates
