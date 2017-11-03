import keras.backend as K
import numpy as np
import png
import theano
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams

from .autoencoder import Autoencoder
from .gumbel import gumbel_softmax, sample_one_hot
from .initializers import uniform_initializer
from .tensor_util import softmax_nd, tensor_one_hot, fix_updates
from .util import make_dir


class GumbelQuantizedAutoencoder(Autoencoder):
    def __init__(self,
                 z_n,
                 z_k,
                 encoder_net,
                 decoder_net,
                 opt,
                 iw=False,
                 iw_samples=10,
                 val_iw=False,
                 val_iw_samples=100,
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
        self.iw = iw
        self.iw_samples = iw_samples
        self.val_iw = val_iw
        self.val_iw_samples = val_iw_samples
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
        self.z_prior = T.ones((z_n, z_k), dtype='float32') / z_k
        pz_params = []

        # Quantization
        span = (z_k - 1.) / 2.
        self.quant_np = (np.arange(z_k, dtype=np.float32) - span) / span
        self.quant = T.constant(self.quant_np, name='quant', dtype='float32')
        print("Quantization: {}".format(self.quant_np))

        # Input
        input_x = T.fmatrix(name='input_x')  # (n, input_units)
        rnd = srng.uniform(size=input_x.shape, low=0., high=1., dtype='float32')
        input_x_binary = T.gt(input_x, rnd)  # (n, input_units)

        (train_loss,
         mean_nll_x,
         mean_kl,
         encode_updates,
         decode_updates) = self.calc_nll_tot(iw=iw,
                                             iw_samples=iw_samples,
                                             input_x_binary=input_x_binary,
                                             validation=False)

        val_loss, val_mean_nll_x, val_mean_kl, _1, _2 = self.calc_nll_tot(iw=val_iw,
                                                                          iw_samples=val_iw_samples,
                                                                          input_x_binary=input_x_binary,
                                                                          validation=True)

        # Validation function
        val_function = theano.function([input_x], [val_mean_nll_x, val_mean_kl, val_loss])
        val_headers = ['Val NLL X', 'KL', 'Val NLL']

        # Regularization
        self.params = pz_params + encoder_net.params + decoder_net.params
        reg_loss = T.constant(0.)
        if regularizer:
            for p in self.params:
                reg_loss += regularizer(p)

        # Training
        loss = train_loss + reg_loss
        train_updates = opt.get_updates(loss, self.params)
        all_updates = train_updates + iter_updates + decode_updates + encode_updates
        train_function = theano.function([input_x], [mean_nll_x, mean_kl, reg_loss, loss, self.tau],
                                         updates=fix_updates(all_updates))
        train_headers = ['NLL X', 'KL', 'Reg', 'Loss', 'Tau']
        weights = (self.params +
                   opt.weights +
                   [self.iteration] +
                   encoder_net.non_trainable_weights +
                   decoder_net.non_trainable_weights)

        # Generation
        input_n = T.iscalar()
        logitrep = T.log(self.ceps + T.repeat(T.reshape(self.z_prior, (1, z_n, z_k)), repeats=input_n, axis=0))
        zsamp = sample_one_hot(logits=logitrep, srng=srng)
        zqsamp = T.dot(zsamp, self.quant)  # (n, z_n)
        xgen, _ = self.decode(zqsamp, validation=True)
        # rnd = srng.uniform(size=xgen.shape, low=0., high=1., dtype='float32')
        # xsamp = T.cast(T.gt(xgen, rnd), 'int32')
        generate_function = theano.function([input_n], xgen)  # xsamp for binarized
        self.sample_z_function = theano.function([input_n], zqsamp)

        # Decoding
        input_zq = T.fmatrix()
        xgen, _ = self.decode(input_zq, validation=True)
        self.decode_function = theano.function([input_zq], xgen)

        # Autoencode
        # rnd = srng.uniform(low=0., high=1., dtype='float32', size=val_xpred.shape)
        # xout = T.cast(T.gt(val_xpred, rnd), dtype='float32')
        pz, z, _ = self.encode(input_x_binary, validation=True)  # (n, z_n, z_k)
        zq = T.dot(z, self.quant)
        xpred, _ = self.decode(zq, validation=True)  # (n, input_units)
        autoencode_function = theano.function([input_x], [input_x_binary, xpred])  # xout for binarized

        super(GumbelQuantizedAutoencoder, self).__init__(
            train_headers=train_headers,
            val_headers=val_headers,
            train_function=train_function,
            generate_function=generate_function,
            val_function=val_function,
            autoencode_function=autoencode_function,
            weights=weights
        )

    def __str__(self):
        ret = "{} z_n={}, z_k={}, hard={}".format(self.__class__.__name__,
                                                               self.z_n,
                                                               self.z_k,
                                                               self.hard)
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
            z = tensor_one_hot(T.argmax(logits, axis=-1), logits.shape[-1])  # (n, z_n, z_k)
        else:
            z = gumbel_softmax(logits=logits, temperature=self.tau, srng=self.srng, hard=self.hard)  # (n, z_n, z_k)
        return pz, z, updates

    def decode(self, zq, validation=False):
        assert zq.ndim == 2
        if validation:
            h = self.decoder_net.call_validation(zq)  # (n, input_units)
            updates = []
        else:
            h, updates = self.decoder_net.call(zq)  # (n, input_units)
        xpred = T.nnet.sigmoid(h)
        return xpred, updates

    def calc_p_x(self, x, xpred):
        return T.switch(x, xpred, 1. - xpred)  # (n, input_units)

    def calc_nll_x(self, x, xpred):
        return -T.sum(T.log(self.ceps + self.calc_p_x(x, xpred)), axis=1)  # (n,)

    def calc_kl(self, pz):
        # KL
        kl = T.sum(pz * (T.log(self.ceps + pz) - (T.log(self.z_prior).dimshuffle(('x', 0, 1)))),
                   axis=(1, 2))  # (n,)
        return kl

    def draw_features(self, output_path, sample_n=10):
        make_dir(output_path)
        zq = self.sample_z_function(sample_n)  # (n, z_n)
        q = self.quant_np  # (z_k,)
        for feat in range(self.z_n):
            path = '{}/feat-{}.png'.format(output_path, feat)
            zrep = np.repeat(np.reshape(zq, (sample_n, 1, self.z_n)), repeats=self.z_k, axis=1)  # (n, z_k, z_n)
            zrep[:, np.arange(self.z_k), feat] = q
            zflat = np.reshape(zrep, (sample_n * self.z_k, self.z_n))
            img = self.decode_function(zflat)
            img = np.reshape(img, (sample_n, self.z_k, 28, 28))
            img = np.transpose(img, (0, 2, 1, 3))
            img = np.reshape(img, (sample_n * 28, self.z_k * 28))
            img = (img * 255.0).astype(np.uint8)
            png.from_array(img, mode='L').save(path)

    def on_epoch_end(self, output_path, epoch, xtest):
        self.draw_features('{}/features-{}'.format(output_path, epoch))
        super(GumbelQuantizedAutoencoder, self).on_epoch_end(output_path, epoch, xtest)

    def calc_nll_tot(self, iw, iw_samples, input_x_binary, validation=False):
        if iw:
            assert iw_samples > 0
            n = input_x_binary.shape[0]
            input_dim = input_x_binary.shape[1]
            xrep = T.repeat(T.reshape(input_x_binary, (n, 1, input_dim)),
                            axis=1, repeats=iw_samples)
            xrep = T.reshape(xrep, (-1, input_dim))
            # Encode
            qz, z, encode_updates = self.encode(xrep, validation=validation)  # (n, z_n, z_k)
            zq = T.dot(z, self.quant)
            # Decode
            xpred, decode_updates = self.decode(zq, validation=validation)  # (n, input_units)
            # p(X|Z)
            p_x_given_z = T.prod(self.ceps+self.calc_p_x(xrep, xpred), axis=1)  # (n,)
            # p(Z)
            p_z = T.constant((1. / self.z_k) ** self.z_n, dtype='float32')  # scalar
            # q(Z|X)
            q_z = T.prod(self.ceps+T.sum(qz * z, axis=2), axis=1)  # (n,)
            # p(X)
            p_x = p_x_given_z * (p_z / q_z)  # (n,)
            p_x = T.reshape(p_x, (n, iw_samples))
            p_x = T.mean(p_x, axis=1)
            mean_nll_x = T.mean(-T.log(p_x_given_z))
            mean_kl = T.mean(-T.log(p_z / q_z))
            train_loss = T.mean(-T.log(p_x))
        else:
            # Encode
            pz, z, encode_updates = self.encode(input_x_binary, validation=validation)  # (n, z_n, z_k)
            zq = T.dot(z, self.quant)
            # Decode
            xpred, decode_updates = self.decode(zq, validation=validation)  # (n, input_units)
            # NLL X
            nll_x = self.calc_nll_x(input_x_binary, xpred)  # (n,)
            # KL
            kl = self.calc_kl(pz)  # (n,)
            train_loss = T.mean(kl + nll_x)
            mean_nll_x = T.mean(nll_x)  # scalar
            mean_kl = T.mean(kl)
        return train_loss, mean_nll_x, mean_kl, encode_updates, decode_updates
