import theano.tensor as T
from keras.optimizers import Adam
from keras.regularizers import l2
from theano.tensor.shared_randomstreams import RandomStreams

from discrete_autoencoder.gumbel_autoencoder import GumbelAutoencoder
from discrete_autoencoder.layers import Stack, DenseLayer, ActivationLayer, BNLayer, DropoutLayer
from discrete_autoencoder.mnist import mnist_data

leak = 0.2


# leak = 0.

def activation(x):
    return T.nnet.relu(x, leak)


def main():
    xtrain, xtest = mnist_data()

    output_path = 'output/gumbel_autoencoder_learn_pz'
    epochs = 1000
    batches = 10000
    batch_size = 128
    test_batches = 5000

    learn_pz = True
    use_mean = True
    z_n = 30
    z_k = 10
    srng = RandomStreams(123)
    tau0 = 1.
    tau_decay = 1e-5
    tau_min = 0.1
    input_units = 28 * 28
    hard = False
    opt = Adam(1e-3)
    regularizer = l2(1e-5)
    units = 256
    dropout = 0.
    encoder_net = Stack([
        DenseLayer(input_units, units),
        #BNLayer(units),
        ActivationLayer(activation),
        DropoutLayer(dropout, srng=srng),
        DenseLayer(units, units),
        #BNLayer(units),
        ActivationLayer(activation),
        DropoutLayer(dropout, srng=srng),
        DenseLayer(units, z_n * z_k)])
    decoder_net = Stack([
        DenseLayer(z_n * z_k, units),
        #BNLayer(units),
        ActivationLayer(activation),
        DropoutLayer(dropout, srng=srng),
        DenseLayer(units, units),
        #BNLayer(units),
        ActivationLayer(activation),
        DropoutLayer(dropout, srng=srng),
        DenseLayer(units, input_units)])

    model = GumbelAutoencoder(z_n=z_n,
                              z_k=z_k,
                              opt=opt,
                              encoder_net=encoder_net,
                              decoder_net=decoder_net,
                              regularizer=regularizer,
                              srng=srng,
                              tau0=tau0,
                              tau_min=tau_min,
                              tau_decay=tau_decay,
                              learn_pz=learn_pz,
                              use_mean=use_mean,
                              hard=hard)
    model.train(output_path=output_path,
                epochs=epochs,
                batches=batches,
                batch_size=batch_size,
                test_batches=test_batches,
                xtrain=xtrain,
                xtest=xtest)


if __name__ == '__main__':
    main()
