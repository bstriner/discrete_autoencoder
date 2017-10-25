import theano.tensor as T
from keras.optimizers import Adam
from keras.regularizers import l2
from theano.tensor.shared_randomstreams import RandomStreams

from discrete_autoencoder.layers import Stack, DenseLayer, ActivationLayer, DropoutLayer
from discrete_autoencoder.mnist import mnist_data
from discrete_autoencoder.reinforce_autoencoder import ReinforceAutoencoder


def activation(x):
    return T.nnet.relu(x, 0.2)


def main():
    xtrain, xtest = mnist_data()

    output_path = 'output/reinforce_autoencoder'
    epochs = 1000
    batches = 10000
    batch_size = 64
    test_batches = 5000

    z_n = 30
    z_k = 10
    srng = RandomStreams(123)
    input_units = 28 * 28
    opt = Adam(3e-4)
    regularizer = l2(1e-4)
    entropy_weight = 1.
    units = 512
    dropout = 0.5
    encoder_net = Stack([
        DenseLayer(input_units, units),
        # BNLayer(512),
        ActivationLayer(activation),
        DropoutLayer(dropout, srng=srng),
        DenseLayer(units, units),
        # BNLayer(256),
        ActivationLayer(activation),
        DropoutLayer(dropout, srng=srng),
        DenseLayer(units, z_n * z_k)])
    decoder_net = Stack([
        DenseLayer(z_n * z_k, units),
        # BNLayer(256),
        ActivationLayer(activation),
        DropoutLayer(dropout, srng=srng),
        DenseLayer(units, units),
        # BNLayer(512),
        ActivationLayer(activation),
        DropoutLayer(dropout, srng=srng),
        DenseLayer(units, input_units)])

    model = ReinforceAutoencoder(z_n=z_n,
                                 z_k=z_k,
                                 opt=opt,
                                 encoder_net=encoder_net,
                                 decoder_net=decoder_net,
                                 regularizer=regularizer,
                                 entropy_weight=entropy_weight,
                                 srng=srng)
    model.train(output_path=output_path,
                epochs=epochs,
                batches=batches,
                batch_size=batch_size,
                test_batches=test_batches,
                xtrain=xtrain,
                xtest=xtest)


if __name__ == '__main__':
    main()
