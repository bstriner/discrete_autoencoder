from keras.optimizers import Adam
from keras.regularizers import l2
from theano.tensor.shared_randomstreams import RandomStreams

from discrete_autoencoder.layers import Stack, DenseLayer, ActivationLayer
from discrete_autoencoder.mnist import mnist_data
from discrete_autoencoder.tensor_util import leaky_relu
from discrete_autoencoder.vae import VAE


def main():
    xtrain, xtest = mnist_data()

    output_path = 'output/vae'
    epochs = 100
    batches = 10000
    batch_size = 128
    test_batches = 5000

    z_k = 30
    srng = RandomStreams(123)
    input_units = 28 * 28
    opt = Adam(1e-3)
    regularizer = l2(1e-5)
    encoder_net = Stack([
        DenseLayer(input_units, 512),
        ActivationLayer(leaky_relu),
        DenseLayer(512, 256),
        ActivationLayer(leaky_relu),
        DenseLayer(256, z_k * 2)])
    decoder_net = Stack([
        DenseLayer(z_k, 256),
        ActivationLayer(leaky_relu),
        DenseLayer(256, 512),
        ActivationLayer(leaky_relu),
        DenseLayer(512, input_units)])

    model = VAE(
        z_k=z_k,
        opt=opt,
        encoder_net=encoder_net,
        decoder_net=decoder_net,
        regularizer=regularizer,
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
