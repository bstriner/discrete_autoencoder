# import os
# os.environ["THEANO_FLAGS"]="optimizer=None,device=cpu"

from keras.optimizers import Adam
from keras.regularizers import l2
from theano.tensor.shared_randomstreams import RandomStreams

from discrete_autoencoder.gumbel_quantized_autoencoder import GumbelQuantizedAutoencoder
from discrete_autoencoder.layers import Stack, DenseLayer, ActivationLayer
from discrete_autoencoder.mnist import mnist_data
from discrete_autoencoder.tensor_util import leaky_relu


def main():
    xtrain, xtest = mnist_data()

    output_path = 'output/gumbel_quantized_autoencoder_iw'
    epochs = 1000
    batches = 5000
    batch_size = 128
    test_batches = 5000

    learn_pz = False
    z_n = 30
    z_k = 5
    srng = RandomStreams(123)
    tau0 = 1.
    tau_decay = 1e-5
    tau_min = 0.1
    input_units = 28 * 28
    hard = False
    opt = Adam(1e-3)
    regularizer = l2(1e-5)
    units = 256
    iw = True
    iw_samples = 20
    val_iw = True
    val_iw_samples = 100
    encoder_net = Stack([
        DenseLayer(input_units, units),
        ActivationLayer(leaky_relu),
        DenseLayer(units, units),
        ActivationLayer(leaky_relu),
        DenseLayer(units, z_n * z_k)])
    decoder_net = Stack([
        DenseLayer(z_n, units),
        ActivationLayer(leaky_relu),
        DenseLayer(units, units),
        ActivationLayer(leaky_relu),
        DenseLayer(units, input_units)])

    model = GumbelQuantizedAutoencoder(z_n=z_n,
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
                                       iw=iw,
                                       iw_samples=iw_samples,
                                       val_iw=val_iw,
                                       val_iw_samples=val_iw_samples,
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
