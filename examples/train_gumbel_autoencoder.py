import theano.tensor as T
from keras.optimizers import Adam
from keras.regularizers import l2

from discrete_autoencoder.gumbel_autoencoder import GumbelAutoencoder
from discrete_autoencoder.mlp import MLP
from discrete_autoencoder.mnist import mnist_data


def main():
    xtrain, xtest = mnist_data()

    output_path = 'output/gumbel_autoencoder'
    epochs = 1000
    batches = 10000
    batch_size = 64
    test_batches = 5000

    z_n = 20
    z_k = 10

    tau0 = 5.
    tau_decay = 1e-6
    tau_min = 0.2
    input_units = 28 * 28
    hard = True
    opt = Adam(1e-3)
    regularizer = l2(1e-5)

    encoder_net = MLP(input_units=input_units,
                      units=[512, 256, z_n * z_k],
                      hidden_activation=T.nnet.relu)
    decoder_net = MLP(input_units=z_n * z_k,
                      units=[256, 512, input_units],
                      hidden_activation=T.nnet.relu)
    model = GumbelAutoencoder(z_n=z_n,
                              z_k=z_k,
                              opt=opt,
                              encoder_net=encoder_net,
                              decoder_net=decoder_net,
                              regularizer=regularizer,
                              tau0=tau0,
                              tau_min=tau_min,
                              tau_decay=tau_decay,
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
