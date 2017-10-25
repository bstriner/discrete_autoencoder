import csv
import os

import numpy as np
import png
from tqdm import tqdm

from .tensor_util import load_latest_weights, save_weights


class DiscreteAutoencoder(object):
    def __init__(self,
                 train_headers,
                 val_headers,
                 generate_function,
                 train_function,
                 val_function,
                 z_prior_function,
                 weights):
        self.train_headers = train_headers
        self.val_headers = val_headers
        self.generate_function = generate_function
        self.train_function = train_function
        self.val_function = val_function
        self.z_prior_function = z_prior_function
        self.weights = weights

    def generate(self, output_path, n):
        if self.generate_function:
            img = self.generate_function(n)
            img = np.reshape(img, (n, 28, 28))
            img = np.reshape(img, (n * 28, 28))
            img = (img * 255.0).astype(np.uint8)
            png.from_array(img, mode='L').save(output_path)

    def train(self,
              output_path,
              epochs,
              batches,
              batch_size,
              test_batches,
              xtrain,
              xtest):
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        initial_epoch = load_latest_weights(output_path, r'model-(\d+).h5', self.weights)
        if initial_epoch < epochs:
            with open(os.path.join(output_path,'summary.txt'), 'w') as f:
                f.write(str(self)+"\n")
            with open(os.path.join(output_path, 'history.csv'), 'ab') as f:
                w = csv.writer(f)
                w.writerow(['Epoch'] + self.train_headers + self.val_headers)
                f.flush()
                for epoch in tqdm(range(initial_epoch, epochs), desc='Training'):
                    it = tqdm(range(batches), desc='Epoch {}'.format(epoch))
                    data = [[] for _ in range(len(self.train_headers))]
                    for _ in it:
                        idx = np.random.random_integers(low=0, high=xtrain.shape[0] - 1, size=(batch_size,))
                        x = xtrain[idx, :]
                        datum = self.train_function(x)
                        for i in range(len(self.train_headers)):
                            data[i].append(datum[i])
                        scalars = [np.asscalar(d) for d in datum]
                        desc = 'Epoch {}'.format(epoch)
                        for d, v in zip(self.train_headers, scalars):
                            desc += ' {} {:.04f}'.format(d, v)
                        it.desc = desc

                    val_data = [[] for _ in range(len(self.val_headers))]
                    for _ in tqdm(range(test_batches), desc='Validating Epoch {}'.format(epoch)):
                        idx = np.random.random_integers(low=0, high=xtest.shape[0] - 1, size=(batch_size,))
                        x = xtest[idx, :]
                        val_datum = self.val_function(x)
                        for i in range(len(self.val_headers)):
                            val_data[i].append(val_datum[i])

                    data = [np.asscalar(np.mean(d)) for d in data]
                    val_means = [np.asscalar(np.mean(d)) for d in val_data]
                    w.writerow([epoch] + data + val_means)
                    f.flush()

                    self.generate('{}/generated-{:08d}.png'.format(output_path, epoch), n=20)
                    if self.z_prior_function:
                        np.savetxt('{}/pz-{:08d}.txt'.format(output_path, epoch), self.z_prior_function())
                    save_weights('{}/model-{:08d}.h5'.format(output_path, epoch), self.weights)
