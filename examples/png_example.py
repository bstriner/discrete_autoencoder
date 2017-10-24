

from discrete_autoencoder.mnist import mnist_data

x, _ = mnist_data()
import numpy as np
import png
img = x[0, :]
img = np.reshape(img, (28, 28))
img = np.clip(img*255.0, 0, 255).astype(np.uint8)

p = png.from_array(a=img, mode='L')

p.save('tst.png')