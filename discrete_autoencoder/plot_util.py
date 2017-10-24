import matplotlib.pyplot as plt

from .util import make_path


def write_image(img, outputpath, figsize=None, cmap='gray'):
    make_path(outputpath)
    fig = plt.figure(figsize=figsize)
    plt.imshow(img, cmap=cmap)
    plt.axis('off')
    fig.savefig(outputpath)
    plt.close(fig)
