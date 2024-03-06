from imageio import imread
import numpy as np
import matplotlib.pyplot  as plt
from mpl_toolkits.axes_grid1 import AxesGrid, ImageGrid


def imshow_grid(images, shape=[2, 8]):
    """Plot images in a grid of a given shape."""
    fig = plt.figure(1)
    grid = ImageGrid(fig, 111, nrows_ncols=shape, axes_pad=0.05)

    size = shape[0] * shape[1]
    for i in range(size):
        grid[i].axis('off')
        grid[i].imshow(images[i])  # The AxesGrid object work as a list of axes.

    plt.show()


def plot_embedding(X, y, d, title=None):
    """Plot an embedding X with the class label y colored by the domain d."""
    x_min, x_max = np.min(X, 0), np.max(X, 0)
    X = (X - x_min) / (x_max - x_min)

    plt.figure(figsize=(3,3))
    ax = plt.subplot(111)
    plt.xticks([]), plt.yticks([])
    for i in range(X.shape[0]):
        plt.scatter(X[i, 0], X[i, 1],
                    c = plt.cm.bwr(d[i] / 1.),
                    marker = 'o',
                    s = 12)
        savename = title + '.png'
        plt.savefig(savename, dpi=300)


