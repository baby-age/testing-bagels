import matplotlib.pyplot as plt
from matplotlib import pylab
from pylab import *

def plot_PCs(components):
    plt.figure(1)
    fst = components[0]
    fst = fst.reshape((58, 58))
    sdn = components[1]
    sdn = sdn.reshape((58, 58))
    trd = components[2]
    trd = trd.reshape((58,58))

    fourth = components[3]
    fourth = fourth.reshape((58,58))

    plt.subplot(221)
    plt.imshow(fst, interpolation="nearest")
    plt.title("1. PC")
    plt.subplot(222)
    plt.imshow(sdn, interpolation="nearest")
    plt.title("2. PC")
    plt.subplot(223)
    plt.imshow(trd, interpolation="nearest")
    plt.title("3. PC")
    plt.subplot(224)
    plt.imshow(fourth, interpolation="nearest")
    plt.title("4. PC")

    plt.show()
