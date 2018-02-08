import matplotlib.pyplot as plt
from matplotlib import pylab
from pylab import *

def plot_PCs(components):
    plt.figure(1)
    for i in range(0, 4):
        component = components[i].reshape((sqrt(len(components)),
            sqrt(len(components))))
        plotint = 220 + i
        plt.subplot(plotint)
        plt.imshow(component, interpolation="nearest")
        plt.title(i + ". PC")
    plt.show()

def visualize_tsne(reduced_data, to_y):
    colors = []
    for y in to_y:
        c = math.floor(1*(y+0))
        colors.append(c)

    fig = plt.figure()
    ax1 = fig.add_subplot(221)
    ax1.scatter(reduced_data[:,0], reduced_data[:,1], c=colors)


    ax1.set_xlabel('1.')


    ax2 = fig.add_subplot(222)
    ax2.scatter(reduced_data[:,0], reduced_data[:,2], c=colors)


    ax2.set_xlabel('2.')
    ax2.set_ylabel('Y Label')


    ax3 = fig.add_subplot(223)
    ax3.scatter(reduced_data[:,1], reduced_data[:,2], c=colors)


    ax3.set_xlabel('X Label')
    ax3.set_ylabel('3.')

    plt.show()

def visualize_3d(reduced_data, y):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    colors = []
    for y in y:
        c = math.floor(1*(y+0))
        colors.append(c)

    ax.scatter(reduced_data[:,0], reduced_data[:,1], reduced_data[:,2], c=colors)

    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')

    plt.show()
