#!/usr/bin/env python
'''
Executable script to demonstrate Spectral Clustering.
Plots the Spectral Clustering clustering process stepwise.
Optionally records a MP4 video.

Author: Gerald Baulig
'''

#Global libs
from argparse import ArgumentParser
import matplotlib.pyplot as plt

#Local libs
from clustering import *
from utils import *


def init_argparse(parents=[]):
    ''' init_argparse(parents=[]) -> parser
    Initialize an ArgumentParser for this module.
    
    Args:
        parents: A list of ArgumentParsers of other scripts, if there are any.
        
    Returns:
        parser: The ArgumentParsers.
    '''
    parser = ArgumentParser(
        description="Demo for Clustering 2D-point-clouds via SVM",
        parents=parents
        )
    
    parser.add_argument(
        '--data', '-X',
        help="The filename of a csv with datapoints.",
        default=None
        )

    parser.add_argument(
        '--centers', '-k',
        type=int,
        help="The number of cluster centers.",
        default=None
        )
        
    parser.add_argument(
        '--gamma', '-g',
        type=int,
        help="The energy factor of similarity.",
        default=None
        )
    
    parser.add_argument(
        '--epsilon', '-e',
        type=int,
        help="The edge weight threshold for partially connected graphs.",
        default=None
        )
    
    parser.add_argument(
        '--laplacian_mode', '-L',
        type=int,
        help="The mode of how to compute Laplacian Matrix.",
        default=None
        )
    
    parser.add_argument(
        '--kmeans_mode', '-K',
        type=int,
        help="The mode of how to initialize KMeans.",
        choices=KMEANS_INIT_MODES,
        default=None
        )
    
    return parser

def main(args):
    ''' main(args) -> exit code
    The main function to execute this script.
    
    Args:
        args: The namespace object of an ArgumentParser.
    Returns:
        An exit code. (0=OK)
    '''
    
    #Validate input. If not given switch to interactive mode!
    print("Validate input...")
    data = args.data if args.data else myinput(
        "The filename of a csv with datapoints.\n" + 
        "    data ('circle.txt'): ",
        default='circle.txt'
        )
    
    centers = args.centers if args.centers else myinput(
        "The number of cluster centers.\n" +
        "    centers (2): ",
        default=2,
        cast=int
        )
    
    gamma = args.gamma if args.gamma else myinput(
        "The energy factor of similarity.\n" +
        "    gamma (50): ",
        default=50,
        cast=float
        )
    
    epsilon = args.epsilon if args.epsilon else myinput(
        "The edge weight threshold for partially connected graphs.\n" +
        "    epsilon (0): ",
        default=0.0,
        cast=float
        )
    
    laplacian_mode = args.laplacian_mode if args.laplacian_mode else myinput(
        "Choose a Laplacian computation mode.\n    > " +
        "\n    > ".join(LAPLACIAN_MODES) +
        "\n    laplacian_mode (default): ",
        default='default',
        cast=lambda laplacian_mode: laplacian_mode if laplacian_mode in LAPLACIAN_MODES else raise_(ValueError("Unknown mode"))
        )
    
    kmeans_mode = args.kmeans_mode if args.kmeans_mode else myinput(
        "Choose a KMeans initialization mode.\n    > " +
        "\n    > ".join(KMEANS_INIT_MODES) +
        "\n    kmeans_mode (select): ",
        default='select',
        cast=lambda kmeans_mode: kmeans_mode if kmeans_mode in KMEANS_INIT_MODES else raise_(ValueError("Unknown mode"))
        ) 
    
    print("Load data...")
    X = np.genfromtxt(data, delimiter=',')
    
    print("Calc spectral...")
    spectral = Spectral(X, gamma, epsilon, laplacian_mode)
    
    print("Run clustering...")
    fig, axes = arrange_subplots(4)
    fig.set_size_inches(7, 7)
    
    for Y, means, delta, step in spectral.cluster(centers, kmeans_mode):
        if plt.fignum_exists(fig.number):
            axes[0].clear()
            axes[0].scatter(X[:,0], X[:,1], s=1, c=Y)
            axes[0].title.set_text("SC gamma: {}, KM step: {}".format(gamma, step))
            plt.show(block=False)
            plt.pause(0.1)
        else:
            return 1
    
    idx = np.argsort(Y)
    W = spectral.W[idx,:]**(1/gamma)
    axes[1].imshow(W[:,idx])
    axes[1].title.set_text("Weight Matrix")
    
    axes[2].plot(range(spectral.eigval.shape[0]), spectral.eigval)
    axes[2].title.set_text("Eigenvalues")
    
    #axes[3].plot(range(spectral.eigval.shape[0]), np.flip(spectral.eigvec[idx,:centers]))
    #axes[3].title.set_text("First {} Eigenvectors".format(centers))
    axes[3].scatter(spectral.eigvec[:,2], spectral.eigvec[:,1], s=1, c=Y)
    axes[3].title.set_text("Eigenspace 2D")
    
    print("Done!")
    plt.show()
    
    return 0

if __name__ == '__main__':
    parser = init_argparse()
    args, _ = parser.parse_known_args()
    main(args)