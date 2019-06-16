#!/usr/bin/env python
'''
Executable script to demonstrate Spectral Clustering.
Plots the Spectral Clustering clustering process stepwise.
Optionally records a MP4 video.

Author: Gerald Baulig
'''

#Global libs
from sys import argv
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
        description="Demo for clustering via Spectral Clustering",
        parents=parents
        )
    
    parser.add_argument(
        '--data', '-X',
        help="The filename of a csv with datapoints.",
        default='circle.txt'
        )
    
    parser.add_argument(
        '--bits', '-b',
        type=int,
        help="Number of bits for binary cluster mode. 0=kmeans mode",
        default=0
        )

    parser.add_argument(
        '--centers', '-k',
        type=int,
        help="The number of cluster centers for kmeans cluster mode.",
        default=2
        )
        
    parser.add_argument(
        '--gamma', '-g',
        type=int,
        help="The energy factor of similarity.",
        default=50
        )
    
    parser.add_argument(
        '--epsilon', '-e',
        type=int,
        help="The edge weight threshold for partially connected graphs. 0=fully connected graph.",
        default=0
        )
    
    parser.add_argument(
        '--laplacian_mode', '-L',
        help="The mode of how to compute Laplacian Matrix.",
        choices=LAPLACIAN_MODES,
        default=LAPLACIAN_MODES[0]
        )
    
    parser.add_argument(
        '--kmeans_mode', '-K',
        help="The mode of how to initialize KMeans.",
        choices=KMEANS_INIT_MODES,
        default=KMEANS_INIT_MODES[0]
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
    if len(argv) == 1:
        args.data = myinput(
            "The filename of a csv with datapoints.\n" + 
            "    data ({}): ".format(args.data),
            default=args.data
            )
        
        args.bits = myinput(
            "The number of bits for binary cluster mode.\n" +
            "    0=kmeans mode\n" +
            "    bits ({}): ".format(args.bits),
            default=args.bits,
            cast=int
            )
        
        if not args.bits:
            args.centers = myinput(
                "The number of cluster centers.\n" +
                "    centers ({}): ".format(args.centers),
                default=args.centers,
                cast=int
                )
            
            args.kmeans_mode = myinput(
                "Choose a KMeans initialization mode.\n    > " +
                "\n    > ".join(KMEANS_INIT_MODES) +
                "\n    kmeans_mode ({}): ".format(args.kmeans_mode),
                default=args.kmeans_mode,
                cast=lambda kmeans_mode: kmeans_mode if kmeans_mode in KMEANS_INIT_MODES else raise_(ValueError("Unknown mode"))
                )
        
        args.laplacian_mode = myinput(
            "Choose a Laplacian computation mode.\n    > " +
            "\n    > ".join(LAPLACIAN_MODES) +
            "\n    laplacian_mode ({}): ".format(args.laplacian_mode),
            default=args.laplacian_mode,
            cast=lambda laplacian_mode: laplacian_mode if laplacian_mode in LAPLACIAN_MODES else raise_(ValueError("Unknown mode"))
            ) 
        
        args.gamma = myinput(
            "The energy factor of similarity.\n" +
            "    gamma ({}): ".format(args.gamma),
            default=args.gamma,
            cast=float
            )
        
        args.epsilon = myinput(
            "The edge weight threshold for partially connected graphs.\n" +
            "    epsilon ({}): ".format(args.epsilon),
            default=args.epsilon,
            cast=float
            )
    
    print("Load data...")
    X = np.genfromtxt(args.data, delimiter=',')
    
    print("Calc spectral...")
    spectral = Spectral(X, args.gamma, args.epsilon, args.laplacian_mode)
    
    print("Run clustering...")
    fig, axes = arrange_subplots(4)
    fig.set_size_inches(7, 7)
    
    if args.bits:
        from matplotlib.widgets import Slider
        sfig, saxes = plt.subplots(10,1)
    
        def update_class(idx):
            idx = int(idx)
            axes[0].clear()
            if idx == -1:
                axes[0].title.set_text("SC gamma: {}, Clusters: {}".format(args.gamma, K))
                axes[0].scatter(X[:,0], X[:,1], s=1, c=Y, cmap='nipy_spectral')
            else:
                axes[0].title.set_text("SC gamma: {}, Class: {}".format(args.gamma, C[idx]))
                axes[0].scatter(X[:,0], X[:,1], s=1, c=Y!=C[idx], cmap='nipy_spectral')
            fig.canvas.draw_idle()
        
        Y = spectral.binary(args.bits)
        C = np.unique(Y)
        K = len(C)
        axes[0].clear()
        axes[0].scatter(X[:,0], X[:,1], s=1, c=Y, cmap='nipy_spectral')
        axes[0].title.set_text("SC gamma: {}, Clusters: {}".format(args.gamma, K))
        
        class_slider = Slider(saxes[0], 'Class', -1, K-1, valinit=-1, valstep=1)
        class_slider.on_changed(update_class)
    
    else:
        for Y, means, _, step in spectral.kmeans(args.centers, args.kmeans_mode):
            if plt.fignum_exists(fig.number):
                axes[0].clear()
                axes[0].scatter(X[:,0], X[:,1], s=1, c=Y, cmap='nipy_spectral')
                axes[0].title.set_text("SC gamma: {}, KM step: {}".format(args.gamma, step))
                plt.show(block=False)
                plt.pause(0.1)
            else:
                return 1
        
    idx = np.argsort(Y)
    W = spectral.W[idx,:]**(1/args.gamma)
    
    axes[1].imshow(W[:,idx])
    axes[1].title.set_text("Weight Matrix")
    
    axes[2].plot(range(spectral.eigval.shape[0]), spectral.eigval)
    axes[2].title.set_text("Eigenvalues")
    
    axes[3].scatter(spectral.eigvec[:,2], spectral.eigvec[:,1], s=1, c=Y, cmap='nipy_spectral')
    axes[3].title.set_text("Eigenspace 2D")
    
    print("Done!")
    plt.show()
    return 0

if __name__ == '__main__':
    parser = init_argparse()
    args, _ = parser.parse_known_args()
    main(args)