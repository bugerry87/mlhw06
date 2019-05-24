#!/usr/bin/env python
'''
'''

#Global libs
from argparse import ArgumentParser
import numpy as np
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
        '--epsilon', '-e',
        type=int,
        help="The convergence threshold.",
        default=None
        )
    
    return parser


def plot2D_kmeans(ax, X, Y, means):
    ax.scatter(X[:,0], X[:,1], s=1, c=Y)
    ax.scatter(means[:,0], means[:,1], s=100, c='r', marker='x')


def main(args):
    '''
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
    
    epsilon = args.epsilon if args.epsilon else myinput(
        "The convergenc threshold.\n" +
        "    epsilon (0): ",
        default=0,
        cast=float
        )
    
    #Load data
    print("Load data...")
    X = np.genfromtxt(data, delimiter=',')    
    means = X[np.random.choice(range(X.shape[0]), centers),:] #Random select a point of the dataset
    
    #Run KMeans
    _, axes = arrange_subplots(2)
    axes[1].title.set_text("Convergence")
    old_delta = 0
    
    print("Compute KMeans...")
    for i, (Y, means, delta) in enumerate(kmeans(X, means)): #Extract update steps of the generator
        print("Mean update:\n    {}".format(means))
        
        axes[0].clear()
        plot2D_kmeans(axes[0], X, Y, means)
        axes[0].title.set_text("KMeans step: {}".format(i))
        
        axes[1].plot([i-1, i], [old_delta, delta], c='r')
        old_delta = delta
        
        plt.show(block=False)
        plt.pause(0.1) #give a update pause
    
    print("Done")
    plt.show() #stay figure


if __name__ == '__main__':
    parser = init_argparse()
    args, _ = parser.parse_known_args()
    main(args)
    