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
        '--min_points', '-p',
        type=int,
        help="The number minimal points for cluster growing.",
        default=None
        )
    
    parser.add_argument(
        '--radius', '-r',
        nargs='*',
        type=float,
        help="The radius for cluster growing. Can be multy-dimensional.",
        default=None
        )
    
    return parser


def plot2D_dbscan(ax, X, Y, x):
    ax.scatter(X[:,0], X[:,1], s=1, c=Y)
    ax.scatter(x[0], x[1], s=100, c='r', marker='x')


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
    
    min_points = args.min_points if args.min_points else myinput(
        "The number minimal points for cluster growing.\n" +
        "    min_points (3): ",
        default=3,
        cast=int
        )
    
    radius = args.radius if args.radius else myinput(
        "The radius for cluster growing. Can be multy-dimensional.\n" +
        "    radius (0.1): ",
        default=0.1,
        cast=lambda x: np.array(x, dtype=float)
        )
    
    #Load data
    print("Load data...")
    X = np.genfromtxt(data, delimiter=',')
    
    #Run DBScan
    _, axes = arrange_subplots(1)
    
    print("Compute DBScan...")
    for Y, x, step in dbscan(X, min_points, radius): #Extract update steps of the generator
        axes[0].clear()
        plot2D_dbscan(axes[0], X, Y, x)
        axes[0].title.set_text("DBScan step: {}".format(step))
        
        plt.show(block=False)
        plt.pause(0.01) #give a update pause
    
    print("Done")
    plt.show() #stay figure


if __name__ == '__main__':
    parser = init_argparse()
    args, _ = parser.parse_known_args()
    main(args)
    