#!/usr/bin/env python
'''
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
        '--epsilon', '-e',
        type=int,
        help="The convergence threshold.",
        default=None
        )
    
    return parser

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
    
    #Load data
    print("Load data...")
    X = np.genfromtxt(data, delimiter=',')
    
    L, eigval, eigvec, C = spectral(X, 2, sigma=1, mode='jordan')
    
    print(C.T)
    
    plt.scatter(X[:,0], X[:,1], s=1, c=C.T[0])
    plt.show()
    
    return 0

if __name__ == '__main__':
    parser = init_argparse()
    args, _ = parser.parse_known_args()
    main(args)