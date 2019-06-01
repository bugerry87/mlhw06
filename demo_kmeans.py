#!/usr/bin/env python
'''
Author: Gerald Baulig
'''

#Global libs
from argparse import ArgumentParser
import matplotlib.pyplot as plt

#Local libs
import kernel
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
    
    parser.add_argument(
        '--mode', '-m',
        help="Choose an initialization mode.",
        choices=KMEANS_INIT_MODES,
        default=None
        )
    
    parser.add_argument(
        '--kernel', '-K',
        type=int,
        help="Choose a kernel for kernel KMeans.",
        choices=kernel.__all__,
        default=None
        )
    
    parser.add_argument(
        '--params', '-p',
        type=int,
        help="Parameters for the kernel, if required. " +
            "E.g: 'gamma=1.0, sigma=0.5'",
        default=None
        )
    
    return parser


def plot2D_kmeans(ax, X, Y, means):
    ax.scatter(X[:,0], X[:,1], s=1, c=Y)
    ax.scatter(means[:,0], means[:,1], s=100, c='r', marker='x')
    
    
def plot_convergence(ax, step, deltas):
    plot([step-1, step], deltas, c='r')
    ax.set_xlabel('step')
    ax.set_ylabel('delta')


def parse_kernel_params(arg_string):
    if arg_string:
        arg_string = arg_string.strip()
        params = args.params.split(',')
        params = [p.split('=') for p in params]
        params = {p[0]:p[1] for p in params}
        return params
    else:
        return None


def main(args):
    '''
    '''
    
    #Validate input. If not given switch to interactive mode!
    print("Validate input...")
    args.data = args.data if args.data else myinput(
        "The filename of a csv with datapoints.\n" + 
        "    data ('circle.txt'): ",
        default='circle.txt'
        )
    
    args.centers = args.centers if args.centers else myinput(
        "The number of cluster centers.\n" +
        "    centers (2): ",
        default=2,
        cast=int
        )
    
    args.epsilon = args.epsilon if args.epsilon else myinput(
        "The convergenc threshold.\n" +
        "    epsilon (0): ",
        default=0.0,
        cast=float
        )
    
    args.mode = args.mode if args.mode else myinput(
        "Choose an initialization mode.\n    > " +
        "\n    > ".join(KMEANS_INIT_MODES) +
        "\n    mode (select): ",
        default='select',
        cast=lambda m: m if m in KMEANS_INIT_MODES else raise_(ValueError("Unknown mode"))
        )
    
    args.kernel = args.kernel if args.kernel else myinput(
        "Choose a kernel for kernel KMeans.\n    > " +
        "\n    > ".join(kernel.__all__) +
        "\n    kernel (none): ",
        default='none',
        cast=lambda k: k if k in kernel.__all__ else raise_(ValueError("Unknown kernel"))
        )
    
    if args.kernel != 'none':
        args.params = args.params if args.params else myinput(
            "Parameters for the kernel, if required.\n" +
            "    params (gamma=1.0): ",
            default='gamma=1.0'
            )
        args.params = parse_kernel_params(args.params)
        kernel_func = getattr(kernel, args.kernel)
    else:
        kernel_func = None
    
    #Load data
    print("\nLoad data...")
    GT = np.genfromtxt(args.data, delimiter=',')
    
    if kernel_func:
        print("\nInit kernel with:")
        print(args.params)
        X = kernel_func(GT, GT, args.params)
    else:
        X = GT
        

    #Init KMeans
    print("\nInit means with mode: {}".format(args.mode))
    means = init_kmeans(X, args.centers, args.mode)
    
    #Run KMeans
    _, axes = arrange_subplots(2)
    axes[1].title.set_text("Convergence")
    old_delta = 0
    
    print("\nCompute KMeans...")
    for Y, means, delta, step in kmeans(X, means, args.epsilon): #Extract update steps of the generator
        print("Mean update:\n    {}".format(means))
        
        axes[0].clear()
        plot2D_kmeans(axes[0], GT, Y, means)
        axes[0].title.set_text("KMeans step: {}".format(step))
        
        plot_convergence(axes[1], step, np.sqrt([old_delta, delta]))
        old_delta = delta
        
        plt.show(block=False)
        plt.pause(0.1) #give a update pause
    
    print("Done!")
    plt.show() #stay figure
    return 0

if __name__ == '__main__':
    parser = init_argparse()
    args, _ = parser.parse_known_args()
    main(args)
    