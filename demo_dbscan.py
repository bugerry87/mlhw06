#!/usr/bin/env python
'''
Executable script to demonstrate DBScan.
Plots the DBScan clustering process stepwise.
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
    
    parser.add_argument(
        '--video', '-V',
        help="A filename for video record.",
        default=None
        )
    
    return parser


def plot2D_dbscan(ax, X, Y, x):
    ''' plot2D_dbscan(ax, X, Y, x)
    Plots a DBScan update step
    '''
    ax.scatter(X[:,0], X[:,1], s=1, c=Y)
    ax.scatter(x[0], x[1], s=100, c='r', marker='x')


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
    args.data = args.data if args.data else myinput(
        "The filename of a csv with datapoints.\n" + 
        "    data ('circle.txt'): ",
        default='circle.txt'
        )
    
    args.min_points = args.min_points if args.min_points else myinput(
        "The number minimal points for cluster growing.\n" +
        "    min_points (3): ",
        default=3,
        cast=int
        )
    
    args.radius = args.radius if args.radius else myinput(
        "The radius for cluster growing. Can be multy-dimensional.\n" +
        "    radius (0.1): ",
        default=0.1,
        cast=lambda x: np.array(x, dtype=float)
        )
    
    args.video = args.video if args.video else myinput(
        "The filename for video record.\n" + 
        "    video (None): ",
        default=None
        )
    
    #Load data
    print("\nLoad data...")
    X = np.genfromtxt(args.data, delimiter=',')
    
    #Run DBScan
    fig, axes = arrange_subplots(1)
    
    def plot_update(fargs):
        Y = fargs[0]
        x = fargs[1]
        step = fargs[2]
        print("Render step: {}".format(step))
        axes[0].clear()
        cax = plot2D_dbscan(axes[0], X, Y, x)
        axes[0].title.set_text("DBScan step: {}".format(step))
    
    print("\nCompute DBScan...")
    if args.video:
        import os
        import matplotlib
        import matplotlib.animation as ani
        
        dir = os.path.dirname(os.path.realpath(args.video))
        if not os.path.isdir(dir):
            os.mkdir(dir)
        
        Writer = ani.writers['ffmpeg'](fps=12, metadata=dict(artist='Gerald Baulig'), bitrate=1800)
        run_dbscan = lambda: dbscan(X, args.min_points, args.radius)
        plot_ani = ani.FuncAnimation(fig, plot_update, run_dbscan, interval=10, save_count=X.shape[0])
        plot_ani.save(args.video, writer=Writer)
        print("\nSave video to {}".format(args.video))
        return 0
    else:
        for Y, x, step in dbscan(X, args.min_points, args.radius): #Extract update steps of the generator
            if plt.fignum_exists(fig.number):
                plot_update((Y, x, step))    
                plt.show(block=False)
                plt.pause(0.01) #give a update pause
            else:
                return 1
    
        print("\nDone!")
        plt.show() #stay figure
    return 0

if __name__ == '__main__':
    parser = init_argparse()
    args, _ = parser.parse_known_args()
    main(args)
    