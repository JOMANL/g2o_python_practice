
import pickle
import argparse

import glob
import os.path as osp

import matplotlib.pyplot as plt

def main(args):

    pkls = [tmp for tmp in glob.glob(args.dir + "/*.pkl")]

    fig, ax = plt.subplots()

    for pkl in pkls:

        name = osp.basename(pkl).split(".")[0]

        with open(pkl, "rb") as f:
            data = pickle.load(f)

        ax.plot(data,label=name)

    ax.set_xlabel("Iteration")
    ax.set_ylabel("Chi^2 error")
    plt.legend(fontsize=14)
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='plot optimization step and errors')
    parser.add_argument('--dir', type=str, default=0, required=True,help='path where exp_solpver.py outputs were saved')

    args = parser.parse_args()
    main(args)