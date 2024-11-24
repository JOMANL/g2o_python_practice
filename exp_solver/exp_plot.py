
import pandas as pd
import matplotlib.pyplot as plt

import argparse

def main(args):
    df = pd.read_csv(args.csv,
                    names=["solver","iteration","chi_error","elapsed_time"],
                    skiprows=1)

    fig, ax = plt.subplots(1,1)

    grouped = df.groupby("solver")["elapsed_time"]
    mean_values = grouped.mean()
    std_dev = grouped.std()

    mean_values.plot(kind='bar', yerr=std_dev, capsize=4, color='skyblue', alpha=0.8, ax = ax)

    plt.xlabel('Linear Solver', fontsize = 12)
    plt.ylabel('elapsed_time [sec]', fontsize = 12)
    plt.tight_layout()
    plt.savefig(args.out)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='A simple program to greet someone.')
    parser.add_argument('--csv', type=str, required=True, help='save file name of csv')
    parser.add_argument('--out', type=str, required=True, help='save file name of graph (png)')

    args = parser.parse_args()
    main(args)