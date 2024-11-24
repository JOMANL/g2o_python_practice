import sys
sys.path.append("../")

from matplotlib.colors import LinearSegmentedColormap

import pickle
import argparse

from typing import List
import random

from ellipse_fit.ellipse_estimater import *
from ellipse_fit.ellipse_optimizer import *
from ellipse_fit.utils import *

import g2o

import time
import pandas as pd

# experiment configulation
solver_kinds = {
    "DenceX": g2o.LinearSolverDenseX(),
    "EigenX": g2o.LinearSolverEigenX(),
    "PCGX": g2o.LinearSolverPCGX()
}

def generate_points_on_ellipse(seed = 0):

    random.seed(seed)

    num_points: int = 100

    center: np.ndarray = np.array([4, 8])
    A: float = 12
    B: float = 2
    points: List[np.array] = []

    phi = 30

    R = get_rot_mat_from_angle(phi/180*np.pi)

    for _ in range(num_points):
        a = random.gauss(A, 0.1)
        b = random.gauss(B, 0.1)
        angle = random.uniform(0.0, 2.0 * np.pi)
        points.append(center + R @ np.array([a * np.cos(angle), b * np.sin(angle)]))

    return points


def main(args):
    points = generate_points_on_ellipse(args.seed)

    max_iterations: int = args.iter
    verbose: bool = True
    # TODO: Parse from command line

    points = np.array(points)
    init_angles,init_cx,init_cy,init_A,init_B,init_phi = est_ellipse_from_points(points)
    ellipse: VertexEllipse = VertexEllipse()
    ellipse.set_id(0)
    ellipse.set_estimate([init_cx,init_cy,init_A,init_B,init_phi])  # some initial value for the circle

    # Setup the optimizer
    optimizer = g2o.SparseOptimizer()
    optimizer.add_vertex(ellipse)
    # 2. add the points we measured

    est_rot = get_rot_mat_from_angle(init_phi)

    for i,(point, angle) in enumerate(zip(points,init_angles),1):
            
        theta: VertexTheta = VertexTheta()
        theta.set_id(i)
        theta.set_estimate([angle])
        
        optimizer.add_vertex(theta)

        edge: EdgePointOnEllipse = EdgePointOnEllipse()
        edge.set_information(np.identity(2))
    
        edge.set_vertex(0, ellipse)
        edge.set_vertex(1, theta)

        edge.set_measurement(point)
        optimizer.add_edge(edge)

    print(f"Number of vertices: {len(optimizer.vertices())}")
    print(f"Number of edges: {len(optimizer.edges())}")

    datas = []
    for solver_name in solver_kinds.keys():
        solver = g2o.OptimizationAlgorithmLevenberg(g2o.BlockSolverX(solver_kinds[solver_name]))
        optimizer.set_algorithm(solver)

        # perform the optimization
        optimizer.initialize_optimization()
        optimizer.set_verbose(False)

        for i in range(max_iterations):
            print(f"Iteration {i}:")
            
            # Optimize one iteration
            start_time = time.perf_counter() # seconds
            optimizer.optimize(1)
            end_time = time.perf_counter() # seconds

            elapsed_time = end_time - start_time # seconds
            
            # Compute and print total chi^2 error
            total_chi2 = optimizer.active_chi2()
            print(f"Total chi^2 error: {total_chi2}")
            datas.append({
                "solver": solver_name,
                "iteration": i,
                "chi_error": total_chi2,
                "elapsed_time":  elapsed_time
            })

    df = pd.DataFrame(datas)
    df.to_csv(args.out)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='A simple program to greet someone.')
    parser.add_argument('--seed', type=int, default=0, help='random seed for generate ellipse sample')
    parser.add_argument('--iter', type=int, default=100, help='Max iteration')
    parser.add_argument('--out', type=str, default=0, required=True,help='save file name of csv')

    args = parser.parse_args()
    main(args)