# -*- coding: utf-8 -*-
"""
Created on Tue Oct 29 20:57:02 2024

@author: Jose
"""

from algorithms.vnd import *
from generator import *
from models import *
from utils import *
from openpyxl import Workbook
import openpyxl
from sklearn.cluster import KMeans
import numpy as np

#(problem_type: str, n_orders: int, min_distance: int, tw_duration: int, n_trucks: int, t_speed:int, x_range: tuple, y_range: tuple, truck_cost: int = None, km_cost: int = None, file_name: str = 'benchmark', max_lockers: int = 5, max_shops: int = 5, plot=False):
#generate_benchmark1('3R', 25, 0.1, 300, 1, 15, (0, 5), (0, 4), 70, 20, 'problem_25', 1, 1, plot=True)
#generate_benchmark1('3R', 50, 0.1, 300, 1, 15, (0, 5), (0, 4), 70, 20, 'problem_50', 1, 1, plot=True)
#generate_benchmark1('3R', 100, 0.05, 300, 2, 15, (0, 3), (0, 2), 70, 20, 'problem_200', 5, 4, plot=True)
generate_benchmark1('3R', 200, 0.05, 300, 4, 15, (0, 3), (0, 2), 70, 20, 'problem_200', 10, 9, plot=True)
#generate_benchmark1('3R', 500, 0.05, 300, 10, 15, (0, 4), (0, 3), 70, 20, 'problem_500', 20, 20, plot=True)
generate_benchmark1('3R', 1000, 0.05, 300, 10, 15, (0, 10), (0, 8), 70, 20, 'problem_500', 20, 20, plot=True)
#Revisar, muy pegados los puntos.


from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import numpy as np
from algorithms.vnd import *
from generator import *
from models import *
from utils import *
from openpyxl import Workbook
import openpyxl
from sklearn.cluster import KMeans
import numpy as np
import time

def elbow_method(problem, max_clusters=10, random_state=42):
    """
    Apply the elbow method to determine the optimal number of clusters for K-Means.

    Parameters:
        problem : Problem
            The problem instance containing delivery points.
        max_clusters : int
            Maximum number of clusters to evaluate.
        random_state : int
            Random state for K-Means initialization.

    Returns:
        None (Plots the elbow curve).
    """
    # Extract delivery point coordinates from problem.dict_xy
    delivery_coords = [coords for dp_id, coords in problem.dict_xy.items() if dp_id != 'DEPOT']
    delivery_coords = np.array(delivery_coords)

    # Calculate the Within-Cluster-Sum of Squares (WCSS) for different cluster numbers
    wcss = []
    for n_clusters in range(1, max_clusters + 1):
        kmeans = KMeans(n_clusters=n_clusters, random_state=random_state)
        kmeans.fit(delivery_coords)
        wcss.append(kmeans.inertia_)

    # Plot the elbow curve
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, max_clusters + 1), wcss, marker='o', linestyle='--', color='b')
    plt.title('Elbow Method for Optimal K')
    plt.xlabel('Number of Clusters (K)')
    plt.ylabel('WCSS (Within-Cluster-Sum of Squares)')
    plt.xticks(range(1, max_clusters + 1))
    plt.grid(True)
    plt.show()

# Example usage

problem = read_problem_file("./3R/problem_files/problem_500.txt")
problem_name = "problem"
problem.create_dictionaries()
list_opt = problem.create_list_of_options()

elbow_method(problem, max_clusters=10)


