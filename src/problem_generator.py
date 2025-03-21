# -*- coding: utf-8 -*-
"""
@author: Jose
"""

from generator import *
from models import *
from utils import *
from openpyxl import load_workbook
from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pyplot as plt

#(problem_type: str, n_orders: int, min_distance: int, tw_duration: int, n_trucks: int, t_speed:int, x_range: tuple, y_range: tuple, truck_cost: int = None, km_cost: int = None, file_name: str = 'benchmark', max_lockers: int = 5, max_shops: int = 5, plot=False):
#generate_benchmark('3R', 25, 0.1, 300, 1, 15, (0, 5), (0, 4), 70, 20, 'problem_25', 1, 1, plot=True)
generate_benchmark('3R', 50, 0.1, 300, 1, 20, (0, 3), (0, 2), 70, 1, 'problem_50', 1, 1, plot=True)
#generate_benchmark('3R', 100, 0.05, 300, 2, 15, (0, 3), (0, 2), 70, 20, 'problem_200', 5, 4, plot=True)
generate_benchmark('3R', 200, 0.05, 300, 4, 20, (0, 3), (0, 2), 70, 1, 'problem_200', 3, 5, plot=True)
generate_benchmark('3R', 500, 0.05, 300, 10, 20, (0, 4), (0, 3), 70, 1, 'problem_500', 6, 10, plot=True)
generate_benchmark('3R', 1000, 0.05, 300, 10, 20, (0, 5), (0, 4), 70, 1, 'problem_1000', 12, 20, plot=True)


problem = read_problem_file("./3R/problem_files/problem_500.txt")
problem_name = "problem"
problem.create_dictionaries()
list_opt = problem.create_list_of_options()

elbow_method(problem)


