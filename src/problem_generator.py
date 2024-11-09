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

#(problem_type: str, n_orders: int, min_distance: int, tw_duration: int, n_trucks: int, x_range: tuple, y_range: tuple, truck_cost: int = None, km_cost: int = None, file_name: str = 'benchmark', max_lockers: int = 5, max_shops: int = 5, plot=False):
generate_benchmark1('3R', 25, 0.2, 180, 2, 15, (0, 2), (0, 2), 70, 10, 'problem_25', 2, 2, plot=True)
#generate_benchmark1('3R', 500, 10, 300, 10, (0, 500), (0, 500), 70, 20, 'problem_500', 20, 20, plot=False)
#generate_benchmark1('3R', 50, 5, 180, 4, (0, 100), (0, 100), 70, 20, 'problem_50', 6, 5, plot=True)
#generate_benchmark1('3R', 200, 10, 250, 7, (0, 600), (0, 600), 70, 20, 'problem_200', 14, 13, plot=True)
#Revisar, muy pegados los puntos.