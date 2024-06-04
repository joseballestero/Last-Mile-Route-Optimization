""" Tabu search algorithm for the VRPTWDO problem"""

import copy
import sys
import time

sys.path.append("..")
from models import *
from utils import *
from algorithms.neighbourhood_search import *
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import numpy as np
import random


def vnd(problem: Problem, alpha: float, beta: float, time_limit: int = float('inf'), log: bool = False, initial_solution: list = None):
    """
    Returns the best solution found by the tabu search algorithm.

    Parameters:
        problem : Problem
            Problem instance.
        iterations : int
            Number of iterations.
        alpha : float
            Alpha parameter.
        beta : float
            Beta parameter.
        log : bool
            If True, prints the execution time.
        initial_solution : list
            Initial solution as a vector. List of ((DeliveryPoint | PersonalPoint, priority), Order) tuples.

    Returns:
        best_solution : list
            Vector solution. List of ((DeliveryPoint | PersonalPoint, priority), Order) tuples.
        best_cost : float
            Total cost of the best solution.
        best_priority : float
            Total priority of the best solution.
    """

    start_time = time.time()
    
    
    # Convert locations XY into an array
    locations = convert_locations(problem)
    print("Número de destinos = ", len(locations))
    
    # Use the optimal_k function to find the appropriate value of k using the elbow inertia method
    #num_clusters = optimal_k(locations)
    num_clusters = 7
    
    # Create clusters with K-means
    kmeans = KMeans(n_clusters=num_clusters, n_init=10, random_state=42).fit(locations)
    
    # Save the labels for each of the destinations
    labels = kmeans.labels_    
    print("Número de labels = ", len(labels))
    
    # Represent clusters
    # Convert locations to numpy format for easy use with matplotlib
    coords = np.array([[point[0], point[1]] for point in locations])
    print(f'coords shape: {coords.shape}')
    
    plot_clusters(locations, labels)
    
    
    # Extract the IDs and associate them with their respective clusters
    points_with_clusters = [(point, cluster) for point, cluster in zip(problem.list_of_options, labels)]

    # Sort the list of points according to clusters in ascending order
    points_with_clusters_sorted = sorted(points_with_clusters, key=lambda x: x[1])

    # Extract only the IDs of the delivery points sorted
    sorted_delivery_point_ids = [point_id for point_id, cluster in points_with_clusters_sorted]
    initial_solution = sorted_delivery_point_ids 
    
    # Randomin initial solution
    #current_solution = get_random_solution(problem)
    
    
    current_solution = copy.deepcopy(initial_solution)
       
    # Evaluate the initial random solution 
    current_priority, distance, routes, total_time, not_served_count = eval_solution(problem, current_solution)
    current_cost = distance * problem.km_cost + len(routes) * problem.truck_cost
    current_solution_fitness = fitness(problem, alpha, beta, current_cost, current_priority, not_served_count)
    best_solution, best_solution_value, best_solution_fitness = copy.deepcopy(current_solution), [copy.deepcopy(current_cost), copy.deepcopy(current_priority)], copy.deepcopy(current_solution_fitness)
    
    # Create the Solution object of the initial solution
    solution_obj2 = create_solution(problem, best_solution)
    print("Solución inicial = ", solution_obj2)
    
    # Save the initial data
    inicial_routes = len(routes)
    inicial_solution_fitness = best_solution_fitness
    
    # Loop for the VND search
    iter = 0
    stats = []
    
    ContadorDoubleSWAP = 0
    countSwap = 0
    countInsert = 0
    bandera = 0
    iterations = 0

    while bandera == 0 and time.time() - start_time < time_limit:
        it_start = time.time()

        # Neighbourhood creation
        neighbourhood = []
        swap = []
        fitness_values_iteration = []  # new empty list to store the fitness values for this iteration
        iterations += 1
        new_fitness = 0
        neighbourhood_new_value = []
        out = 0
        #To create neighbourhoods
        #neighbourhood = neighbourhoodSWAP(current_solution)
        
#SWAP search   
        neighbourhood = neighbourhoodSWAP(current_solution)
        countSwap += 1
        print("ContadorSWAP =", countSwap) 
        
        
        #Loop to assess each neighbourhood
        neighbourhood_new_value = []
        for i in range(len(neighbourhood)):
            priority, distance, routes, total_time, not_served_count = eval_solution(problem, neighbourhood[i])
            cost = distance * problem.km_cost + len(routes) * problem.truck_cost
            neighbourhood_new_value = [cost, priority]
            fitness_values_iteration.append(fitness(problem, alpha, beta, cost, priority, not_served_count))
            #print("Fitness =", fitness_values_iteration)
            
            if fitness_values_iteration[i] < best_solution_fitness:
                min_fitness = fitness_values_iteration[i]
                min_solution = neighbourhood[i]
                min_solution_value = neighbourhood_new_value
                current_solution_fitness, current_solution, current_solution_value = min_fitness, min_solution, min_solution_value
                best_solution, best_solution_value, best_solution_fitness = copy.deepcopy(current_solution), copy.deepcopy(current_solution_value), copy.deepcopy(current_solution_fitness)
                out = 1
                break

           

#INSERT search
        if out == 0:
           neighbourhood = neighbourhoodINSERT(best_solution)
           countInsert += 1
           print("ContadorINSERT =", countInsert)
            
           
           #Loop to assess each neighbourhood
           neighbourhood_value = []
           for i in range(len(neighbourhood)):
               priority, distance, routes, total_time, not_served_count = eval_solution(problem, neighbourhood[i])
               cost = distance * problem.km_cost + len(routes) * problem.truck_cost
               neighbourhood_new_value = [cost, priority]
               fitness_values_iteration.append(fitness(problem, alpha, beta, cost, priority, not_served_count))
               
               
               if fitness_values_iteration[i] < best_solution_fitness:
                   min_fitness = fitness_values_iteration[i]
                   min_solution = neighbourhood[i]
                   min_solution_value = neighbourhood_new_value
                   current_solution_fitness, current_solution, current_solution_value = min_fitness, min_solution, min_solution_value
                   best_solution, best_solution_value, best_solution_fitness = copy.deepcopy(current_solution), copy.deepcopy(current_solution_value), copy.deepcopy(current_solution_fitness)
                   break

               else:
                bandera = 1

        
           
        # Calculate the mean, minimum, and maximum fitness values for this iteration using numpy functions
        #fitness_values_iteration = fitness_values_iteration[fitness_values_iteration != float('inf')]
        mean_fitness = np.mean(fitness_values_iteration)
        min_fitness = np.min(fitness_values_iteration)
        max_fitness = np.max(fitness_values_iteration)

        if log: print(f"Iteration {iter} fitness: avg = {mean_fitness:.4f}, max = {max_fitness:.4f}, min = {min_fitness:.4f}, best = {best_solution_fitness:.4f}")

        # Append the mean, minimum, and maximum fitness values to the stats list as a dictionary
        stats.append({'avg': mean_fitness, 'min': best_solution_fitness, 'max': max_fitness})

        iter += 1
        iteration_time = time.time() - it_start
        if log: print(f"Iteration time: {iteration_time:.4f} seconds")

    execution_time = time.time() - start_time
    if log: print(f"Execution time: {execution_time:.4f} seconds")
    
    print("Prioridad = ", priority)
    print("Not_served_count =", not_served_count)
    print("ContadorDoubleSWAP =", ContadorDoubleSWAP)
    print("ContadorSWAP =", countSwap)
    print("ContadorINSERT =", countInsert)
    
    return best_solution, best_solution_value, best_solution_fitness, stats, execution_time, inicial_routes, inicial_solution_fitness, iter, countSwap, countInsert







