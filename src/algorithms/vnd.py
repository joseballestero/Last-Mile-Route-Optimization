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
    
    
    #Paso las ubicaciones como un array
    locations = []
    for order in problem.orders:
        for option in order.delivery_options:
            dp, priority = option[0], option[1]
            locations.append([dp.loc.x, dp.loc.y])
    
    destinos = np.array(locations)
    print("Número de destinos = ", len(destinos))
    
    #Utilizo la función optimal_k para encontrar el valor adecuado de k mediante el metodo de elbow inertia
    #num_clusters = optimal_k(destinos)
    num_clusters = 7
    
    # Crear clusters con K-means
    kmeans = KMeans(n_clusters=num_clusters, n_init=10, random_state=42).fit(destinos)
    
    # #Se guardan los labels para cada uno de los destinos
    labels = kmeans.labels_    
    print("Número de labels = ", len(labels))
    
    #Representar clusters
    # Convertir las ubicaciones a formato numpy para facilidad de uso con matplotlib
    coords = np.array([[point[0], point[1]] for point in destinos])
    print(f'coords shape: {coords.shape}')
    
    plot_clusters(destinos, labels)
    
    
    # Extraer los IDs y asociarlos con sus respectivos clusters
    points_with_clusters = [(point, cluster) for point, cluster in zip(problem.list_of_options, labels)]
    
    # Ordenar la lista de puntos según los clusters en orden ascendente
    points_with_clusters_sorted = sorted(points_with_clusters, key=lambda x: x[1])

    # Extraer solo los IDs de los puntos de entrega ordenados
    sorted_delivery_point_ids = [point_id for point_id, cluster in points_with_clusters_sorted]
    
    initial_solution = sorted_delivery_point_ids 
    
    
    #current_solution = get_random_solution(problem)
    
    current_solution = copy.deepcopy(initial_solution)
    #print(current_solution)    
    #Evaluar la solución inicial aleatoria    
    current_priority, distance, routes, total_time, not_served_count = eval_solution(problem, current_solution)
    
    current_cost = distance * problem.km_cost + len(routes) * problem.truck_cost
    current_solution_fitness = fitness(problem, alpha, beta, current_cost, current_priority, not_served_count)
    best_solution, best_solution_value, best_solution_fitness = copy.deepcopy(current_solution), [copy.deepcopy(current_cost), copy.deepcopy(current_priority)], copy.deepcopy(current_solution_fitness)
    
    #Se crea el objeto Solution de la solución inicial
    solution_obj2 = create_solution(problem, best_solution)
    print("Solución inicial = ", solution_obj2)
    
    #Se guardan los datos iniciales
    inicial_routes = len(routes)
    inicial_solution_fitness = best_solution_fitness
    
    
    
    print(len(problem.depot.fleet))

    iter = 0
    stats = []
    
    ContadorDoubleSWAP = 0
    contadorSWAP = 0
    contadorINSERT = 0
    bandera = 0
    iterations = 0

    while bandera == 0 and time.time() - start_time < time_limit:
        it_start = time.time()

        # Neighbourhood creation
        neighbourhood = []
        swap = []
        fitness_values_iteration = []  # new empty list to store the fitness values for this iteration
        iterations += 1
        #To create neighbourhoods
        #neighbourhood = neighbourhoodSWAP(current_solution)
        
#SWAP   
        neighbourhood = neighbourhoodSWAP(current_solution)
        contadorSWAP += 1
        print("ContadorSWAP =", contadorSWAP) 
        
        
        #Loop to assess each neighbourhood
        neighbourhood_value = []
        for i in range(len(neighbourhood)):
            priority, distance, routes, total_time, not_served_count = eval_solution(problem, neighbourhood[i])
            cost = distance * problem.km_cost + len(routes) * problem.truck_cost
            neighbourhood_value.append([cost, priority])
            fitness_values_iteration.append(fitness(problem, alpha, beta, cost, priority, not_served_count))

        # Find the best solution in the neighbourhood
        min_fitness = float('inf')
        min_solution = None
        min_solution_value = None
        
        # print(f"Neighbourhood size: {len(neighbourhood)}")
        for i in range(len(neighbourhood)):
            if fitness_values_iteration[i] < min_fitness:
                min_fitness = fitness_values_iteration[i]
                min_solution = neighbourhood[i]
                min_solution_value = neighbourhood_value[i]

        current_solution_fitness, current_solution, current_solution_value = min_fitness, min_solution, min_solution_value

#INSERT
        # Update solution
        if current_solution_fitness < best_solution_fitness:
            best_solution, best_solution_value, best_solution_fitness = copy.deepcopy(current_solution), copy.deepcopy(current_solution_value), copy.deepcopy(current_solution_fitness)
            
        else:
           neighbourhood = neighbourhoodINSERT(best_solution)
           contadorINSERT += 1
           print("ContadorINSERT =", contadorINSERT)
            
           
           #Loop to assess each neighbourhood
           neighbourhood_value = []
           for i in range(len(neighbourhood)):
               priority, distance, routes, total_time, not_served_count = eval_solution(problem, neighbourhood[i])
               cost = distance * problem.km_cost + len(routes) * problem.truck_cost
               neighbourhood_value.append([cost, priority])
               fitness_values_iteration.append(fitness(problem, alpha, beta, cost, priority, not_served_count))

           # Find the best solution in the neighbourhood
           min_fitness = float('inf')
           min_solution = None
           min_solution_value = None

           # print(f"Neighbourhood size: {len(neighbourhood)}")
           for i in range(len(neighbourhood)):
               if fitness_values_iteration[i] < min_fitness:
                   min_fitness = fitness_values_iteration[i]
                   min_solution = neighbourhood[i]
                   min_solution_value = neighbourhood_value[i]

           current_solution_fitness, current_solution, current_solution_value = min_fitness, min_solution, min_solution_value

#DoubleSWAP
            # Update solution
           if current_solution_fitness < best_solution_fitness:
                best_solution, best_solution_value, best_solution_fitness = copy.deepcopy(current_solution), copy.deepcopy(current_solution_value), copy.deepcopy(current_solution_fitness)
    
           else:
               
               neighbourhood = neighbourhoodDoubleSWAP(best_solution)
               ContadorDoubleSWAP += 1
               print("ContadorDoubleSWAP =", ContadorDoubleSWAP)

               
              
               #Loop to assess each neighbourhood
               neighbourhood_value = []
               for i in range(len(neighbourhood)):
                   priority, distance, routes, total_time, not_served_count = eval_solution(problem, neighbourhood[i])
                   cost = distance * problem.km_cost + len(routes) * problem.truck_cost
                   neighbourhood_value.append([cost, priority])
                   fitness_values_iteration.append(fitness(problem, alpha, beta, cost, priority, not_served_count))
    
               # Find the best solution in the neighbourhood
               min_fitness = float('inf')
               min_solution = None
               min_solution_value = None
    
               # print(f"Neighbourhood size: {len(neighbourhood)}")
               for i in range(len(neighbourhood)):
                   if fitness_values_iteration[i] < min_fitness:
                       min_fitness = fitness_values_iteration[i]
                       min_solution = neighbourhood[i]
                       min_solution_value = neighbourhood_value[i]
    
               current_solution_fitness, current_solution, current_solution_value = min_fitness, min_solution, min_solution_value
    
    
               # Update solution
               if current_solution_fitness < best_solution_fitness:
                   best_solution, best_solution_value, best_solution_fitness = copy.deepcopy(current_solution), copy.deepcopy(current_solution_value), copy.deepcopy(current_solution_fitness)
                   
               else:
                   bandera = 1

        
           
        # Calculate the mean, minimum, and maximum fitness values for this iteration using numpy functions
        fitness_values_iteration = fitness_values_iteration[fitness_values_iteration != float('inf')]
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
    print("ContadorSWAP =", contadorSWAP)
    print("ContadorINSERT =", contadorINSERT)
    
    return best_solution, best_solution_value, best_solution_fitness, stats, execution_time, inicial_routes, inicial_solution_fitness, iter


def __get_option_id(option: tuple):
    """
    Returns the id of a delivery option.

    Parameters:
        option : tuple
            ((DeliveryPoint | PersonalPoint, priority), Order) tuple.

    Returns:
        option_id : str
            Id of the delivery option.
    """

    dp_id = id(option[0][0].id)
    order_id = id(option[1].id)
    return f"{dp_id}-{order_id}"



# Elbow method to determine optimal k
def optimal_k(destinos, max_k=10):
    
    inertia = []
    for k in range(1, max_k + 1):
        kmeans = KMeans(n_clusters=k, random_state=42).fit(destinos)
        inertia.append(kmeans.inertia_)
    
    plt.plot(range(1, max_k + 1), inertia, 'bx-')
    plt.xlabel('Number of clusters (k)')
    plt.ylabel('Inertia')
    plt.title('Elbow Method for Optimal k')
    plt.show()

    # Elige k basándote en el gráfico
    return int(input("Enter the optimal number of clusters (k): "))


def two_opt(route, problem):
    def calculate_distance(route):
        total_distance = 0
        for i in range(len(route) - 1):
            total_distance += problem.dict_distance[(route[i][1], route[i+1][1])]
        return total_distance

    best_route = route
    best_distance = calculate_distance(route)
    
    improved = True
    while improved:
        improved = False
        for i in range(1, len(route) - 2):
            for j in range(i + 1, len(route)):
                if j - i == 1: continue
                new_route = route[:i] + route[i:j][::-1] + route[j:]
                new_distance = calculate_distance(new_route)
                if new_distance < best_distance:
                    best_route = new_route
                    best_distance = new_distance
                    improved = True
        route = best_route
    return best_route

def plot_clusters(destinos, labels):
    plt.scatter(destinos[:, 0], destinos[:, 1], c=labels, s=50, cmap='viridis')
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.title('Clusters of Delivery Points')
    plt.show()

def generate_initial_solution(problem, num_clusters):
    coords = np.array([[point.x, point.y] for point in problem.dict_xy.values()])
    kmeans = KMeans(n_clusters=num_clusters, random_state=0).fit(coords)
    labels = kmeans.labels_

    clusters = {i: [] for i in range(num_clusters)}
    for point, label in zip(problem.list_of_options, labels):
        clusters[label].append(point)

    initial_solution = []
    for cluster_id, points in clusters.items():
        sorted_points = sorted(points, key=lambda x: x[0].twe)
        route = []
        current_capacity = problem.truck_capacity
        for point in sorted_points:
            if current_capacity > 0:
                route.append((point[1].id, point[0].id))
                current_capacity -= 1
            else:
                initial_solution.append(route)
                route = [(point[1].id, point[0].id)]
                current_capacity = problem.truck_capacity - 1
        if route:
            initial_solution.append(route)

    return initial_solution

def evaluate_and_optimize_routes(problem, initial_solution):
    optimized_solution = []
    total_distance = 0
    total_priority = 0

    for route in initial_solution:
        optimized_route = two_opt(route, problem)
        priority, distance, _, _, _ = eval_solution(problem, optimized_route)
        optimized_solution.append(optimized_route)
        total_distance += distance
        total_priority += priority

    return optimized_solution, total_distance, total_priority

