""" Tabu search algorithm for the VRPTWDO problem"""

import copy
import sys
import time

sys.path.append("..")
from models import *
from utils import *
from algorithms.neighbourhood_search import *


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

    # Initial solution
    current_solution = get_random_solution(problem)
    
    if initial_solution:
        current_solution = copy.deepcopy(initial_solution)
        
    #Evaluar la solución inicial aleatoria    
    current_priority, distance, routes, total_time, not_served_count = eval_solution(problem, current_solution)
    
    current_cost = distance * problem.km_cost + len(routes) * problem.truck_cost
    current_solution_fitness = fitness(problem, alpha, beta, current_cost, current_priority, not_served_count)
    best_solution, best_solution_value, best_solution_fitness = copy.deepcopy(current_solution), [copy.deepcopy(current_cost), copy.deepcopy(current_priority)], copy.deepcopy(current_solution_fitness)
    
    #Se crea el objeto Solution de la solución inicial
    solution_obj2 = create_solution(problem, best_solution)
    
    #Se guardan los datos iniciales
    inicial_routes = len(routes)
    inicial_solution_fitness = best_solution_fitness
    
    #Se imprime solución inicial
    print(solution_obj2)
    print(len(problem.depot.fleet))

    iter = 0
    stats = []
    
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
        neighbourhood = neighbourhoodSWAP(current_solution)
        contadorSWAP += 1

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
           neighbourhood = neighbourhoodINSERT(current_solution)
           contadorINSERT += 1
           
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






