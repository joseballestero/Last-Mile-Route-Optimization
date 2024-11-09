"""Algorithm for the VRPTWDO problem"""

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


def vns(problem, initial_routes, alpha, beta, gamma, max_time_per_vehicle, penalty_weight, apply_swap=True, apply_insert=True):
    """
    Perform VNS optimization within each route in initial_routes by applying swap and/or insert moves.
    
    Parameters:
        problem : Problem
            The problem instance.
        initial_routes : list
            List of routes for each vehicle.
        alpha, beta, gamma : float
            Weights for distance, waiting time, and priority in the cost function.
        max_time_per_vehicle : int
            Maximum time each vehicle can operate.
        apply_swap, apply_insert : bool
            Flags to decide if swap or insert should be applied in the neighborhood generation.
            
    Returns:
        best_solution : list
            Best route configuration for each vehicle.
        best_cost : float
            The cost associated with the best solution found.
    """
    # Initialize the best solution as the initial routes and evaluate its cost
    best_solution = initial_routes
    if not isinstance(best_solution, list) or not all(isinstance(route, list) for route in best_solution):
        raise ValueError("Expected final_routes to be a list of lists (routes for each vehicle).")

    # Ensure initial cost evaluation is properly structured
    best_cost = evaluate_final_solution(best_solution, problem, alpha, beta, gamma)[1]
    improvement = True
    swap_count = 0

    while improvement:
        improvement = False
        neighborhoods = generate_neighborhood_within_routes(best_solution, apply_swap, apply_insert)
        len_neigh = len(neighborhoods)
        for neighbor in neighborhoods:
            # Validate each route within the neighbor to ensure they are feasible
            if all(validate_route_constraints(route, problem, max_time_per_vehicle) for route in neighbor):
                # Calculate cost only if all routes in neighbor are feasible
                cost = evaluate_final_solution(neighbor, problem, alpha, beta, gamma)[1]
                not_served_count = evaluate_final_solution(neighbor, problem, alpha, beta, gamma)[3]
                total_cost = cost + (penalty_weight * not_served_count)
                
                # Update best solution if we find a better cost
                if total_cost < best_cost:
                    best_solution = neighbor
                    best_cost = total_cost
                    improvement = True
                    swap_count += 1
                    break  # Continue with the next iteration after finding an improvement

    return best_solution, best_cost, swap_count, len_neigh


def generate_neighborhood_within_routes(routes, apply_swap, apply_insert):
    """
    Generate neighbors by applying swap and insert moves within each route.
    """
    neighborhoods = []

    for route_index, route in enumerate(routes):
        # Vecindarios específicos de la ruta actual
        route_neighbors = []

        # Aplica `swap` dentro de la ruta actual
        if apply_swap:
            route_neighbors.extend(neighbourhoodSWAP(route))
        
        # Aplica `insert` dentro de la ruta actual
        if apply_insert:
            route_neighbors.extend(neighbourhoodINSERT(route))
        
        # Para cada vecino generado en la ruta actual, reconstruye la solución completa
        for neighbor in route_neighbors:
            # Crea una nueva solución basada en el vecino de la ruta actual
            new_routes = routes[:route_index] + [neighbor] + routes[route_index+1:]
            neighborhoods.append(new_routes)

    # Verificación de que cada elemento es una lista de rutas
    for route in neighborhoods:
        if not all(isinstance(route_elem, list) for route_elem in route):
            raise ValueError("Each neighborhood route should be a list of lists")

    return neighborhoods




def validate_route_constraints(route, problem, max_time_per_vehicle):
    """
    Validate that the route meets time constraints.
    """
    total_time = 0
    for i in range(len(route) - 1):
        delivery_point = route[i][1]
        next_point = route[i + 1][1]
        
        # Calculate travel time
        travel_time = problem.dict_distance[(delivery_point, next_point)] * 60 / problem.truck_speed
        delivery_time = problem.dict_delivery_time['DEFAULT']  # Adjust if necessary for different delivery points
        total_time += travel_time + delivery_time
        
        # Check if exceeding max allowed time
        if total_time > max_time_per_vehicle:
            return False
    return True





