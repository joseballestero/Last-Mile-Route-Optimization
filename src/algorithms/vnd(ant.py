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


# Vamos a modificar el código para crear la solución inicial basada en la fórmula de coste
# y asegurando el cumplimiento de las ventanas de tiempo.

vnd_updated_code_new_solution = ""
import copy
import sys
import time
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import numpy as np
import random
from models import *
from utils import *

def calculate_cost(current_location, delivery_point, alpha, beta, gamma, current_time):
    # Usamos el método distance de la clase Location
    distance = current_location.loc.distance(delivery_point.loc)

    # Si el punto de entrega tiene ventanas de tiempo
    if hasattr(delivery_point, 'time_windows'):
        time_window = delivery_point.time_windows[0]  # Para simplicidad, tomamos la primera ventana
        wait_time = max(0, time_window[0] - current_time)
    else:
        wait_time = 0  # Si no tiene ventanas de tiempo

    # Manejo de prioridad dependiendo del tipo de punto de entrega
    # Si no tiene "orders", asignamos una prioridad por defecto
    if hasattr(delivery_point, 'orders') and delivery_point.orders:
        priority = delivery_point.orders[0].priority
    else:
        priority = 1  # Prioridad por defecto si no hay órdenes o no existe el atributo

    # Fórmula de coste
    cost = alpha * distance + beta * wait_time + gamma * priority
    return cost


def find_next_delivery(current_location, candidates, alpha, beta, gamma, current_time):
    # Selecciona el siguiente punto de entrega basado en el costo calculado.
    min_cost = float('inf')
    best_candidate = None
    for delivery_point in candidates:
        cost = calculate_cost(current_location, delivery_point, alpha, beta, gamma, current_time)
        if cost < min_cost:
            min_cost = cost
            best_candidate = delivery_point
    return best_candidate

def build_initial_solution(problem: Problem, num_vehicles: int, alpha: float, beta: float, gamma: float):
    """
    Genera la solución inicial basada en la fórmula de costo.
    Los camiones salen desde el almacén (depot) y seleccionan el siguiente destino en base a la fórmula de costo.
    """
    vehicles = [{"route": [], "location": problem.depot, "time": 0} for _ in range(num_vehicles)]  # Estado inicial de cada camión
    candidates = problem.delivery_points[:]  # Lista de puntos de entrega (clientes, lockers, tiendas)
    solution = []

    for vehicle in vehicles:
        while candidates:
            next_delivery = find_next_delivery(vehicle["location"], candidates, alpha, beta, gamma, vehicle["time"])
            if next_delivery:
                # Actualizamos el estado del camión
                vehicle["route"].append(next_delivery)
                vehicle["location"] = next_delivery
                vehicle["time"] += vehicle["location"].loc.distance(next_delivery.loc)  # Actualizamos el tiempo como distancia recorrida
                
                # Verificamos la ventana de tiempo
                if hasattr(next_delivery, 'time_windows'):
                    start_window = next_delivery.time_windows[0][0]
                    if vehicle["time"] < start_window:
                        vehicle["time"] = start_window  # Esperamos hasta que se abra la ventana de tiempo
                
                # Quitamos el destino de los candidatos
                candidates.remove(next_delivery)
            else:
                break  # Si no hay un siguiente destino factible, pasamos al siguiente camión
        
        # Añadimos la ruta del camión a la solución
        solution.append(vehicle["route"])
    
    return solution


    """
    Algoritmo de búsqueda de vecindario variable (VND) que selecciona las rutas basadas en un costo heurístico.
    """
def vnd(problem, alpha, beta, gamma, num_vehicles):
    """
    Variable Neighborhood Descent (VND) algorithm.

    Parameters:
        problem : Problem
            The problem instance loaded from the file.
        alpha, beta, gamma: float
            Parameters for the cost function.
        num_vehicles: int
            Number of vehicles available.

    Returns:
        best_solution: list
            The best found solution after all iterations.
        total_cost: float
            Total cost of the best solution.
        fitness_initial: float
            Fitness of the initial solution.
        fitness_final: float
            Fitness of the final solution.
    """
    # Paso 1: Construir la solución inicial utilizando el problema cargado
    initial_solution = build_initial_solution(problem, num_vehicles, alpha, beta, gamma)

    # Paso 2: Calcular el coste inicial y el fitness
    initial_cost = calculate_total_cost(initial_solution)
    fitness_initial = evaluate_solution(problem, initial_solution, alpha, beta)

    best_solution = initial_solution
    best_cost = initial_cost
    fitness_best = fitness_initial

    improved = True
    while improved:
        improved = False
        
        # Generar vecinos utilizando Swap e Insert
        neighbors = neighbourhoodSWAP(best_solution)

        for neighbor in neighbors:
            neighbor_cost = calculate_total_cost(neighbor)
            fitness_neighbor = evaluate_solution(problem, neighbor, alpha, beta)

            # Si encontramos una solución mejor, actualizamos
            if fitness_neighbor < fitness_best:
                best_solution = neighbor
                best_cost = neighbor_cost
                fitness_best = fitness_neighbor
                improved = True

    # Devolver la mejor solución, coste, fitness inicial y final
    return best_solution, best_cost, fitness_initial, fitness_best
