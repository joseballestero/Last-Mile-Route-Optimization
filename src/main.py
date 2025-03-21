"""
@author: Jose Ballestero de Juan
"""

from generator import *
from models import *
from utils import *
from openpyxl import Workbook
import openpyxl
from sklearn.cluster import KMeans
import numpy as np
import time
import pandas as pd


# Limitar el número de threads para evitar advertencias de MKL
os.environ["OMP_NUM_THREADS"] = "3"
os.environ["MKL_NUM_THREADS"] = "1"

import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

# Almacena el tiempo de inicio
start_time = time.time()

# Read problem from file

#problem = read_problem_file("./problem_files/3H_100/txt/test_3.txt")
problem = read_problem_file("./3R/problem_files/problem_1000.txt")
problem_name = "problem"
problem.create_dictionaries()
list_opt = problem.create_list_of_options()

# Parámetros del algoritmo
alpha = 1  # Peso de la distancia/coste
beta = 0.1   # Peso del tiempo de espera
gamma = 1  # Peso de la prioridad
num_vehicles = 11  # Número de camiones
max_time_per_vehicle = 480

# Aplicar K-Means clustering para dividir las entregas entre los camiones
clustered_solution = apply_kmeans(problem, num_vehicles, alpha, beta, gamma, max_time_per_vehicle, problem_name)


# delivery_coords = []
# delivery_points = []

# # Collect the coordinates and delivery points for clustering
# for order in problem.orders:
#     for option in order.delivery_options:
#         dp = option[0]  # Delivery point
#         delivery_coords.append([dp.loc.x, dp.loc.y])
#         delivery_points.append((order.id, dp.id))

# # Convert delivery coordinates to numpy array for K-Means
# delivery_coords = np.array(delivery_coords)
# elbow_method(delivery_coords, 10)

# Asignar las entregas a los vehículos y obtener las rutas finales
final_routes, total_waiting_time, current_storage = assign_deliveries_to_vehicles(clustered_solution, problem, num_vehicles, max_time_per_vehicle, alpha, beta, gamma)
#print("Current storage = ", current_storage)

# Evaluar la solución completa para K-Means
# Imprimir las rutas finales
# for vehicle_id, route in enumerate(final_routes):
#     print(f"Ruta del Vehículo {vehicle_id}: {route}")
print("==== Evaluando la solución generada por K-Means ====")

# Evaluar la solución final
normalized_priority, total_cost, total_priority, not_served_count, total_time, total_distance = evaluate_final_solution(final_routes, problem, alpha, beta, gamma)

# Imprimir los resultados finales
print(f"\nResultados finales de la evaluación:")
print(f"Coste total: {total_cost}")
print(f"Prioridad total: {total_priority}")
print(f"Prioridad normalizada: {normalized_priority}")
print(f"Órdenes no servidas: {not_served_count}")
print(f"Tiempo total: {total_time} min")
print(f"Distancia total: {total_distance} km")
print(f"Tiempo de espera total: {total_waiting_time} min")

# Compute delivery times using problem's real data
exact_delivery_times = calculate_delivery_times(final_routes, problem)

# Convert results to DataFrame for review
df_delivery_times = pd.DataFrame(
    [entry for vehicle in exact_delivery_times for entry in vehicle],
    columns=["Vehicle", "Order ID", "Location", "Arrival Time", "End Time"]
)


# Plot the Gantt chart
plot_gantt_chart(exact_delivery_times)

best_solution = final_routes
# Gráfico de rutas finales
plot_vehicle_routes(problem, best_solution)



