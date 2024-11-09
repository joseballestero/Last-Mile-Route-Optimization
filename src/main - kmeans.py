from algorithms.vnd import *
from generator import *
from models import *
from utils import *
from openpyxl import Workbook
import openpyxl
from sklearn.cluster import KMeans
import numpy as np


# Limitar el número de threads para evitar advertencias de MKL
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

# Read problem from file

#problem = read_problem_file("./problem_files/3H_100/txt/test_3.txt")
problem = read_problem_file("./3R/problem_files/problem_500.txt")
problem_name = "problem"
problem.create_dictionaries()
list_opt = problem.create_list_of_options()

#Read excel file
name = "Results.xlsx"
excel_document = openpyxl.load_workbook(name, data_only=True)

max_shops = 5
max_lockers = 5
#Este funciona - generate_benchmark1(problem_type: str, n_orders: int, min_distance: int, tw_duration: int, n_trucks: int, x_range: tuple, y_range: tuple, truck_cost: int = None, km_cost: int = None, file_name: str = 'benchmark', max_lockers: int = 5, max_shops: int = 5, plot=False):
#generate_benchmark1('3R', 100, 10, 300, 5, (0,200), (0,200), 70, 5, 'problem', max_lockers, max_shops, plot=False)
#generate_benchmark1('3R', 500, 10, 300, 10, (0, 500), (0, 500), 70, 20, 'problem_500', 20, 20, plot=False)


#generate_problem_file(100, 3, 10, (0, 6), (0, 6), 'problem', plot=False)

# Parámetros del algoritmo
alpha = 0.7  # Peso de la distancia
beta = 0.2   # Peso del tiempo de espera
gamma = 0.1  # Peso de la prioridad
num_vehicles = 5  # Número de camiones
max_time_per_vehicle = 480

# Aplicar K-Means clustering para dividir las entregas entre los camiones
#clustered_solution = apply_kmeans(problem, num_vehicles,alpha, beta, gamma)
# Ejemplo de uso:
# - Primero aplicar K-Means para distribuir las entregas
# - Luego asignar esas entregas a cada vehículo con las restricciones de tiempo

# Aplicar K-Means clustering para dividir las entregas entre los camiones
clustered_solution = apply_kmeans(problem, num_vehicles, alpha, beta, gamma, max_time_per_vehicle)

# Asignar las entregas a los vehículos y obtener las rutas finales
final_routes = assign_deliveries_to_vehicles(clustered_solution, problem, num_vehicles, max_time_per_vehicle, alpha, beta, gamma)

# Evaluar la solución completa para K-Means
# Imprimir las rutas finales
for vehicle_id, route in enumerate(final_routes):
    print(f"Ruta del Vehículo {vehicle_id}: {route}")
print("==== Evaluando la solución generada por K-Means ====")

# Evaluar la solución final
fitness_value, total_cost, total_priority, not_served_count, total_time, total_distance = evaluate_final_solution(final_routes, problem, alpha, beta, gamma)

# Imprimir los resultados finales
print(f"\nResultados finales de la evaluación:")
print(f"Fitness total: {fitness_value}")
print(f"Coste total: {total_cost}")
print(f"Prioridad total: {total_priority}")
print(f"Órdenes no servidas: {not_served_count}")
print(f"Tiempo total: {total_time} min")
print(f"Distancia total: {total_distance} km")


plot_vehicle_routes(problem, final_routes)








# Evaluar todas las rutas agrupadas de K-Means juntas
# clustered_routes = []
# for route in clustered_solution:
#     clustered_routes.extend(route)







# # Evaluar la solución completa generada por K-Means
# priority, distance, routes, total_time, not_served_count, delivery_times = eval_solution(problem, clustered_routes)

# # Calcular el coste total de la solución K-Means
# cost = distance * problem.km_cost + len(routes) * problem.truck_cost

# # Calcular el fitness total de la solución K-Means
# fitness_value = fitness(problem, alpha, beta, cost, priority, not_served_count)

# # Imprimir los resultados de la solución K-Means
# print(f"Solución K-Means:")
# print(f" - Prioridad total: {priority}")
# print(f" - Distancia total: {distance} km")
# print(f" - Tiempo total: {total_time} min")
# print(f" - Número de órdenes no servidas: {not_served_count}")
# print(f" - Coste total: {cost}")
# print(f" - Fitness total: {fitness_value}")
# print(f" - Número de rutas de la solución inicial: {len(routes)}")
# print("=========================")










# #Parameters
# alpha = 0.7
# beta = 0.2
# gamma = 0.1
# VNS_type = "SWAP and INSERT - Kmeans"
# num_vehicles = 3

# # Crear la solución inicial basada en el coste más bajo
# initial_solution = create_initial_solution(problem, alpha=alpha, beta=beta, gamma=gamma)

# # Evaluar la solución inicial
# current_priority, distance, routes, total_time, not_served_count, delivery_times = eval_solution(problem, initial_solution)

# # Calcular el coste total de la solución
# current_cost = distance * problem.km_cost + len(routes) * problem.truck_cost

# # Calcular el fitness de la solución inicial
# current_solution_fitness = fitness(problem, alpha, beta, current_cost, current_priority, not_served_count)

# # Guardar la solución inicial como la mejor solución
# best_solution = copy.deepcopy(initial_solution)
# best_solution_value = [copy.deepcopy(current_cost), copy.deepcopy(current_priority)]
# best_solution_fitness = copy.deepcopy(current_solution_fitness)

# # Imprimir los resultados de la solución inicial
# print("==== Solución inicial basada en el coste más bajo ====")
# print(f"Prioridad total: {current_priority}")
# print(f"Distancia total: {distance} km")
# print(f"Rutas (número de camiones usados): {len(routes)}")
# print(f"Tiempo total: {total_time} min")
# print(f"Número de órdenes no servidas: {not_served_count}")
# print(f"Coste total: {current_cost}")
# print(f"Fitness de la solución inicial: {current_solution_fitness}")

# # Imprimir detalles de las rutas generadas
# for i, route in enumerate(routes):
#     print(f"Ruta {i+1}:")
#     for stop in route:
#         print(f" - {stop}")
#     print(f"Tiempo de entrega para esta ruta: {delivery_times[i]}")
# print("=========================")



  
