from algorithms.vnd import *
from generator import *
from models import *
from utils import *
from openpyxl import Workbook
import openpyxl
from sklearn.cluster import KMeans
import numpy as np
import time


# Limitar el número de threads para evitar advertencias de MKL
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

# Almacena el tiempo de inicio
start_time = time.time()

# Read problem from file

#problem = read_problem_file("./problem_files/3H_100/txt/test_3.txt")
problem = read_problem_file("./3R/problem_files/problem_25.txt")
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
gamma = 0.5  # Peso de la prioridad
num_vehicles = 3  # Número de camiones
max_time_per_vehicle = 480

# Aplicar K-Means clustering para dividir las entregas entre los camiones
#clustered_solution = apply_kmeans(problem, num_vehicles,alpha, beta, gamma)
# Ejemplo de uso:
# - Primero aplicar K-Means para distribuir las entregas
# - Luego asignar esas entregas a cada vehículo con las restricciones de tiempo

# Aplicar K-Means clustering para dividir las entregas entre los camiones
clustered_solution = apply_kmeans(problem, num_vehicles, alpha, beta, gamma, max_time_per_vehicle, problem_name)

# Asignar las entregas a los vehículos y obtener las rutas finales
final_routes = assign_deliveries_to_vehicles(clustered_solution, problem, num_vehicles, max_time_per_vehicle, alpha, beta, gamma)

# Evaluar la solución completa para K-Means
# Imprimir las rutas finales
for vehicle_id, route in enumerate(final_routes):
    print(f"Ruta del Vehículo {vehicle_id}: {route}")
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


plot_vehicle_routes(problem, final_routes)

# Ejecutar VNS en la solución inicial
# Definir los parámetros para VNS
time_limit = 1800  # Tiempo límite en segundos
log = True  # Mostrar el progreso de VNS

alpha = 0.5  # Peso de la distancia
beta = 0.5   # Peso del tiempo de espera
gamma = 0.5

# Llamada a la función VNS
best_solution, best_solution_value, swap_count, len_neigh = vns(problem, final_routes, alpha, beta, gamma, time_limit, 0, True, False)

# Evaluar la solución mejorada obtenida por VNS
normalized_priority, final_cost, final_priority, not_served_count, total_time, total_distance = evaluate_final_solution(best_solution, problem, alpha, beta, gamma)
total_cost = total_distance * problem.km_cost + len(final_routes) * problem.truck_cost
# Almacena el tiempo de finalización
end_time = time.time()
# Calcula el tiempo total transcurrido en segundos
elapsed_time = end_time - start_time

cost_variation = (total_cost - final_cost) / total_cost * 100
priority_variation = (total_priority - final_priority) * 100

# Mostrar resultados finales
print("\n==== Resultados finales de la solución mejorada por VNS ====")
print(f"Coste total: {final_cost}")
print(f"Prioridad total: {final_priority}")
print(f"Órdenes no servidas: {not_served_count}")
print(f"Tiempo total: {total_time} min")
print(f"Distancia total: {total_distance} km")
print(f"Prioridad normalizada: {normalized_priority}")
print(f"Swap count: {swap_count}")
print(f"Count neigh: {len_neigh}")
print(f"Porcentaje de variación del coste: {cost_variation:.2f} %")
print(f"Porcentaje de variación de la prioridad: {priority_variation} %")
print(f"Tiempo de ejecución: {elapsed_time:.2f} segundos")
# (Opcional) Gráfico de rutas finales
plot_vehicle_routes(problem, best_solution)



