# -*- coding: utf-8 -*-
"""
@author: Jose Ballestero
"""

from generator import *
from models import *
from utils import *
from openpyxl import Workbook
import openpyxl
from sklearn.cluster import KMeans
import numpy as np
import os
from datetime import datetime
import time

# Limitar el número de threads para evitar advertencias de MKL
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

# Leer problema
problem_nm = "problem_1000"
problem = read_problem_file("./3R/problem_files/" + problem_nm + ".txt")
problem_name = "problem"
problem.create_dictionaries()

# Parámetros que vamos a probar
alphas = [1, 10, 1, 1]  # Variaciones de peso para la distancia
betas = [1/10, 1/10, 1, 1/10]   # Variaciones de peso para el tiempo de espera
gammas = [1, 1, 1, 5]       # Variaciones de peso para la prioridad
#num_vehicles_list = [1]
#num_vehicles_list = [2, 3, 4]     # Número de camiones diferentes
#num_vehicles_list = [5, 6, 7]     # Número de camiones diferentes
num_vehicles_list = [11, 12, 13]     # Número de camiones diferentes
max_time_per_vehicle = 480      # Máximo tiempo por vehículo (8 horas)

# Configuración de archivo de resultados
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
result_file = f"./Resultados/Resultados_Pruebas_KMeans_{problem_nm}_{timestamp}.xlsx"
workbook = Workbook()
sheet = workbook.active
sheet.title = "Resultados"

# Añadir encabezados
sheet.append(["Prueba", "Alfa", "Beta", "Gamma", "Num_Vehículos", "Distancia Total", "Tiempo Total", "Prioridad Normalizada", "Coste Total", "Tiempo total de espera", "Órdenes No Servidas"])

# Ejecutar las pruebas
test_count = 1
for alpha, beta, gamma in zip(alphas, betas, gammas):
    for num_vehicles in num_vehicles_list:
        print(f"\n=== Corriendo prueba {test_count} con parámetros: alfa={alpha}, beta={beta}, gamma={gamma}, num_vehicles={num_vehicles} ===")
    
        # Aplicar K-Means clustering para dividir las entregas entre los camiones
        clustered_solution = apply_kmeans(problem, num_vehicles, alpha, beta, gamma, max_time_per_vehicle)
    
        # Asignar las entregas a los vehículos y obtener las rutas finales
        final_routes, total_waiting_time, current_storage = assign_deliveries_to_vehicles(clustered_solution, problem, num_vehicles, max_time_per_vehicle, alpha, beta, gamma)
    
        # Evaluar la solución final
        normalized_priority, total_cost, total_priority, not_served_count, total_time, total_distance = evaluate_final_solution(final_routes, problem, alpha, beta, gamma)
    
        # Guardar resultados en Excel
        sheet.append([test_count, alpha, beta, gamma, num_vehicles, total_distance, total_time, normalized_priority, total_cost, total_waiting_time, not_served_count])
        
        # Compute delivery times using problem's real data
        exact_delivery_times = calculate_delivery_times(final_routes, problem)
        
        # Plot the Gantt chart
        plot_gantt_chart(exact_delivery_times)

        # Imprimir resultados de la prueba actual
        print(f"\nResultados de la prueba {test_count}:")
        print(f"Coste total: {total_cost}")
        print(f"Prioridad total: {total_priority}")
        print(f"Prioridad normalizada: {normalized_priority}")
        print(f"Órdenes no servidas: {not_served_count}")
        print(f"Tiempo total: {total_time} min")
        print(f"Distancia total: {total_distance} km")
        print(f"Tiempo de espera total: {total_waiting_time} min")
    
        # Incrementar el contador de pruebas
        test_count += 1

# Guardar el archivo Excel
workbook.save(result_file)
print(f"Resultados guardados en {result_file}")
