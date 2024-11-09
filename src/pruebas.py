# -*- coding: utf-8 -*-
"""
Created on Thu Oct 17 16:26:03 2024

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
import os
from datetime import datetime

# Limitar el número de threads para evitar advertencias de MKL
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

# Leer problema
problem = read_problem_file("./3R/problem_files/problem_500.txt")
problem_name = "problem"
problem.create_dictionaries()

# Parámetros que vamos a probar
alphas = [0.5, 0.6, 0.7, 0.8]  # Variaciones de peso para la distancia
betas = [0.1, 0.2, 0.3, 0.4]   # Variaciones de peso para el tiempo de espera
gammas = [0.1, 0.2, 0.3]       # Variaciones de peso para la prioridad
num_vehicles_list = [5, 10]     # Número de camiones diferentes
max_time_per_vehicle = 480      # Máximo tiempo por vehículo (8 horas)

# Configuración de archivo de resultados
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
result_file = f"Resultados_Pruebas_KMeans_{timestamp}.xlsx"
workbook = Workbook()
sheet = workbook.active
sheet.title = "Resultados"

# Añadir encabezados
sheet.append(["Prueba", "Alfa", "Beta", "Gamma", "Num_Vehículos", "Fitness", "Coste Total", "Prioridad Total", "Órdenes No Servidas", "Tiempo Total", "Distancia Total"])

# Ejecutar las pruebas
test_count = 1
for alpha in alphas:
    for beta in betas:
        for gamma in gammas:
            for num_vehicles in num_vehicles_list:
                print(f"\n=== Corriendo prueba {test_count} con parámetros: alfa={alpha}, beta={beta}, gamma={gamma}, num_vehicles={num_vehicles} ===")

                # Aplicar K-Means clustering para dividir las entregas entre los camiones
                clustered_solution = apply_kmeans(problem, num_vehicles, alpha, beta, gamma, max_time_per_vehicle)

                # Asignar las entregas a los vehículos y obtener las rutas finales
                final_routes = assign_deliveries_to_vehicles(clustered_solution, problem, num_vehicles, max_time_per_vehicle, alpha, beta, gamma)

                # Evaluar la solución final
                fitness_value, total_cost, total_priority, not_served_count, total_time, total_distance = evaluate_final_solution(final_routes, problem, alpha, beta, gamma)

                # Guardar resultados en Excel
                sheet.append([test_count, alpha, beta, gamma, num_vehicles, fitness_value, total_cost, total_priority, not_served_count, total_time, total_distance])

                # Imprimir resultados de la prueba actual
                print(f"\nResultados de la prueba {test_count}:")
                print(f"Fitness total: {fitness_value}")
                print(f"Coste total: {total_cost}")
                print(f"Prioridad total: {total_priority}")
                print(f"Órdenes no servidas: {not_served_count}")
                print(f"Tiempo total: {total_time} min")
                print(f"Distancia total: {total_distance} km")

                # Incrementar el contador de pruebas
                test_count += 1

# Guardar el archivo Excel
workbook.save(result_file)
print(f"Resultados guardados en {result_file}")
