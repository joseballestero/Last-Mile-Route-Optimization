from algorithms.vnd import *
from generator import *
from models import *
from utils import *
from openpyxl import Workbook
import openpyxl
from sklearn.cluster import KMeans
import numpy as np
import pandas as pd
from datetime import datetime

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

# Definir las configuraciones de los experimentos
configs = [
    {'alpha': 0.7, 'beta': 0.2, 'gamma': 0.1, 'num_vehicles': 5},
    {'alpha': 0.6, 'beta': 0.3, 'gamma': 0.1, 'num_vehicles': 5},
    {'alpha': 0.5, 'beta': 0.3, 'gamma': 0.2, 'num_vehicles': 7},
    {'alpha': 0.4, 'beta': 0.4, 'gamma': 0.2, 'num_vehicles': 6},
    # Añadir más configuraciones si es necesario...
]

# Cargar el problema generado previamente
#problem = load_previously_generated_problem('problem_500')  # Carga el problema ya existente

# Lista para almacenar los resultados
results = []

# Ejecutar experimentos
for i, config in enumerate(configs):
    print(f"Corriendo experimento {i + 1}")
    
    plot_name = f"cluster_plot_{i+1}"  # Nombre del archivo para el plot
    
    initial_fitness, initial_cost, fitness_value, cost, not_served, total_time, distance = run_experiment(problem, config, plot_name)

    # Registrar los resultados en un diccionario
    result = {
        'Ejecución': i + 1,
        'Alpha': config['alpha'],
        'Beta': config['beta'],
        'Gamma': config['gamma'],
        'Num_vehicles': config['num_vehicles'],
        'Coste inicial': initial_cost,
        'Fitness inicial': initial_fitness,
        'Coste K-Means': cost,
        'Fitness K-Means': fitness_value,
        'Órdenes no servidas K-Means': not_served,
        'Tiempo total K-Means': total_time,
        'Distancia total K-Means': distance,
        'Plot': plot_name + '.png'  # Guardar el nombre del archivo del plot
    }
    results.append(result)

# Convertir resultados a DataFrame de pandas
df_results = pd.DataFrame(results)

# Guardar resultados en un archivo Excel
file_name = f"resultados_experimentos_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
df_results.to_excel(file_name, index=False)

print(f"Resultados guardados en {file_name}")


  
