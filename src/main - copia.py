from algorithms.vnd import *
from generator import *
from models import *
from utils import *
from openpyxl import Workbook
import openpyxl


# Limitar el número de threads para evitar advertencias de MKL
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

# Read problem from file

problem = read_problem_file("./problem_files/3H_100/txt/3H_20_1.txt")
problem_name = "3H_20_1"
problem.create_dictionaries()
list_opt = problem.create_list_of_options()

#Read excel file
name = "Results.xlsx"
excel_document = openpyxl.load_workbook(name, data_only=True)

#VNS


    #Parameters for VNS
    # iterations = 200
    # alpha = 0.8
    # beta = 0.2
    # VNS_type = "insercción"
    
    #Parameters for loop VNS
alpha = 0.8
beta = 0.2
VNS_type = "SWAP and INSERT - Kmeans"
    
best_solution, best_solution_value, best_solution_fitness, stats, exec_time, inicial_routes, inicial_solution_fitness, iterations, countSwap, countInsert = vnd(problem, alpha, beta, log=True)
solution_obj = create_solution(problem, best_solution)
    
    #Validar metodo de comprobar rutas, is_feasible (reduced_tws)
    
    # if (solution_obj.is_feasible(problem)):
    #     print(solution_obj, exec_time)
    #     
save_solution(best_solution, "./solution_files/" + problem_name + "_TS_solution_" + str(round(best_solution_fitness, 2)))
get_solution_charts(problem, solution_obj, "./solution_files/" + problem_name + "_TS_solution_" + str(round(best_solution_fitness, 2)), stats)
    # else:
    #     print("Hay más rutas que trucks")
    
  
print(solution_obj, exec_time) 
       
    #Write Solution metrics on excel   
export_to_excel(excel_document, name, iterations, alpha, beta, VNS_type, inicial_routes, len(solution_obj.routes), inicial_solution_fitness, best_solution_fitness, exec_time, len(problem.customers),countSwap, countInsert)




# Solution from file

# solution, stats = read_solution_file("./tests/solution_files/test_2_SA_80-20.npz", problem)
# priority, distance, miss_prob, routes, delivery_times, total_time, not_served_count = eval_solution_prob(problem, solution)
# cost = distance * problem.km_cost + len(routes) * problem.truck_cost
# solution_value = fitness(problem, 0.8, 0.2, 0., cost, priority, miss_prob, not_served_count)
# print("test_2_SA: priority = {}, distance = {}, miss_prob = {}, vehicles = {}, not served = {}, cost = {}, fitness = {}".format(
#     priority, round(distance, 2), round(miss_prob, 2), len(routes), not_served_count, round(cost, 2), round(solution_value, 2)))

# Simulation Test

# hit_rates = []
# for i in range(1000):
#   dict_delivery_success = generate_delivery_simulation(problem)
#   hit_rate = simulate_routes(problem, routes, delivery_times, dict_delivery_success)
#   hit_rates.append(hit_rate)
# print("test_2_SA mean hit rate: ", sum(hit_rates) / len(hit_rates))