"""This module contains utility functions for the VRPTWDO project."""

import copy
import random
import time

import matplotlib.pyplot as plt
import numpy as np

from openpyxl import Workbook
from openpyxl.utils import get_column_letter

from matplotlib.backends.backend_pdf import PdfPages
from models import *

# PROBLEM FILE EXAMPLE

# NUM_VEHICLES	AV_CAPAC.   TRUCK_COST  KM_COST   X_DEPOT	Y_DEPOT TW_E    TW_L
# 10  10000 70  100 40	50  540  1320

# LOCKER_ID	X	Y	AV_CAPAC.
# L1  41  6 25
# L2	30	8	25
# L3	99	98	25
# L4	46	68	25
# L5	19	66	25
# L6	48	35	25
# L7	26	35	25

# SHOP_ID X	Y	TW_E1	TW_L1	TW_E2	TW_L2	AV_CAPAC.
# S1	40	81	540	1260	-	-	50
# S2	49	41	540	1260	-	-	50
# S3	94	47	600	840	1020	1290	50
# S4	9	98	600	840	1020	1290	50
# S5	75	98	540	1260	-	-	50
# S6	36	49	540	1260	-	-	50
# S7	40	21	600	840	1020	1290	50

# ORDER_ID	WEIGHT	VOLUME	PRIORITY	DP_ID	X	Y	TW_PROBABILITIES
# 1 0.5 0.7 1 L1
# 1 0.5 0.7 2 S1
# 1 0.5 0.7 3 L2
# 1 0.5 0.7 4 H1  25  78  600,840,0.6; 1200,1450,0.4
# 2 0.8 0.9 1 H2  34  57  500,670,0.85; 950,1130,0.3
# 2 0.8 0.9 2 L3
# 2 0.8 0.9 3 S2
# 2 0.8 0.9 4 L4
# 3 0.6 0.8 1 S5
# 3 0.6 0.8 2 H3  60  18  120,200,0.95; 250,560,0.7; 800,1000,0.2
# 4 0.7 0.9 1 L2
# 4 0.7 0.9 2 S6


def read_problem_file(file_path: str):
    """
    Reads a problem file and returns a Problem instance.

    Parameters:
        file_path : str
            File path.

    Returns:
        problem : Problem
            Problem instance.
    """

    with open(file_path, 'r') as f:
        lines = f.readlines()

    num_vehicles = int(lines[1].split()[0])
    truck_cost = int(lines[1].split()[1])
    truck_speed = int(lines[1].split()[2])
    km_cost = int(lines[1].split()[3])
    if '.' in lines[1].split()[4] and '.' in lines[1].split()[5]:
        depot_location = Location(float(lines[1].split()[4]), float(lines[1].split()[5]))
    else:
        depot_location = Location(int(lines[1].split()[4]), int(lines[1].split()[5]))
    depot_tw = tuple(map(int, lines[1].split()[6:8]))

    fleet = []
    for i in range(num_vehicles):
        new_truck = Truck(truck_cost)
        fleet.append(new_truck)
    depot = Depot(depot_location, fleet, depot_tw)

    customers = []
    delivery_points = []
    orders = []
    current_order_id = 'O1'
    current_order = None

    for line in lines[3:]:
        strings = line.split()
        strings =  [string.strip() for string in strings]

        if line[0] == 'L' and not 'LOCKER_ID' in line:
            if '.' in strings[1] and '.' in strings[2]:
                loc = Location(float(strings[1]), float(strings[2]))
            else:
                loc = Location(int(strings[1]), int(strings[2]))
            time_to_depot = loc.distance(depot_location) / truck_speed * 60
            new_locker = DeliveryPoint(DeliveryType.LOCKER, loc, time_to_depot, [], int(strings[3]), strings[0])
            delivery_points.append(new_locker)

        elif line[0] == 'S' and not 'SHOP_ID' in line:
            tws = []
            tw = list(map(int, strings[3:5]))
            tw.append(1.0)
            tws.append(tuple(tw))
            if strings[5] != '-':
                tw = list(map(int, strings[5:7]))
                tw.append(1.0)
                tws.append(tuple(tw))
            if '.' in strings[1] and '.' in strings[2]:
                loc = Location(float(strings[1]), float(strings[2]))
            else:
                loc = Location(int(strings[1]), int(strings[2]))
            time_to_depot = loc.distance(depot_location) / truck_speed * 60
            new_shop = DeliveryPoint(DeliveryType.SHOP, loc, time_to_depot, tws, int(strings[7]), strings[0])
            delivery_points.append(new_shop)

        elif line[0] == 'O' and not 'ORDER_ID' in line:
            new_order_id = strings[0]
            new_delivery_option = None

            if strings[4][0] == 'L' or strings[4][0] == 'S':
                dp = next((dp for dp in delivery_points if dp.id == strings[4]), None)
                new_delivery_option = (dp, int(strings[3]))
            elif strings[4][0] == 'H':
                tws = []
                str_tws_probs = strings[7:]
                for str in str_tws_probs:
                    str = str.replace(';', '')
                    nums = str.split(',')
                    tws.append((int(nums[0]), int(nums[1]), float(nums[2])))
                if '.' in strings[5] and '.' in strings[6]:
                    loc = Location(float(strings[5]), float(strings[6]))
                else:
                    loc = Location(int(strings[5]), int(strings[6]))
                time_to_depot = loc.distance(depot_location) / truck_speed * 60
                new_home = PersonalPoint(loc, time_to_depot, False, tws, [], id=strings[4])
                delivery_points.append(new_home)
                new_delivery_option = (new_home, int(strings[3]))

            if new_order_id == current_order_id:
                if current_order is None:
                    new_customer = Customer('C' + current_order_id[1:])
                    customers.append(new_customer)
                    current_order = Order(new_customer, int(strings[2]), int(strings[1]), -1, [], 'O' + current_order_id[1:])
                current_order.add_delivery_option(new_delivery_option)
            else:
                orders.append(current_order)
                current_order_id = new_order_id
                new_customer = Customer('C' + current_order_id[1:])
                customers.append(new_customer)
                current_order = Order(new_customer, int(strings[2]), int(strings[1]), -1, [], 'O' + current_order_id[1:])
                current_order.add_delivery_option(new_delivery_option)

    orders.append(current_order)

    problem = Problem(depot, truck_speed, truck_cost, km_cost, customers, delivery_points, orders)

    return problem


def plot_problem(problem: Problem, file_name: str = 'problem', loc_names: bool = True):
    """
    Plots a problem.

    Parameters:
        problem : Problem
            Problem instance.
        file_name : str
            File name.
    """

    lockers = [dp for dp in problem.delivery_points if dp.delivery_type == DeliveryType.LOCKER]
    shops = [dp for dp in problem.delivery_points if dp.delivery_type == DeliveryType.SHOP]
    home_delivery_points = [dp for dp in problem.delivery_points if dp.delivery_type == DeliveryType.HOME]

    plt.clf()
    plt.rcParams['figure.figsize'] = plt.rcParamsDefault['figure.figsize']

    marker_size = 50
    if len(problem.delivery_points) > 500:
        marker_size = 10
        plt.figure(figsize=(8, 6))

    plt.scatter([dp.loc.x for dp in lockers], [dp.loc.y for dp in lockers], color='#7C92F3', marker='p', s=marker_size, label='Lockers')
    plt.scatter([dp.loc.x for dp in shops], [dp.loc.y for dp in shops], color='#86E9AC', marker='H', s=marker_size, label='Shops')
    plt.scatter([dp.loc.x for dp in home_delivery_points], [dp.loc.y for dp in home_delivery_points], color='#F6A2A8', marker='^', s=marker_size, label='Home Delivery Points')
    plt.scatter(problem.depot.loc.x, problem.depot.loc.y, color='black', marker='D', s=2*marker_size, label='Depot')

    if loc_names:
        name_sep = 0.5
        if plt.xlim()[1] - plt.xlim()[0] < 30: name_sep = 0.15

        for dp in problem.delivery_points:
            plt.text(dp.loc.x + name_sep, dp.loc.y, str(dp.id), fontsize=6, ha='left', va='bottom')

    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.title('Problem Locations')
    plt.grid(True)
    plt.savefig(file_name + '_plot.pdf', format='pdf')


def get_random_solution(problem: Problem):
    """
    Returns a random solution for the tabu search algorithm.

    Parameters:
        problem : Problem
            Problem instance.

    Returns:
        solution : list
            Vector solution. List of (Order_ID, DeliveryPoint_ID) tuples.
    """
    random_solution = problem.create_list_of_options()
    random.shuffle(random_solution)

    return random_solution


def fitness(problem: Problem, cost_weight: float, priority_weight: float, cost: float, priority: int, not_served_count: int = 0):
    """
    Returns the fitness of a solution for optimization algorithms.

    Parameters:
        problem : Problem
            Problem instance.
        cost_weight : float
            Distance weight.
        priority_weight : float
            Priority weight.
        miss_prob_weight: float
            Miss probability weight.
        cost : float
            Total cost of the solution.
        priority : float
            Total priority of the solution.
        miss_prob: float
            Mean probability of missed deliveries.
        not_served_count : int
            Number of orders not served.
    Returns:
        fitness : float
            Fitness of the solution.
    """

    max_cost = get_max_cost(problem)
    max_priority = get_max_priority(problem)
    min_priority = len(problem.orders)

    normalized_cost = cost / max_cost
    normalized_priority = (priority - min_priority) / (max_priority - min_priority)

    return (cost_weight * normalized_cost + priority_weight * normalized_priority + 5 * not_served_count) * 100


def save_solution(solution: list, file_name: str, stats: list = []):
    """
    Saves a solution to a file.

    Parameters:
        solution : list
            Vector solution. List of (Order_ID, DeliveryPoint_ID) tuples.
        file_name : str
            File name.
    """
    min_values = [s["min"] for s in stats]
    np.savez(file_name, solution=np.array(solution), stats=np.array(min_values))


def read_solution_file(file_path: str, problem: Problem):
    """
    Reads a solution file and returns a solution as a vector.

    Parameters:
        file_path : str
            File path.
        problem : Problem
            Problem instance.

    Returns:
        solution : list
            Vector solution. List of (Order_ID, DeliveryPoint_ID) tuples.
    """

    solution, stats = [], []
    data = np.load(file_path)
    if file_path.endswith(".npz"):
        if len(data["stats"]):
            solution = [tuple(row) for row in data["solution"]]
            stats = list(data["stats"])
        else:
            solution = [tuple(row) for row in data["solution"]]
    elif file_path.endswith(".npy"):
        solution = [tuple(row) for row in data]
    return solution, stats



# def eval_solution(problem: Problem, solution: list):
#     """
#     Returns a solution evaluation.

#     Parameters:
#         problem : Problem
#             Problem instance.
#         solution : list
#             Vector solution. List of delivery options.

#     Returns:
#         priority : float
#             Total priority of the solution.
#         distance : float
#             Total distance of the solution.
#         routes : list
#             List of routes. Each route is a list of delivery points.
#         delivery_times : list
#             List of delivery times for each route.
#         total_time : float
#             Total time of the solution.
#         not_served_count : int
#             Number of orders not served.
#     """
#     start = time.time()

#     routes = []
#     served = {order.id: 0 for order in problem.orders}
#     visited = set()  # Keep track of visited destinations
#     capacity = copy.deepcopy(problem.dict_capacity)
#     delivery_times = []

#     route = [(None, 'DEPOT')]
#     route_arrival = [problem.dict_twe['DEPOT']]
#     route_initial_time = [problem.dict_twe['DEPOT']]
#     route_departure = [problem.dict_twe['DEPOT']]
#     route_delivery_times = []
#     distance = 0
#     total_time = 0
#     priority = 0

#     ind = 0
#     while ind < len(solution):
#         option = solution[ind]
#         order, dp = option
#         last_dp = route[-1][1]

#         if not served[order] and capacity[dp] != 0 and dp not in visited:
#             arrival_time = route_departure[-1] + problem.dict_distance[(last_dp, dp)] * 60 / problem.truck_speed
#             if isinstance(problem.dict_twe[dp], list) and isinstance(problem.dict_twl[dp], list):
#                 for te, tl in zip(problem.dict_twe[dp], problem.dict_twl[dp]):
#                     initial_time = max(arrival_time, float(te))
#                     twe, twl = te, tl
#                     if initial_time <= tl:
#                         break
#             else:
#                 twe, twl = problem.dict_twe[dp], problem.dict_twl[dp]
#                 initial_time = max(arrival_time,  twe)

#             delivery_time = 0
#             if dp != last_dp:
#                 delivery_time += problem.dict_delivery_time['DEFAULT']
#             if 'H' == dp[0]:
#                 delivery_time += problem.dict_delivery_time['HOME']
#             elif 'L' == dp[0]:
#                 delivery_time += problem.dict_delivery_time['LOCKER']
#             elif 'S' == dp[0]:
#                 delivery_time += problem.dict_delivery_time['SHOP']

#             departure_time = initial_time + delivery_time
#             arrival_time_depot = departure_time + problem.dict_distance[(dp, 'DEPOT')] * 60 / problem.truck_speed
#             if arrival_time <= twl and arrival_time_depot <= problem.dict_twl['DEPOT']:
#                 distance += problem.dict_distance[(last_dp, dp)]
#                 total_time += problem.dict_distance[(last_dp, dp)] * 60 / problem.truck_speed
#                 priority += problem.dict_priority[option]
#                 route.append(option)
#                 route_arrival.append(arrival_time)
#                 route_initial_time.append(initial_time)
#                 route_departure.append(departure_time)
#                 route_delivery_times.append(round(departure_time, 2))
#                 served[order] = 1
#                 visited.add(dp)  # Mark destination as visited
#                 capacity[dp] -= 1
#                 ind += 1
#             else:
#                 if len(route) > 1 and last_dp != 'DEPOT':
#                     distance += problem.dict_distance[(last_dp, 'DEPOT')]
#                     total_time += problem.dict_distance[(last_dp, 'DEPOT')] * 60 / problem.truck_speed
#                     route.append((None, 'DEPOT'))
#                     routes.append(route)
#                     delivery_times.append(route_delivery_times)
#                     route = [(None, 'DEPOT')]
#                     route_arrival = [problem.dict_twe['DEPOT']]
#                     route_initial_time = [problem.dict_twe['DEPOT']]
#                     route_departure = [problem.dict_twe['DEPOT']]
#                     route_delivery_times = []
#                 else:
#                     # DP is not reachable
#                     ind += 1
#         else:
#             ind += 1

#         if ind + 1 == len(solution):
#             distance += problem.dict_distance[(last_dp, 'DEPOT')]
#             total_time += problem.dict_distance[(last_dp, 'DEPOT')] * 60 / problem.truck_speed
#             route.append((None, 'DEPOT'))
#             routes.append(route)
#             delivery_times.append(route_delivery_times)

#     not_served_count = sum(1 for x in served.values() if x == 0)
#     # if not_served_count > 0:
#     #     raise Exception("Invalid solution: " + str(not_served_count) + " orders could not be served.")

#     # print("Time (seconds): ", time.time() - start)

#     return priority, distance, routes, total_time, not_served_count



def eval_solution(problem: Problem, solution: list):
    """
    Returns a solution evaluation.

    Parameters:
        problem : Problem
            Problem instance.
        solution : list
            Vector solution. List of delivery options.

    Returns:
        priority : float
            Total priority of the solution.
        distance : float
            Total distance of the solution.
        routes : list
            List of routes. Each route is a list of delivery points.
        delivery_times : list
            List of delivery times for each route.
        total_time : float
            Total time of the solution.
        not_served_count : int
            Number of orders not served.
    """
    start = time.time()

    routes = []
    served = {order.id: 0 for order in problem.orders}
    capacity = copy.deepcopy(problem.dict_capacity)
    delivery_times = []

    route = [(None,'DEPOT')]
    route_arrival = [problem.dict_twe['DEPOT']]
    route_initial_time = [problem.dict_twe['DEPOT']]
    route_departure = [problem.dict_twe['DEPOT']]
    route_delivery_times = []
    distance = 0
    total_time = 0
    priority = 0

    ind = 0
    while ind < len(solution):
        option = solution[ind]
        order, dp = option
        last_dp = route[-1][1]

        if not served[order] and capacity[dp] != 0:
            arrival_time = route_departure[-1] + problem.dict_distance[(last_dp, dp)] * 60 / problem.truck_speed
            if isinstance(problem.dict_twe[dp], list) and isinstance(problem.dict_twl[dp], list):
                for te, tl in zip(problem.dict_twe[dp], problem.dict_twl[dp]):
                    initial_time = max(arrival_time, float(te))
                    twe, twl = te, tl
                    if initial_time <= tl:
                        break
            else:
                twe, twl = problem.dict_twe[dp], problem.dict_twl[dp]
                initial_time = max(arrival_time,  twe)

            delivery_time = 0
            if dp != last_dp:
                delivery_time += problem.dict_delivery_time['DEFAULT']
            if 'H' == dp[0]:
                delivery_time += problem.dict_delivery_time['HOME']
            elif 'L' == dp[0]:
                delivery_time += problem.dict_delivery_time['LOCKER']
            elif 'S' == dp[0]:
                delivery_time += problem.dict_delivery_time['SHOP']

            departure_time = initial_time + delivery_time
            arrival_time_depot = departure_time + problem.dict_distance[(dp, 'DEPOT')] * 60 / problem.truck_speed
            if arrival_time <= twl and arrival_time_depot <= problem.dict_twl['DEPOT']:
                distance += problem.dict_distance[(last_dp, dp)]
                total_time += problem.dict_distance[(last_dp, dp)] * 60 / problem.truck_speed
                priority += problem.dict_priority[option]
                route.append(option)
                route_arrival.append(arrival_time)
                route_initial_time.append(initial_time)
                route_departure.append(departure_time)
                route_delivery_times.append(round(departure_time, 2))
                served[order] = 1
                capacity[dp] -= 1
                ind += 1
            else:
                if len(route) > 1 and last_dp != 'DEPOT':
                    distance += problem.dict_distance[(last_dp, 'DEPOT')]
                    total_time += problem.dict_distance[(last_dp, 'DEPOT')] * 60 / problem.truck_speed
                    route.append((None,'DEPOT'))
                    routes.append(route)
                    delivery_times.append(route_delivery_times)
                    route = [(None,'DEPOT')]
                    route_arrival = [problem.dict_twe['DEPOT']]
                    route_initial_time = [problem.dict_twe['DEPOT']]
                    route_departure = [problem.dict_twe['DEPOT']]
                else:
                    # DP is not reachable
                    ind += 1
        else:
            ind += 1

        if ind + 1 == len(solution):
            distance += problem.dict_distance[(last_dp, 'DEPOT')]
            total_time += problem.dict_distance[(last_dp, 'DEPOT')] * 60 / problem.truck_speed
            route.append((None,'DEPOT'))
            routes.append(route)
            delivery_times.append(route_delivery_times)

    not_served_count = sum(1 for x in served.values() if x == 0)
    # if not_served_count > 0:
    #     raise Exception("Invalid solution: " + str(not_served_count) + " orders could not be served.")

    # print("Time (seconds): ", time.time() - start)

    return priority, distance, routes, total_time, not_served_count


def create_solution(problem: Problem, solution: list):
    """
    Returns a solution object.

    Parameters:
        problem : Problem
            Problem instance.
        solution : list
            Vector solution. List of delivery options.

    Returns:
        solution : Solution
            Solution instance.
    """
    start = time.time()

    routes = []
    served = {order.id: 0 for order in problem.orders}
    capacity = copy.deepcopy(problem.dict_capacity)

    truck = Truck(problem.truck_cost)
    route = Route(truck)
    route_as_list = [(None,'DEPOT')]
    route.start_time = [problem.dict_twe['DEPOT']]
    route_arrival = [problem.dict_twe['DEPOT']]
    route_initial_time = [problem.dict_twe['DEPOT']]
    route_departure = [problem.dict_twe['DEPOT']]
    distance = 0
    total_time = 0
    priority = 0

    ind = 0
    while ind < len(solution):
        option = solution[ind]
        order, dp = option
        last_dp = route_as_list[-1][1]

        if not served[order] and capacity[dp] != 0:
            arrival_time = route_departure[-1] + problem.dict_distance[(last_dp, dp)] * 60 / problem.truck_speed
            dp_opts = []
            if isinstance(problem.dict_twe[dp], list) and isinstance(problem.dict_twl[dp], list):
                for te, tl in zip(problem.dict_twe[dp], problem.dict_twl[dp]):
                    initial_time = max(arrival_time, te)
                    twe, twl = te, tl
                    if initial_time <= tl:
                        dp_opts.append((initial_time, twe, twl))
            else:
                twe, twl = problem.dict_twe[dp], problem.dict_twl[dp]
                initial_time = max(arrival_time,  twe)
                dp_opts.append((initial_time, twe, twl))

            delivery_time = 0
            if dp != last_dp:
                delivery_time += problem.dict_delivery_time['DEFAULT']
            if 'H' == dp[0]:
                delivery_time += problem.dict_delivery_time['HOME']
            elif 'L' == dp[0]:
                delivery_time += problem.dict_delivery_time['LOCKER']
            elif 'S' == dp[0]:
                delivery_time += problem.dict_delivery_time['SHOP']

            if len(dp_opts) > 0:
                selected = False
                for initial_time, twe, twl in dp_opts:
                    departure_time = initial_time + delivery_time
                    arrival_time_depot = departure_time + problem.dict_distance[(dp, 'DEPOT')] * 60 / problem.truck_speed
                    if arrival_time <= twl and arrival_time_depot <= problem.dict_twl['DEPOT']:
                        distance += problem.dict_distance[(last_dp, dp)]
                        total_time += problem.dict_distance[(last_dp, dp)] * 60 / problem.truck_speed
                        priority += problem.dict_priority[option]

                        dp = problem.find_dp_by_id(dp)
                        order_obj = problem.find_order_by_id(order)
                        stop = Stop(dp, [order_obj], initial_time)
                        route.stops.append(stop)

                        route_as_list.append(option)
                        route_arrival.append(arrival_time)
                        route_initial_time.append(initial_time)
                        route_departure.append(departure_time)
                        served[order] = 1
                        capacity[dp.id] -= 1
                        ind += 1
                        selected = True
                        break
                if not selected:
                    if len(route_as_list) > 1 and last_dp != 'DEPOT':
                        distance += problem.dict_distance[(last_dp, 'DEPOT')]
                        total_time += problem.dict_distance[(last_dp, 'DEPOT')] * 60 / problem.truck_speed
                        route_copy = copy.deepcopy(route)
                        routes.append(route_copy)
                        new_truck = Truck(problem.truck_cost)
                        route.clear()
                        route.truck = new_truck
                        route_as_list = copy.deepcopy([(None,'DEPOT')])
                        route.start_time = [problem.dict_twe['DEPOT']]
                        route_arrival = [problem.dict_twe['DEPOT']]
                        route_initial_time = [problem.dict_twe['DEPOT']]
                        route_departure = [problem.dict_twe['DEPOT']]
                    else:
                        # DP is not reachable
                        ind += 1
            else:
                ind += 1
        else:
            ind += 1

        if ind + 1 == len(solution):
            distance += problem.dict_distance[(last_dp, 'DEPOT')]
            total_time += problem.dict_distance[(last_dp, 'DEPOT')] * 60 / problem.truck_speed
            route_copy = copy.deepcopy(route)
            routes.append(route_copy)

    not_served_count = sum(1 for x in served.values() if x == 0)
    # print("Not served: ", not_served_count)

    # print("Time (seconds): ", time.time() - start)

    solution = Solution(routes)

    return solution


def eval_solution_prob(problem: Problem, solution: list):
    """
    Returns a solution evaluation.

    Parameters:
        problem : Problem
            Problem instance.
        solution : list
            Vector solution. List of delivery options.

    Returns:
        priority : float
            Total priority of the solution.
        distance : float
            Total distance of the solution.
        miss_prob : float
            Average miss probability of the solution.
        routes : list
            List of routes. Each route is a list of delivery points.
        delivery_times : list
            List of delivery times for each route.
        total_time : float
            Total time of the solution.
        not_served_count : int
            Number of orders not served.
    """
    start = time.time()

    routes = []
    served = {order.id: 0 for order in problem.orders}
    capacity = copy.deepcopy(problem.dict_capacity)
    delivery_times = []

    route = [(None,'DEPOT')]
    route_arrival = [problem.dict_twe['DEPOT']]
    route_initial_time = [problem.dict_twe['DEPOT']]
    route_departure = [problem.dict_twe['DEPOT']]
    route_delivery_times = []
    distance = 0
    total_time = 0
    priority = 0
    miss_probs = []

    ind = 0
    while ind < len(solution):
        option = solution[ind]
        order, dp = option
        last_dp = route[-1][1]

        if not served[order] and capacity[dp] != 0:
            arrival_time = route_departure[-1] + problem.dict_distance[(last_dp, dp)] * 60 / problem.truck_speed
            dp_opts = []
            if isinstance(problem.dict_twe[dp], list) and isinstance(problem.dict_twl[dp], list):
                for te, tl, p in zip(problem.dict_twe[dp], problem.dict_twl[dp], problem.dict_twp[dp]):
                    initial_time = max(arrival_time, float(te))
                    twe, twl, twp = te, tl, p
                    if initial_time <= tl:
                        dp_opts.append((initial_time, twe, twl, twp))
            else:
                twe, twl, twp = problem.dict_twe[dp], problem.dict_twl[dp], problem.dict_twp[dp]
                initial_time = max(arrival_time,  twe)
                dp_opts.append((initial_time, twe, twl, twp))

            delivery_time = 0
            if dp != last_dp:
                delivery_time += problem.dict_delivery_time['DEFAULT']
            if 'H' == dp[0]:
                delivery_time += problem.dict_delivery_time['HOME']
            elif 'L' == dp[0]:
                delivery_time += problem.dict_delivery_time['LOCKER']
            elif 'S' == dp[0]:
                delivery_time += problem.dict_delivery_time['SHOP']

            if len(dp_opts) > 0:
                selected = False
                for initial_time, twe, twl, twp in dp_opts:
                    departure_time = initial_time + delivery_time
                    arrival_time_depot = departure_time + problem.dict_distance[(dp, 'DEPOT')] * 60 / problem.truck_speed
                    if arrival_time <= twl and arrival_time_depot <= problem.dict_twl['DEPOT']:
                        distance += problem.dict_distance[(last_dp, dp)]
                        total_time += problem.dict_distance[(last_dp, dp)] * 60 / problem.truck_speed
                        priority += problem.dict_priority[option]
                        route.append(option)
                        route_arrival.append(arrival_time)
                        route_initial_time.append(initial_time)
                        route_departure.append(departure_time)
                        route_delivery_times.append(round(departure_time, 2))
                        served[order] = 1
                        capacity[dp] -= 1
                        miss_probs.append(1.0 - twp)
                        ind += 1
                        selected = True
                        break
                if not selected:
                    if len(route) > 1 and last_dp != 'DEPOT':
                        distance += problem.dict_distance[(last_dp, 'DEPOT')]
                        total_time += problem.dict_distance[(last_dp, 'DEPOT')] * 60 / problem.truck_speed
                        route.append((None,'DEPOT'))
                        routes.append(route)
                        delivery_times.append(route_delivery_times)
                        route = [(None,'DEPOT')]
                        route_arrival = [problem.dict_twe['DEPOT']]
                        route_initial_time = [problem.dict_twe['DEPOT']]
                        route_departure = [problem.dict_twe['DEPOT']]
                        route_delivery_times = []
                    else:
                        # DP is not reachable
                        ind += 1
            else:
                ind += 1
        else:
            ind += 1

        if ind + 1 == len(solution):
            distance += problem.dict_distance[(last_dp, 'DEPOT')]
            total_time += problem.dict_distance[(last_dp, 'DEPOT')] * 60 / problem.truck_speed
            route.append((None,'DEPOT'))
            routes.append(route)
            delivery_times.append(route_delivery_times)

    not_served_count = sum(1 for x in served.values() if x == 0)
    miss_prob = np.mean(miss_probs)

    # print("Time (seconds): ", time.time() - start)

    return priority, distance, miss_prob, routes, delivery_times, total_time, not_served_count


def simulate_routes(problem: Problem, routes: list, delivery_times: list, dict_delivery_success = None, log: bool = False):
    if dict_delivery_success is None:
        dict_delivery_success = generate_delivery_simulation(problem)
    hits = 0
    for i in range(len(routes)):
        routes_copy = copy.deepcopy(routes)
        route = routes_copy[i]
        del route[0], route[-1]
        route_delivery_times = delivery_times[i]
        for j in range(len(route)):
            order, dp = route[j]
            dt = route_delivery_times[j]
            if isinstance(problem.dict_twe[dp], list) and isinstance(problem.dict_twl[dp], list):
                for k in range(len(problem.dict_twe[dp])):
                    twe, twl = problem.dict_twe[dp][k], problem.dict_twl[dp][k]
                    if twe <= dt and dt <= twl and dict_delivery_success[dp][k] == 1:
                        hits += 1
                        break
            else:
                twe, twl = problem.dict_twe[dp], problem.dict_twl[dp]
                if twe <= dt and dt <= twl and dict_delivery_success[dp] == 1:
                    hits += 1

    hit_rate = hits / len(problem.orders)
    if log:
        print("Hit rate: ", hit_rate, "(" + str(hits) + " / " + str(len(problem.orders)) + ")" )
    return hit_rate


def generate_delivery_simulation(problem: Problem):
    dict_delivery_success = {}
    for dp, tw_prob in problem.dict_twp.items():
        if isinstance(tw_prob, list):
            success = []
            for twp in tw_prob:
                if random.random() <= twp:
                    success.append(1)
                else:
                    success.append(0)
            dict_delivery_success[dp] = success
        else:
            dict_delivery_success[dp] = 1 if random.random() <= tw_prob else 0
    return dict_delivery_success


def get_solution_charts(problem: Problem, solution: Solution, file_name: str, stats: list = [], loc_names: bool = True):
    """
    Generates the charts of a solution.

    Parameters:
        problem : Problem
            Problem instance.
        solution : Solution
            Solution instance.
        file_name : str
            File name.
        stats : list
            List of stats.
    """

    pdf = PdfPages(file_name + '.pdf')

    tws_chart = plot_solution_tws(problem, solution)
    pdf.savefig(tws_chart)

    routes_chart = plot_solution_routes(problem, solution, loc_names=loc_names)
    pdf.savefig(routes_chart)

    if len(stats):
        fitness = plot_fitness_evolution(stats)
        pdf.savefig(fitness)

    pdf.close()


def plot_solution_tws(problem: Problem, solution: Solution):

    # generate a gantt bar chart with the time windows of each stop, being the x axis the time and the y axis the stops
    plt.clf()
    plt.rcParams['figure.figsize'] = plt.rcParamsDefault['figure.figsize']
    y = 0
    colors = ["#ff0e41","#d500e0","#ff8b1f","#01befe","#7be000","#500aff","#e0c200","#20ac56", "#8b8c89", "#005fac"]
    light_colors = ["#ff9fb4","#f98dff","#ffd1a5","#9ae5fe","#ccff8d","#b99dff","#fff08d","#98ecb8", "#d0d1d0", "#78c2ff"]
    routes = copy.deepcopy(solution.routes)
    routes.reverse()
    for route in routes:
        r_index = solution.routes.index(route)
        color_index = r_index % len(colors)
        stop_exec_times = []
        stops = copy.deepcopy(route.stops)
        stops.reverse()
        for i in range(len(stops)):
            tws = stops[i].dp.time_windows
            if stops[i].dp != stops[i - 1].dp:
                y += 1
                if len(tws) == 0:
                    width = problem.depot.time_window[1] - problem.depot.time_window[0]
                    plt.barh(y, width, left=problem.depot.time_window[0], height=0.5, color=light_colors[color_index], alpha=0.5)
                else:
                    for tw in tws:
                        plt.barh(y, tw[1] - tw[0], left=tw[0], height=0.5, color=light_colors[color_index], alpha=0.5)

            plt.scatter(stops[i].exec_time, y, color=colors[color_index], marker=".", s=30)
            stop_exec_times.append((stops[i].exec_time, y))

        for i in range(len(stop_exec_times) - 1):
            plt.plot([stop_exec_times[i][0], stop_exec_times[i + 1][0]], [stop_exec_times[i][1], stop_exec_times[i + 1][1]], color=colors[color_index], linestyle='dashed', linewidth=1)


    plt.xlabel('Time')
    plt.ylabel('Delivery Points')
    plt.yticks([])  #  remove y axis ticks
    plt.title('Time Windows Chart')
    fig = plt.gcf()

    return fig



def plot_solution_routes(problem: Problem, solution: Solution, loc_names: bool = True):
    """
    Plots a solution.

    Parameters:
        problem : Problem
            Problem instance.
        solution : Solution
            Solution instance.
        file_name : str
            File name.
    """

    lockers = [dp for dp in problem.delivery_points if dp.delivery_type == DeliveryType.LOCKER]
    shops = [dp for dp in problem.delivery_points if dp.delivery_type == DeliveryType.SHOP]
    home_delivery_points = [dp for dp in problem.delivery_points if dp.delivery_type == DeliveryType.HOME]

    plt.clf()
    plt.rcParams['figure.figsize'] = plt.rcParamsDefault['figure.figsize']

    marker_size = 50
    if len(problem.delivery_points) > 500:
        marker_size = 10
        plt.figure(figsize=(8, 6))

    plt.scatter([dp.loc.x for dp in lockers], [dp.loc.y for dp in lockers], color='#7C92F3', marker='p', s=marker_size, label='Lockers')
    plt.scatter([dp.loc.x for dp in shops], [dp.loc.y for dp in shops], color='#86E9AC', marker='H', s=marker_size, label='Shops')
    plt.scatter([dp.loc.x for dp in home_delivery_points], [dp.loc.y for dp in home_delivery_points], color='#F6A2A8', marker='^', s=marker_size, label='Home Delivery Points')
    plt.scatter(problem.depot.loc.x, problem.depot.loc.y, color='black', marker='D', s=2*marker_size, label='Depot')

    colors = ["#ff0e41","#d500e0","#ff8b1f","#01befe","#7be000","#500aff","#e0c200","#20ac56", "#8b8c89", "#005fac"]

    for route in solution.routes:
        r_index = solution.routes.index(route)
        color_index = r_index % len(colors)
        for i in range(len(route.stops)):
            if i == 0:
                plt.plot([problem.depot.loc.x, route.stops[i].dp.loc.x], [problem.depot.loc.y, route.stops[i].dp.loc.y], color=colors[color_index], linestyle='dashed', linewidth=1)
            elif i == len(route.stops) - 1:
                plt.plot([route.stops[i - 1].dp.loc.x, route.stops[i].dp.loc.x], [route.stops[i - 1].dp.loc.y, route.stops[i].dp.loc.y], color=colors[color_index], linestyle='dashed', linewidth=1)
                plt.plot([route.stops[i].dp.loc.x, problem.depot.loc.x], [route.stops[i].dp.loc.y, problem.depot.loc.y], color=colors[color_index], linestyle='dashed', linewidth=1)
            else:
                plt.plot([route.stops[i - 1].dp.loc.x, route.stops[i].dp.loc.x], [route.stops[i - 1].dp.loc.y, route.stops[i].dp.loc.y], color=colors[color_index], linestyle='dashed', linewidth=1)

    if loc_names:
        for dp in problem.delivery_points:
            plt.text(dp.loc.x * 1.005, dp.loc.y, str(dp.id), fontsize=6, ha='left', va='bottom')

    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.title('Locations and Routes')
    fig = plt.gcf()

    return fig


def plot_fitness_evolution(stats_list: list):
    if isinstance(stats_list[0], float):
        stats_list = [{"min": s} for s in stats_list]

    # max_list = [s["max"] for s in stats_list]
    min_list = [s["min"] for s in stats_list]
    # avg_list = [s["avg"] for s in stats_list]

    plt.clf()
    # plt.figure(figsize=(10, 5))
    plt.plot(min_list, color="#ff0e41", label='Min fitness')
    # plt.plot(max_list, color="#005fac", label='Max')
    # plt.plot(avg_list, color="#e0c200", label='Avg')

    plt.xlabel('Iterations')
    plt.ylabel('Fitness value')
    plt.title('Fitness evolution')
    plt.legend()
    fig = plt.gcf()

    return fig


def is_dp_in_route(route: Route, dp: DeliveryPoint):
    """
    Returns True if a delivery point is in a route.

    Parameters:
        route : Route
            Route instance.
        dp : DeliveryPoint
            DeliveryPoint instance.

    Returns:
        is_in_route : bool
            True if the delivery point is in the route.
        index : int
            Index of the stop in the route.
    """

    for stop in route.stops:
        if stop.dp == dp:
            return True, route.stops.index(stop)

    return False, -1


def get_max_cost(problem: Problem):
    """
    Returns the estimated maximum cost of the problem.

    Parameters:
        problem : Problem
            Problem instance.

    Returns:
        max_cost : float
            Maximum cost of the problem.
    """

    max_cost = 0.0
    initial_options = problem.create_list_of_options()
    priority, distance, routes, total_time, not_served_count = eval_solution(problem, initial_options)

    cost = distance * problem.km_cost + len(routes) * problem.truck_cost
    max_cost = 1.3 * cost
    # print("Max cost: ", max_cost)

    return max_cost


def get_max_priority(problem: Problem):
    """
    Returns the maximum priority of the problem.

    Parameters:
        problem : Problem
            Problem instance.

    Returns:
        max_priority : float
            Maximum priority of the problem.
    """

    max_priority = 0

    for order in problem.orders:
        order_priority = len(order.delivery_options)
        max_priority += order_priority

    return max_priority


def export_to_excel(wb, name, *args):
    
    # Seleccionar la hoja activa (por defecto, la primera hoja)
    ws = wb.active
    
    # Obtener la fila actual para escribir los datos
    current_row = ws.max_row + 1
    
    # Iterar sobre los argumentos pasados y escribir cada uno en una columna diferente
    for i, arg in enumerate(args, start=1):
        # Obtener la letra de la columna correspondiente
        col_letter = get_column_letter(i)
        # Escribir el valor del argumento en la celda adecuada
        ws[col_letter + str(current_row)] = arg
    
    # Guardar el libro de trabajo en el archivo especificado
    wb.save(name)


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

    
    
def plot_clusters(destinos, labels):
    
    #Representar clusters
    # Convertir las ubicaciones a formato numpy para facilidad de uso con matplotlib
    coords = np.array([[point[0], point[1]] for point in destinos])
    print(f'coords shape: {coords.shape}')
    
    # Representar los puntos de entrega con diferentes colores según los clusters
    plt.figure(figsize=(10, 8))
    unique_labels = np.unique(labels)
    
    # Utilizar un mapa de colores con suficiente diversidad para los clusters
    colors = plt.cm.get_cmap('tab20', len(unique_labels))
    
    for label in unique_labels:
        cluster_points = coords[labels == label]
        print(f'Cluster {label} points shape: {cluster_points.shape}')
        
        plt.scatter(cluster_points[:, 0], cluster_points[:, 1], label=f'Cluster {label}', color=colors(label / len(unique_labels)))
    
        
    plt.title('Delivery Points Clusters')
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.legend()
    plt.show()



#Convert locations XY from delivery_points to a tupla.
def convert_locations(problem):
    locations = []
    for order in problem.orders:
        for option in order.delivery_options:
            dp, priority = option[0], option[1]
            locations.append([dp.loc.x, dp.loc.y])
    return np.array(locations)          
    

# Elbow method to determine optimal k
def optimal_k(locations, max_k=10):
    
    inertia = []
    for k in range(1, max_k + 1):
        kmeans = KMeans(n_clusters=k, random_state=42).fit(locations)
        inertia.append(kmeans.inertia_)
    
    plt.plot(range(1, max_k + 1), inertia, 'bx-')
    plt.xlabel('Number of clusters (k)')
    plt.ylabel('Inertia')
    plt.title('Elbow Method for Optimal k')
    plt.show()

    # Elige k basándote en el gráfico
    return int(input("Enter the optimal number of clusters (k): "))


def plot_clusters(locations, labels):
    plt.scatter(locations[:, 0], locations[:, 1], c=labels, s=50, cmap='viridis')
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.title('Clusters of Delivery Points')
    plt.show()




def two_opt(route, problem):
    def calculate_distance(route):
        total_distance = 0
        for i in range(len(route) - 1):
            total_distance += problem.dict_distance[(route[i][1], route[i+1][1])]
        return total_distance

    best_route = route
    best_distance = calculate_distance(route)
    
    improved = True
    while improved:
        improved = False
        for i in range(1, len(route) - 2):
            for j in range(i + 1, len(route)):
                if j - i == 1: continue
                new_route = route[:i] + route[i:j][::-1] + route[j:]
                new_distance = calculate_distance(new_route)
                if new_distance < best_distance:
                    best_route = new_route
                    best_distance = new_distance
                    improved = True
        route = best_route
    return best_route


def generate_initial_solution(problem, num_clusters):
    coords = np.array([[point.x, point.y] for point in problem.dict_xy.values()])
    kmeans = KMeans(n_clusters=num_clusters, random_state=0).fit(coords)
    labels = kmeans.labels_

    clusters = {i: [] for i in range(num_clusters)}
    for point, label in zip(problem.list_of_options, labels):
        clusters[label].append(point)

    initial_solution = []
    for cluster_id, points in clusters.items():
        sorted_points = sorted(points, key=lambda x: x[0].twe)
        route = []
        current_capacity = problem.truck_capacity
        for point in sorted_points:
            if current_capacity > 0:
                route.append((point[1].id, point[0].id))
                current_capacity -= 1
            else:
                initial_solution.append(route)
                route = [(point[1].id, point[0].id)]
                current_capacity = problem.truck_capacity - 1
        if route:
            initial_solution.append(route)

    return initial_solution

def evaluate_and_optimize_routes(problem, initial_solution):
    optimized_solution = []
    total_distance = 0
    total_priority = 0

    for route in initial_solution:
        optimized_route = two_opt(route, problem)
        priority, distance, _, _, _ = eval_solution(problem, optimized_route)
        optimized_solution.append(optimized_route)
        total_distance += distance
        total_priority += priority

    return optimized_solution, total_distance, total_priority

