import math
import os

from models import *
from utils import *


def generate_benchmark1(problem_type: str, n_orders: int, min_distance: int, tw_duration: int, n_trucks: int, t_speed:int, x_range: tuple, y_range: tuple, truck_cost: int = None, km_cost: int = None, file_name: str = 'benchmark', max_lockers: int = 5, max_shops: int = 5, plot=False):

    # Set number of delivery points based on problem type
    if problem_type == 'HL':
        n_homes = n_orders
        n_lockers = min(math.ceil(n_orders * 0.15), max_lockers)
        n_shops = 0
    elif problem_type == 'HLR':
        n_homes = n_orders
        n_lockers = min(math.ceil(n_orders * 0.15), max_lockers)
        n_shops = min(math.ceil(n_orders * 0.1), max_shops)
    elif problem_type == '3R':
        n_homes = int(n_orders * 2)
        n_lockers = min(math.ceil(n_orders * 0.2), max_lockers)
        n_shops = min(math.ceil(n_orders * 0.2), max_shops)
    elif problem_type == '3H':
        n_homes = 3 * n_orders
        n_lockers, n_shops = 0, 0

    # Create file directories
    if not os.path.exists(problem_type):
        os.makedirs(problem_type + '/problem_files')
        os.makedirs(problem_type + '/solution_files')
        os.makedirs(problem_type + '/results')
    
    # Open file to write problem data
    file = open(problem_type + '/' + 'problem_files/' + file_name + '.txt', 'w')

    # Generate random locations
    total_num_locations = n_homes + n_lockers + n_shops + 1  # Adding depot
    depot_location, locker_locations, shop_locations, home_locations = __generate_random_locations(
        total_num_locations, x_range, y_range, min_distance, 10000, n_homes, n_lockers, n_shops)
    depot_location = ((x_range[0] + x_range[1]) / 2, (y_range[0] + y_range[1]) / 2)
    # Generate depot information
    truck_capacity = int(10 * random.uniform(1.0, 1.5))
    depot_av_capac = truck_capacity * n_trucks
    #truck_speed = random.randint(40, 60)
    truck_speed = t_speed
    if truck_cost is None:
        truck_cost = random.randint(50, 100)
    if km_cost is None:
        km_cost = random.randint(70, 100)
    tw_e = random.choice([540, 570, 600])  # 9:00, 9:30, 10:00
    tw_l = random.choice([1260, 1290, 1320])  # 21:00, 21:30, 22:00
    
    # Write depot info to file
    file.write("NUM_VEHICLES\tTRUCK_COST\tTRUCK_SPEED\tKM_COST\tX_DEPOT\tY_DEPOT\tTW_E\tTW_L\n")
    file.write(f"{n_trucks}\t{truck_cost}\t{truck_speed}\t{km_cost}\t{depot_location[0]}\t{depot_location[1]}\t{tw_e}\t{tw_l}\n")

    # Write locker information
    if locker_locations:
        file.write("\nLOCKER_ID\tX\tY\tAV_CAPAC.\n")
        for i, loc in enumerate(locker_locations[:max_lockers]):  # Limit to max_lockers
            locker_av_capac = random.randint(5, 20)
            file.write(f"L{i+1}\t{loc[0]}\t{loc[1]}\t{locker_av_capac}\n")

    # Write shop information
    if shop_locations:
        file.write("\nSHOP_ID\tX\tY\tTW_E1\tTW_L1\tTW_E2\tTW_L2\tAV_CAPAC.\n")
        for i, loc in enumerate(shop_locations[:max_shops]):  # Limit to max_shops
            tw_e1 = random.randint(540, 660)  # Example: time window 1 for shop
            tw_l1 = tw_e1 + tw_duration
            tw_e2 = random.randint(780, 960)  # Time window 2
            tw_l2 = tw_e2 + tw_duration
            shop_av_capac = random.randint(10, 50)
            file.write(f"S{i+1}\t{loc[0]}\t{loc[1]}\t{tw_e1}\t{tw_l1}\t{tw_e2}\t{tw_l2}\t{shop_av_capac}\n")

    # Write home delivery points and generate orders
    if home_locations:
        home_delivery_points = __generate_home_delivery_points(home_locations, depot_location, (tw_e, tw_l), truck_speed, tw_duration)
        file.write("\nORDER_ID\tWEIGHT\tVOLUME\tPRIORITY\tDP_ID\tX\tY\tTW_PROBABILITIES\n")
        
        # Generate orders with limits on lockers and shops
        orders = __generate_orders(n_orders, locker_locations[:max_lockers], shop_locations[:max_shops], home_delivery_points, max_total_lockers=max_lockers, max_total_shops=max_shops, problem_type=problem_type)

        # Write each order to the file
        for order in orders:
            if isinstance(order[4], str):  # Check if DP_ID is a string
                if order[4][0] == 'L' or order[4][0] == 'S':
                    file.write(f"{order[0]}\t{order[1]}\t{order[2]}\t{order[3]}\t{order[4]}\n")
                else:
                    file.write(f"{order[0]}\t{order[1]}\t{order[2]}\t{order[3]}\t{order[4]}\t{order[5]}\t{order[6]}\t{order[7]}\n")

    # Close file
    file.close()

    # Plot the problem (optional)
    if plot:
        __plot_problem_from_tuples(depot_location, locker_locations, shop_locations, home_locations, problem_type + '/problem_files/' + file_name)
        print('Plot saved to ' + problem_type + '/problem_files/' + file_name + '.pdf')



def generate_problem_file(n_orders: int, n_delivery_points: int, n_trucks: int, x_range: tuple, y_range=tuple, file_name: str = 'problem', plot=False):
    """
    Generates a problem file.

    Parameters:
        n_orders : int
            Number of orders.
        n_delivery_points : int
            Number of delivery points.
        n_trucks : int
            Number of trucks.
        x_range : tuple
            Range of x coordinates.
        y_range : tuple
            Range of y coordinates.
        file_name : str
            File name.
        plot : bool
            If True, the problem is plotted.
    """

    # Create file
    file = open(file_name + '.txt', 'w')

    # Generate locations
    depot_location, locker_locations, shop_locations, home_locations = __generate_random_locations(
        n_delivery_points, x_range, y_range, 2, 10000)

    # Generate depot
    truck_capacity = int(n_orders / n_trucks * random.uniform(1.1, 1.5))
    depot_av_capac = truck_capacity * n_trucks
    truck_cost = random.randint(50, 100)
    truck_speed = random.randint(40, 60)
    km_cost = random.randint(70, 100)
    tw_e = random.choice([540, 570, 600])  # 9:00, 9:30, 10:00
    tw_l = random.choice([1260, 1290, 1320])  # 21:00, 21:30, 22:00
    file.write(
        "NUM_VEHICLES\tTRUCK_COST\tTRUCK_SPEED\tKM_COST\tX_DEPOT\tY_DEPOT\tTW_E\tTW_L\n")
    file.write(
        f"{n_trucks}\t{truck_cost}\t{truck_speed}\t{km_cost}\t{depot_location[0]}\t{depot_location[1]}\t{tw_e}\t{tw_l}\n")

    # Generate lockers (15 % of delivery points)
    file.write("\nLOCKER_ID\tX\tY\tAV_CAPAC.\n")
    lockers = []
    for loc in locker_locations:
        locker_av_capac = random.randint(5, 30)
        lockers.append(
            (f"L{len(lockers) + 1}", loc[0], loc[1], locker_av_capac))

    for locker in lockers:
        file.write(f"{locker[0]}\t{locker[1]}\t{locker[2]}\t{locker[3]}\n")

    # Generate shops (15 % of delivery points)
    file.write("\nSHOP_ID\tX\tY\tTW_E1\tTW_L1\tTW_E2\tTW_L2\tAV_CAPAC.\n")
    shops = __generate_shops(shop_locations)

    for shop in shops:
        file.write(f"{shop[0]}\t{shop[1]}\t{shop[2]}\t{shop[3]}\t{shop[4]}\n")

    # Generate home delivery points (70 % of delivery points)
    home_delivery_points = __generate_home_delivery_points(home_locations, depot_location, (tw_e, tw_l), truck_speed)

    # Generate orders
    file.write("\nORDER_ID\tWEIGHT\tVOLUME\tPRIORITY\tDP_ID\tX\tY\tTW_PROBABILITIES\n")
    orders = __generate_orders(n_orders, lockers, shops, home_delivery_points)
    for order in orders:
        if order[4][0] == 'L' or order[4][0] == 'S':
            file.write(
                f"{order[0]}\t{order[1]}\t{order[2]}\t{order[3]}\t{order[4]}\n")
        else:
            file.write(
                f"{order[0]}\t{order[1]}\t{order[2]}\t{order[3]}\t{order[4]}\t{order[5]}\t{order[6]}\t{order[7]}\n")

    # Close file
    file.close()

    # Plot
    if plot:
        __plot_problem_from_tuples(depot_location, locker_locations, shop_locations, home_locations, file_name)


def __euclidean_distance(point1: tuple, point2: tuple):
    """
    Returns the Euclidean distance between two points.

    Parameters:
        point1 : tuple
            First point.
        point2 : tuple
            Second point.

    Returns:
        distance : float
            Euclidean distance between the two points.
    """
    x1, y1 = point1
    x2, y2 = point2
    return np.sqrt((x2 - x1)**2 + (y2 - y1)**2)


def __find_central_location(locations: list):
    """
    Returns the central location (depot).

    Parameters:
        locations : list
            List of locations.

    Returns:
        central_location : tuple
            Central location.
    """

    centroid = np.mean(locations, axis=0)
    central_location = min(
        locations, key=lambda loc: np.linalg.norm(loc - centroid))
    return central_location


import random
import math

def __euclidean_distance(loc1, loc2):
    return math.sqrt((loc1[0] - loc2[0]) ** 2 + (loc1[1] - loc2[1]) ** 2)

def __find_central_location(locations):
    x_coords, y_coords = zip(*locations)
    return (sum(x_coords) / len(x_coords), sum(y_coords) / len(y_coords))

def __generate_random_locations(max_num_locations: int, range_x: tuple, range_y: tuple, min_distance: int, max_attempts: int, n_homes: int = None, n_lockers: int = None, n_shops: int = None):
    """
    Generates random locations with specified numbers of homes, lockers, and shops.

    Parameters:
        max_num_locations : int
            Maximum number of locations.
        range_x : tuple
            Range of x coordinates.
        range_y : tuple
            Range of y coordinates.
        min_distance : int
            Minimum distance between locations.
        max_attempts : int
            Maximum number of attempts to generate a location.
        n_homes : int
            Number of home delivery locations required.
        n_lockers : int
            Number of locker locations required.
        n_shops : int
            Number of shop locations required.

    Returns:
        depot_location : tuple
            Depot location.
        locker_locations : list
            List of locker locations.
        shop_locations : list
            List of shop locations.
        home_locations : list
            List of home delivery point locations.

    Raises:
        ValueError: If the required number of lockers, shops, or homes cannot be generated within the constraints.
    """
    total_required = (n_homes or 0) + (n_lockers or 0) + (n_shops or 0) + 1  # +1 for the depot
    if total_required > max_num_locations:
        raise ValueError(f"Total required locations ({total_required}) exceed max_num_locations ({max_num_locations})")

    locations = []
    attempts = 0

    while len(locations) < max_num_locations and attempts < max_attempts:
        x = random.uniform(*range_x) if range_x[1] - range_x[0] < 20 else random.randint(*range_x)
        y = random.uniform(*range_y) if range_y[1] - range_y[0] < 20 else random.randint(*range_y)
        new_location = (round(x, 2), round(y, 2))

        if not any(__euclidean_distance(new_location, loc) < min_distance for loc in locations):
            locations.append(new_location)
            attempts = 0
        else:
            attempts += 1

    if len(locations) < total_required:
        raise ValueError("Unable to generate the required number of unique locations within the specified constraints.")

    # Find the central location (depot)
    depot_location = __find_central_location(locations)

    # Divide the locations into three groups (randomly): lockers, shops, and homes
    available_locations = locations.copy()
    if depot_location in available_locations:
        available_locations.remove(depot_location)

    def assign_locations(available, num_required):
        assigned = []
        attempts = 0
        while len(assigned) < num_required and attempts < max_attempts:
            loc = random.choice(available)
            assigned.append(loc)
            available.remove(loc)
            attempts += 1
        if len(assigned) < num_required:
            raise ValueError(f"Unable to assign the required {num_required} locations within the given constraints.")
        return assigned

    locker_locations = assign_locations(available_locations, n_lockers or int(len(locations) * 0.15))
    shop_locations = assign_locations(available_locations, n_shops or int(len(locations) * 0.15))
    home_locations = available_locations  # Remainder goes to homes

    # Check if the number of homes meets the requirement
    if n_homes and len(home_locations) < n_homes:
        raise ValueError(f"Unable to assign the required number of home locations ({n_homes}) with the current configuration.")

    return depot_location, locker_locations, shop_locations, home_locations



def __generate_shops(shop_locations: list, depot_location: tuple, depot_tw: tuple, truck_speed: int = 50, tw_duration: int = None):
    """
    Generates shops.

    Parameters:
        shop_locations : list
            List of shop locations.

    Returns:
        shops : list
            List of shops.
    """
    shops = []

    # some random time windows for shops
    shop_tws = []

    morning_start_times = [540, 600, 630, 660]  # 9:00, 10:00, 10:30, 11:00
    morning_end_times = [840, 870, 900, 1020]  # 14:00, 14:30, 15:00, 17:00
    afternoon_start_times = [1020, 1050, 1080]  # 17:00, 17:30, 18:00
    # 20:30, 21:00, 21:30, 22:00
    afternoon_end_times = [1230, 1260, 1290, 1320]

    for i in range(5):
        shop_tws.append(
            f'{random.choice(morning_start_times)}\t{random.choice(morning_end_times)}\t-\t-')
        shop_tws.append(
            f'{random.choice(afternoon_start_times)}\t{random.choice(afternoon_end_times)}\t-\t-')
        shop_tws.append(
            f'{random.choice(morning_start_times)}\t{random.choice(morning_end_times)}\t{random.choice(afternoon_start_times)}\t{random.choice(afternoon_end_times)}')
        shop_tws.append(
            f'{random.choice(morning_start_times)}\t{random.choice(afternoon_end_times)}\t-\t-')

    for loc in shop_locations:
        time_to_depot = __euclidean_distance(loc, depot_location) / truck_speed * 60
        tws = random.choice(shop_tws)

        #check if time windows are valid
        valid_tw = False
        while not valid_tw:
            separated_tws = tws.split('\t')
            if int(separated_tws[1]) + time_to_depot <= depot_tw[1]:
                valid_tw = True

            if separated_tws[2] != '-' and separated_tws[3] != '-':
                if int(separated_tws[3]) + time_to_depot <= depot_tw[1]:
                    valid_tw = True

            if not valid_tw:
                tws = random.choice(shop_tws)

        av_capac = random.randint(1, 30)
        shops.append((f"S{len(shops) + 1}", loc[0], loc[1], tws, av_capac))

    return shops


def __generate_home_delivery_points(home_locations: list, depot_location: tuple, depot_tw: tuple, truck_speed: int = 50, tw_duration: int = None):
    """
    Generates home delivery points.

    Parameters:
        home_locations : list
            List of home delivery point locations.

    Returns:
        home_delivery_points : list
            List of home delivery points.
    """

    home_delivery_points = []
    home_delivery_tws = []

    if tw_duration is None:
        morning_start_times = [540, 600, 630, 660]
        morning_end_times = [840, 870, 900, 1020]
        afternoon_start_times = [1020, 1050, 1080]
        afternoon_end_times = [1230, 1260, 1290, 1320]

        for i in range(8):
            home_delivery_tws.append(f'{random.choice(morning_start_times)},{random.choice(morning_end_times)},{round(random.uniform(0.2, 1.0), 2)}')
            home_delivery_tws.append(f'{random.choice(afternoon_start_times)},{random.choice(afternoon_end_times)},{round(random.uniform(0.2, 1.0), 2)}')
            home_delivery_tws.append(f'{random.choice(morning_start_times)},{random.choice(morning_end_times)},{round(random.uniform(0.2, 1.0), 2)}; {random.choice(afternoon_start_times)},{random.choice(afternoon_end_times)},{round(random.uniform(0.2, 1.0), 2)}')
            home_delivery_tws.append(f'{random.choice(morning_start_times)},{random.choice(afternoon_end_times)},{round(random.uniform(0.2, 1.0), 2)}')

        for i in range(len(home_locations)):
            tw_probabilities = random.choice(home_delivery_tws)
            home_delivery_points.append(
                (f"H{len(home_delivery_points) + 1}", home_locations[i][0], home_locations[i][1], tw_probabilities))
    else:
        morning_start_times = [540, 570, 600, 630, 660, 690, 720]
        afternoon_start_times = [840, 870, 900, 960]
        evening_start_times = [1020, 1050, 1080, 1140]

        if tw_duration < 240:
            start_times = morning_start_times + afternoon_start_times + evening_start_times
        else:
            start_times = morning_start_times + afternoon_start_times

        for i in range(len(home_locations)):
            time_to_depot = __euclidean_distance(home_locations[i], depot_location) / truck_speed * 60
            start = random.choice(start_times)

            # Debug: Mostrar distancia y tiempo al depot
            print(f"Home location {i + 1}: distancia al depot = {__euclidean_distance(home_locations[i], depot_location):.2f}, tiempo estimado al depot = {time_to_depot:.2f} minutos")

            valid_tw = False
            attempt_count = 0  # Añadimos un contador de intentos
            max_attempts = 50  # Número máximo de intentos para encontrar una ventana de tiempo válida

            while not valid_tw and attempt_count < max_attempts:
                attempt_count += 1
                if start + tw_duration + time_to_depot <= depot_tw[1] and time_to_depot <= tw_duration:
                    valid_tw = True
                else:
                    start = random.choice(start_times)
                    # Debug: Mostrar el intento fallido
                    print(f"Intento {attempt_count} fallido para la ubicación {i + 1}: start = {start}, duración TW = {tw_duration}, tiempo al depot = {time_to_depot}")

            if valid_tw:
                tw_and_prob = f'{start},{start + tw_duration},{round(random.uniform(0.2, 1.0), 2)}'
                home_delivery_points.append((f"H{len(home_delivery_points) + 1}", home_locations[i][0], home_locations[i][1], tw_and_prob))
            else:
                print(f"Advertencia: no se encontró una ventana de tiempo válida para la ubicación {i + 1} tras {max_attempts} intentos.")

    return home_delivery_points


def __generate_home_delivery_points1(home_locations: list, depot_location: tuple, depot_tw: tuple, truck_speed: int = 50, tw_duration: int = None):
    """
    Generates home delivery points.

    Parameters:
        home_locations : list
            List of home delivery point locations.

    Returns:
        home_delivery_points : list
            List of home delivery points.
    """

    # DP_ID	X	Y	TW_PROBABILITIES
    # H1  25  78  600,840,0.6; 1200,1450,0.4
    home_delivery_points = []
    home_delivery_tws = []

    if tw_duration is None:
        # Random time windows for home delivery points
        morning_start_times = [540, 600, 630, 660]  # 9:00, 10:00, 10:30, 11:00
        morning_end_times = [840, 870, 900, 1020]  # 14:00, 14:30, 15:00, 17:00
        afternoon_start_times = [1020, 1050, 1080]  # 17:00, 17:30, 18:00
        afternoon_end_times = [1230, 1260, 1290, 1320] # 20:30, 21:00, 21:30, 22:00

        for i in range(8):
            home_delivery_tws.append(
                f'{random.choice(morning_start_times)},{random.choice(morning_end_times)},{round(random.uniform(0.2, 1.0), 2)}')
            home_delivery_tws.append(
                f'{random.choice(afternoon_start_times)},{random.choice(afternoon_end_times)},{round(random.uniform(0.2, 1.0), 2)}')
            home_delivery_tws.append(
                f'{random.choice(morning_start_times)},{random.choice(morning_end_times)},{round(random.uniform(0.2, 1.0), 2)}; {random.choice(afternoon_start_times)},{random.choice(afternoon_end_times)},{round(random.uniform(0.2, 1.0), 2)}')
            home_delivery_tws.append(
                f'{random.choice(morning_start_times)},{random.choice(afternoon_end_times)},{round(random.uniform(0.2, 1.0), 2)}')

        for i in range(len(home_locations)):
            tw_probabilities = random.choice(home_delivery_tws)
            home_delivery_points.append(
                (f"H{len(home_delivery_points) + 1}", home_locations[i][0], home_locations[i][1], tw_probabilities))
    else:
        # Fixed time windows duration for home delivery points
        morning_start_times = [540, 570, 600, 630, 660, 690, 720] # 9:00, 9:30, 10:00, 10:30, 11:00, 11:30, 12:00
        afternoon_start_times = [840, 870, 900, 960] # 14:00, 14:30, 15:00, 16:00
        evening_start_times = [1020, 1050, 1080, 1140] # 17:00, 17:30, 18:00, 19:00

        if tw_duration < 240:
            start_times = morning_start_times + afternoon_start_times + evening_start_times
        else:
            start_times = morning_start_times + afternoon_start_times

        for i in range(len(home_locations)):
            time_to_depot = __euclidean_distance(home_locations[i], depot_location) / truck_speed * 60
            start = random.choice(start_times)

            #check if the time window is valid
            valid_tw = False
            while not valid_tw:
                # print(start, tw_duration, time_to_depot, depot_tw[1], depot_location, home_locations[i])
                # print(start + tw_duration + time_to_depot <= depot_tw[1], time_to_depot <= tw_duration, "\n")
                if start + tw_duration + time_to_depot <= depot_tw[1] and time_to_depot <= tw_duration:
                    valid_tw = True
                else:
                    start = random.choice(start_times)

            tw_and_prob = f'{start},{start + tw_duration},{round(random.uniform(0.2, 1.0), 2)}'

            home_delivery_points.append(
                (f"H{len(home_delivery_points) + 1}", home_locations[i][0], home_locations[i][1], tw_and_prob))

    return home_delivery_points



import time

def __generate_orders(n_orders: int, lockers: list, shops: list, home_delivery_points: list, max_total_lockers: int = 10, max_total_shops: int = 10, problem_type: str = None):
    """
    Generates orders with exactly 3 delivery options for each order.
    Each order has one home delivery point, one locker (if available), and one shop (if available).

    Parameters:
        n_orders : int
            Number of orders.
        lockers : list
            List of lockers.
        shops : list
            List of shops.
        home_delivery_points : list
            List of home delivery points.
        max_total_lockers : int
            Maximum number of unique locker locations.
        max_total_shops : int
            Maximum number of unique shop locations.
        problem_type : str
            Problem type ('HL', 'HLR', '3R', '3H').

    Returns:
        orders : list
            List of orders, each with 3 delivery options (Home, Locker, Shop).
    """

    orders = []
    used_hd_points = set()

    # Limita lockers y tiendas a la cantidad máxima especificada
    if len(lockers) > max_total_lockers:
        lockers = random.sample(lockers, max_total_lockers)
    if len(shops) > max_total_shops:
        shops = random.sample(shops, max_total_shops)

    # Verifica que haya suficientes viviendas
    if len(home_delivery_points) < n_orders:
        raise ValueError("No hay suficientes puntos de viviendas disponibles para generar las órdenes.")

    for i in range(n_orders):
        if i % 10 == 0:
            print(f"Generando orden {i+1}/{n_orders}")

        order_id = f"O{i + 1}"
        weight = random.randint(1, 10)
        volume = random.randint(1, 10)
        priority = 1
        options = []

        # Asigna siempre una vivienda
        dp_id, x, y, tw_probabilities = random.choice(home_delivery_points)
        
        # Eliminar "HH" usando f"H{dp_id[1:]}" si dp_id ya contiene una "H"
        if dp_id.startswith("H"):
            dp_id = dp_id[1:]
        home_option = (order_id, weight, volume, priority, f"H{dp_id}", x, y, tw_probabilities)
        
        options.append(home_option)

        # Asigna un locker si hay disponible
        if lockers:
            locker_choice = random.choice(lockers)
            dp_id = f"L{lockers.index(locker_choice) + 1}"  # Locker ID basado en la posición
            locker_option = (order_id, weight, volume, priority + 1, dp_id)
            options.append(locker_option)

        # Asigna una tienda si hay disponible
        if shops:
            shop_choice = random.choice(shops)
            dp_id = f"S{shops.index(shop_choice) + 1}"  # Shop ID basado en la posición
            shop_option = (order_id, weight, volume, priority + 2, dp_id)
            options.append(shop_option)

        # Asegura que haya exactamente 3 opciones por orden
        if len(options) < 3:
            # Añade una segunda vivienda si no hay lockers o tiendas suficientes
            dp_id, x, y, tw_probabilities = random.choice(home_delivery_points)
            while len(options) < 3:
                if f"H{dp_id}" not in [opt[4] for opt in options]:  # Verifica si la vivienda no está ya en las opciones
                    home_option = (order_id, weight, volume, priority + len(options), f"H{dp_id}", x, y, tw_probabilities)
                    options.append(home_option)

        orders.extend(options)

    print(f"Órdenes generadas correctamente")
    return orders





#Codigo antiguo
def __generate_orders1(n_orders: int, lockers: list, shops: list, home_delivery_points: list, problem_type: str = None):
    """
    Generates orders.

    Parameters:
        n_orders : int
            Number of orders.
        lockers : list
            List of lockers.
        shops : list
            List of shops.
        home_delivery_points : list
            List of home delivery points.

    Returns:
        orders : list
            List of orders.
    """

    # ORDER_ID	WEIGHT	VOLUME	PRIORITY	DP_ID	X	Y	TW_PROBABILITIES
    orders = []
    used_hd_points = []

    for i in range(n_orders):
        order_id = f"O{i + 1}"
        used_lockers = []
        used_shops = []
        has_home_delivery = False
        has_locker_delivery = False
        has_shop_delivery = False
        weight = random.randint(1, 10)
        volume = random.randint(1, 10)

        if problem_type == 'HL':
            num_options = 2
        elif problem_type == 'HLR' or problem_type == '3R' or problem_type == '3H':
            num_options = 3
        else:
            num_options = random.randint(1, 5)

        priority = 1
        order_dp_ids = []
        for j in range(num_options):

            if problem_type != None:
                delivery_type = __select_delivery_type(problem_type, has_home_delivery, has_locker_delivery, has_shop_delivery)
            else:
                if has_home_delivery:
                    delivery_type = random.choice(["L", "S"])
                else:
                    delivery_type = random.choice(["L", "S", "H"])
            x = 0
            y = 0
            tw_probabilities = ""

            if delivery_type == "L":
                dp_id = random.choice(lockers)[0]
                while (dp_id in used_lockers and len(used_lockers) != len(lockers)) or dp_id in order_dp_ids:
                    dp_id = random.choice(lockers)[0]
                if dp_id not in used_lockers:
                    used_lockers.append(dp_id)
                orders.append((order_id, weight, volume, priority, dp_id))
                order_dp_ids.append(dp_id)
                has_locker_delivery = True

            elif delivery_type == "S":
                dp_id = random.choice(shops)[0]
                while (dp_id in used_shops and len(used_shops) != len(shops)) or dp_id in order_dp_ids:
                    dp_id = random.choice(shops)[0]
                if dp_id not in used_shops:
                    used_shops.append(dp_id)
                orders.append((order_id, weight, volume, priority, dp_id))
                order_dp_ids.append(dp_id)
                has_shop_delivery = True

            elif delivery_type == "H":
                dp_id, x, y, tw_probabilities = random.choice(home_delivery_points)
                while (dp_id in used_hd_points and len(used_hd_points) != len(home_delivery_points)) or dp_id in order_dp_ids:
                    dp_id, x, y, tw_probabilities = random.choice(home_delivery_points)
                if dp_id not in used_hd_points:
                    used_hd_points.append(dp_id)
                orders.append((order_id, weight, volume, priority, dp_id, x, y, tw_probabilities))
                order_dp_ids.append(dp_id)
                has_home_delivery = True

            priority += 1

    return orders


def __select_delivery_type(problem_type, has_home_delivery, has_locker_delivery, has_shop_delivery):
    if problem_type == 'HL':
        if has_home_delivery:
            delivery_type = "L"
        elif has_locker_delivery:
            delivery_type = "H"
        else:
            delivery_type = random.choice(["H", "L"])

    elif problem_type == 'HLR':
        if has_home_delivery and not has_locker_delivery:
            delivery_type = "L"
        elif has_locker_delivery and not has_home_delivery:
            delivery_type = "H"
        elif not has_shop_delivery:
            delivery_type = random.choice(["H", "L", "S"])
        else:
            delivery_type = random.choice(["H", "L"])

    elif problem_type == '3R':
        delivery_type = random.choice(["H", "L", "S"])
    elif problem_type == '3H':
        delivery_type = "H"

    return delivery_type


def __read_problem_file_as_tuples(file_path: str):
    """
    Reads a problem file and returns the problem data as tuples.

    Parameters:
        file_path : str
            File path.

    Returns:
        depot_location : tuple
            Depot location.
        locker_locations : list
            List of locker locations.
        shop_locations : list
            List of shop locations.
        home_locations : list
            List of home delivery point locations.
        lockers : list
            List of lockers.
        shops : list
            List of shops.
        home_delivery_points : list
            List of home delivery points.
        orders : list
            List of orders.
    """

    with open(file_path, 'r') as f:
        lines = f.readlines()

    num_vehicles = int(lines[1].split()[0])
    av_capacity = int(lines[1].split()[1])
    truck_cost = int(lines[1].split()[2])
    km_cost = int(lines[1].split()[3])
    depot_location = tuple(map(int, lines[1].split()[4:6]))
    depot_tw = tuple(map(int, lines[1].split()[6:8]))

    locker_locations = []
    shop_locations = []
    home_locations = []
    lockers = []
    shops = []
    home_delivery_points = []
    orders = []

    for line in lines[3:]:
        strings = line.split()
        strings = [string.strip() for string in strings]

        if line[0] == 'L' and not 'LOCKER_ID' in line:
            lockers.append(
                (strings[0], tuple(map(int, strings[1:3])), int(strings[3])))
            locker_locations.append(tuple(map(int, strings[1:3])))

        elif line[0] == 'S' and not 'SHOP_ID' in line:
            tws = []
            tws.append(tuple(map(int, strings[3:5])))
            if strings[5] != '-':
                tws.append(tuple(map(float, strings[5:7])))

            shops.append(
                (line[0], tuple(map(int, strings[1:3])), tws, int(strings[7])))
            shop_locations.append(tuple(map(int, strings[1:3])))

        elif line[0] == 'O' and not 'ORDER_ID' in line:
            if strings[4][0] == 'L' or strings[4][0] == 'S':
                orders.append(((strings[0]), int(strings[1]), int(
                    strings[2]), int(strings[3]), strings[4]))
            elif strings[4][0] == 'H':
                tws = []
                str_tws_probs = strings[7:]
                for str in str_tws_probs:
                    str = str.replace(';', '')
                    nums = str.split(',')
                    tws.append((int(nums[0]), int(nums[1]), float(nums[2])))
                orders.append(((strings[0]), int(strings[1]), int(strings[2]), int(
                    strings[3]), strings[4], int(strings[5]), int(strings[6]), tws))
                home_locations.append(tuple(map(int, strings[5:7])))

    for order in orders:
        print(order)

    return depot_location, locker_locations, shop_locations, home_locations, lockers, shops, home_delivery_points, orders


def __plot_problem_from_tuples(depot_location: tuple, locker_locations: list, shop_locations: list, home_locations: list, file_name: str = 'problem'):
    """
    Plots a problem using data in tuple format.

    Parameters:
        depot_location : tuple
            Depot location.
        locker_locations : list
            List of locker locations.
        shop_locations : list
            List of shop locations.
        home_locations : list
            List of home delivery point locations.
        file_name : str
            File name.
    """

    plt.clf()
    plt.scatter([loc[0] for loc in locker_locations], [loc[1]
                for loc in locker_locations], color='#7C92F3', marker='p', s=75, label='Lockers')
    plt.scatter([loc[0] for loc in shop_locations], [loc[1]
                for loc in shop_locations], color='#86E9AC', marker='H', s=75, label='Shops')
    plt.scatter([loc[0] for loc in home_locations], [loc[1] for loc in home_locations],
                color='#F6A2A8', marker='o', s=20, label='Home Delivery Points')
    plt.scatter(depot_location[0], depot_location[1],
                color='black', marker='D', s=100, label='Depot')

    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.title('Problem Locations')
    plt.savefig(file_name + '_plot.pdf', format='pdf')
