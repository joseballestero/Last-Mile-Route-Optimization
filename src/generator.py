import math
import os

from models import *
from utils import *


def generate_benchmark(problem_type: str, n_orders: int, min_distance: int, tw_duration: int, n_trucks: int, x_range: tuple, y_range: tuple, truck_cost: int = None, km_cost: int = None, file_name: str = 'benchmark', plot=False):

    # set number of delivery points of each type depending on the problem type
    # problem_type = 'HL' -> each order has a home delivery point and a locker
    # problem_type = 'HLR' -> each order has a home delivery point, a locker and a random delivery point (home, locker or shop)
    # problem_type = '3R' -> each order has 3 random delivery points (home, locker or shop)
    # problem_type = '3H' -> each order has 3 home delivery points

    if problem_type == 'HL':
        n_homes = n_orders
        n_lockers = math.ceil(n_orders * 0.15)
        n_shops = 0
    elif problem_type == 'HLR':
        n_homes = n_orders
        n_lockers = math.ceil(n_orders * 0.15)
        n_shops = math.ceil(n_orders * 0.1)
    elif problem_type == '3R':
        n_homes = math.ceil(n_orders * 0.8)
        n_lockers = math.ceil(n_orders * 0.2)
        n_shops = math.ceil(n_orders * 0.2)
    elif problem_type == '3H':
        n_homes = 3 * n_orders
        n_lockers, n_shops = 0, 0

    # Create file
    if not os.path.exists(problem_type):
        os.makedirs(problem_type + '/problem_files')
        os.makedirs(problem_type + '/solution_files')
        os.makedirs(problem_type + '/results')
    file = open(problem_type + '/' + 'problem_files/' +  file_name + '.txt', 'w')

    # Generate locations
    total_num_locations = n_homes + n_lockers + n_shops + 1
    depot_location, locker_locations, shop_locations, home_locations = __generate_random_locations(
        total_num_locations, x_range, y_range, min_distance, 10000, n_homes, n_lockers, n_shops)

    # Generate depot
    truck_capacity = int(10 * random.uniform(1.0, 1.5))
    depot_av_capac = truck_capacity * n_trucks
    truck_speed = random.randint(40, 60)
    if truck_cost == None:
        truck_cost = random.randint(50, 100)
    if km_cost == None:
        km_cost = random.randint(70, 100)
    tw_e = random.choice([540, 570, 600])  # 9:00, 9:30, 10:00
    tw_l = random.choice([1260, 1290, 1320])  # 21:00, 21:30, 22:00
    file.write("NUM_VEHICLES\tTRUCK_COST\tTRUCK_SPEED\tKM_COST\tX_DEPOT\tY_DEPOT\tTW_E\tTW_L\n")
    file.write(f"{n_trucks}\t{truck_cost}\t{truck_speed}\t{km_cost}\t{depot_location[0]}\t{depot_location[1]}\t{tw_e}\t{tw_l}\n")

    # Generate lockers
    file.write("\nLOCKER_ID\tX\tY\tAV_CAPAC.\n")
    lockers = []
    for loc in locker_locations:
        locker_av_capac = random.randint(5, 20)
        lockers.append((f"L{len(lockers) + 1}", loc[0], loc[1], locker_av_capac))
    for locker in lockers:
        file.write(f"{locker[0]}\t{locker[1]}\t{locker[2]}\t{locker[3]}\n")

    # Generate shops
    file.write("\nSHOP_ID\tX\tY\tTW_E1\tTW_L1\tTW_E2\tTW_L2\tAV_CAPAC.\n")
    shops = __generate_shops(shop_locations, depot_location, (tw_e, tw_l), truck_speed, tw_duration)
    for shop in shops:
        file.write(f"{shop[0]}\t{shop[1]}\t{shop[2]}\t{shop[3]}\t{shop[4]}\n")

    # Generate home delivery points
    home_delivery_points = __generate_home_delivery_points(home_locations, depot_location, (tw_e, tw_l), truck_speed, tw_duration)

    # Generate orders
    file.write("\nORDER_ID\tWEIGHT\tVOLUME\tPRIORITY\tDP_ID\tX\tY\tTW_PROBABILITIES\n")
    orders = __generate_orders(n_orders, lockers, shops, home_delivery_points, problem_type)

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
        __plot_problem_from_tuples(
            depot_location, locker_locations, shop_locations, home_locations, problem_type + '/problem_files/' + file_name)
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


def __generate_random_locations(max_num_locations: int, range_x: tuple, range_y: tuple, min_distance: int, max_attempts: int, n_homes: int = None, n_lockers: int = None, n_shops: int = None):
    """
    Generates random locations.

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

    Returns:
        depot_location : tuple
            Depot location.
        locker_locations : list
            List of locker locations.
        shop_locations : list
            List of shop locations.
        home_locations : list
            List of home delivery point locations.
    """
    locations = []
    attempts = 0

    while len(locations) < max_num_locations and attempts < max_attempts:
        if range_x[1] - range_x[0] < 20:
            x = random.uniform(*range_x)
            y = random.uniform(*range_y)
            new_location = (round(x, 2), round(y, 2))
        else:
            x = random.randint(*range_x)
            y = random.randint(*range_y)
            new_location = (x, y)

        if not any(__euclidean_distance(new_location, loc) < min_distance for loc in locations):
            locations.append(new_location)
            attempts = 0
        else:
            attempts += 1

    # Find the central location (depot)
    depot_location = __find_central_location(locations)

    # Divide the locations into three groups (randomly): lockers, shops and home delivery points
    avaliable_locations = locations.copy()
    avaliable_locations.remove(depot_location)
    locker_locations = []
    shop_locations = []
    home_locations = []

    if n_homes is not None and n_lockers is not None and n_shops is not None:
        print('n_homes=' + str(n_homes), 'n_lockers=' + str(n_lockers), 'n_shops=' + str(n_shops), 'total=' + str(len(avaliable_locations)))
        for i in range(n_lockers):
            locker_location = random.choice(avaliable_locations)
            locker_locations.append(locker_location)
            avaliable_locations.remove(locker_location)

        for i in range(n_shops):
            shop_location = random.choice(avaliable_locations)
            shop_locations.append(shop_location)
            avaliable_locations.remove(shop_location)
    else:
        for i in range(int(len(locations) * 0.15)):
            locker_location = random.choice(avaliable_locations)
            locker_locations.append(locker_location)
            avaliable_locations.remove(locker_location)

        for i in range(int(len(locations) * 0.15)):
            shop_location = random.choice(avaliable_locations)
            shop_locations.append(shop_location)
            avaliable_locations.remove(shop_location)

    home_locations = avaliable_locations

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


def __generate_orders(n_orders: int, lockers: list, shops: list, home_delivery_points: list, problem_type: str = None):
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
                color='#F6A2A8', marker='^', s=75, label='Home Delivery Points')
    plt.scatter(depot_location[0], depot_location[1],
                color='black', marker='D', s=100, label='Depot')

    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.title('Problem Locations')
    plt.savefig(file_name + '_plot.pdf', format='pdf')
