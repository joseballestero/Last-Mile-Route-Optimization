"""This module contains the classes that represent the entities of the problem."""

import uuid
import numpy as np

from enum import Enum
from math import dist


class Customer:
    """
    Customer class represents a customer of the problem.

    Parameters:
        id : str
            Customer id. If no id is provided, a random uuid will be generated.

    Attributes:
        id : str
            Customer id.
        mean_satisfaction : float
            Mean satisfaction of the customer.
    """

    def __init__(self, id = None):
        self.id = id
        self.mean_satisfaction = None

        if self.id is None:
            self.id = str(uuid.uuid4())

    def __str__(self):
        return f"Customer(id={self.id})"

    def __repr__(self):
        return self.__str__()

    def __eq__(self, other):
        return self.id == other.id

    def __hash__(self):
        return hash(self.id)


class Order:
    """
    Order class represents an order of the problem.

    Parameters:
        customer: Customer
            The customer who creates the order.
        volume: int
            The amount of space that the order occupies.
        weight: int
            The amount that the order weights.
        delivery_deadline: int
            The time limit for the order to be delivered.
        delivery_options: list
            The list of possible delivery options for the order.
        id: str
            The id of the order. If no id is provided, a random uuid will be generated.

    Attributes:
        id : str

        customer: Customer
            The customer who creates the order.
        volume: int
            The amount of space that the order occupies.
        weight: int
            The amount that the order weights.
        delivery_deadline: int
            The time limit for the order to be delivered.
        delivery_options: list
            The list of possible delivery options for the order.
        release_time: int
            The time when the order is released.
        delivery_point: DeliveryPoint | PersonalPoint
            The delivery point where the order is delivered.
    """

    def __init__(self, customer: Customer, volume: int, weight: int, delivery_deadline: int, delivery_options: list, id = None):
        self.id = id
        self.customer = customer
        self.volume = volume
        self.weight = weight
        self.delivery_deadline = delivery_deadline
        self.delivery_options = delivery_options
        self.release_time = None
        self.delivery_point = None

        if self.id is None:
            self.id = str(uuid.uuid4())

        if self.delivery_options is None:
            self.delivery_options = []

    def __str__(self):
        return f"Order(id={self.id}, customer={self.customer}, delivery_deadline={self.delivery_deadline}, delivery_options={self.delivery_options})"

    def __repr__(self):
        return self.__str__()

    def __eq__(self, other):
        return self.id == other.id

    def __hash__(self):
        return hash(self.id)

    def add_delivery_option(self, delivery_option: tuple):
        """
        Adds a delivery option to the order.

        Parameters:
            delivery_option: tuple (DeliveryPoint | PersonalPoint, int)
                The delivery option to be added.

        Raises:
            **ValueError**: If delivery_option is not ``(DeliveryPoint | PersonalPoint, int)`` or if it is already in the list.
        """
        # check if delivery option is (DeliveryPoint | PersonalPoint, int) and if it is not already in the list
        if len(delivery_option) == 2 and isinstance(delivery_option[0], DeliveryPoint | PersonalPoint) and isinstance(delivery_option[1], int) and delivery_option not in self.delivery_options:
            self.delivery_options.append(delivery_option)
        else:
            raise ValueError(
                f"Delivery option {delivery_option} is not valid! It should be a tuple of (DeliveryPoint | PersonalPoint, int) and cannot be already in the list")


class Location:
    """
    Location class represents a location of the problem.

    Parameters:
        x: int
            The x coordinate of the location.
        y: int
            The y coordinate of the location.

    Attributes:
        x: int
            The x coordinate of the location.
        y: int
            The y coordinate of the location.
    """

    def __init__(self, x: int | float, y: int | float):
        self.x = x
        self.y = y

    def __str__(self):
        return f"Location({self.x}, {self.y})"

    def __repr__(self):
        return self.__str__()

    def __eq__(self, other):
        return self.x == other.x and self.y == other.y

    def __hash__(self):
        return hash((self.x, self.y))

    def distance(self, other):
        """
        Calculates the distance between two locations.

        Parameters:
            other: Location
                The other location.

        Returns:
            **float**: The distance between the two locations.
        """
        return ((self.x - other.x)**2 + (self.y - other.y)**2)**0.5


class DeliveryPoint:
    """
    DeliveryPoint class represents a delivery point of the problem.

    Parameters:
        delivery_type: str
            The type of delivery point. It can be 'home', 'locker' or 'shop'.
        loc: Location
            The location of the delivery point.
        time_windows: list
            The list of time windows of the delivery point.
        capacity: int
            The capacity of the delivery point. It is only used when the delivery point is a 'locker' or a 'shop'.
        id: str
            The id of the delivery point. If no id is provided, a random uuid will be generated.

    Attributes:
        id : str
            The id of the delivery point.
        delivery_type: str
            The type of delivery point. It can be 'home', 'locker' or 'shop'.
        loc: Location
            The location of the delivery point.
        time_windows: list
            The list of time windows of the delivery point.
        capacity: int
            The capacity of the delivery point. It is only used when the delivery point is a 'locker' or a 'shop'.
    """

    def __init__(self, delivery_type: str, loc: Location, time_to_depot: float, time_windows: list = [], capacity: int = -1, id = None):
        self.id = id
        self.delivery_type = delivery_type
        self.loc = loc
        self.time_to_depot = time_to_depot
        self.capacity = capacity
        self.time_windows = time_windows

        if self.id is None:
            self.id = str(uuid.uuid4())

        # check if time_windows is a list of tuples (int, int, float)
        if not all(isinstance(time_window, tuple) and len(time_window) == 3
                   and isinstance(time_window[0], int) and isinstance(time_window[1], int) and isinstance(time_window[2], float)
                   for time_window in self.time_windows):
            raise ValueError(
                f"Time windows {self.time_windows} are not valid! They should be a list of tuples of (int, int, float)")

        # check if capacity is a int greater than 0.0 when delivery_type is not HOME
        if self.delivery_type != DeliveryType.HOME and self.capacity == -1:
            raise ValueError(
                f"Capacity of {self.delivery_type} must be specified! It should be a int greater than 0.0")
        elif self.delivery_type == DeliveryType.HOME:
            self.capacity = -1

    def __str__(self):
        return f"DeliveryPoint(id={self.id}, delivery_type={self.delivery_type}, loc={self.loc}, capacity={self.capacity}, time_windows={self.time_windows or None}, time_to_depot={self.time_to_depot})"

    def __repr__(self):
        return self.__str__()

    def __eq__(self, other):
        return self.id == other.id

    def __hash__(self):
        return hash(self.id)

    def add_time_window(self, time_window: tuple):
        """
        Adds a time window to the delivery point.

        Parameters:
            time_window: tuple (int, int)
                The time window to be added.

        Raises:
            **ValueError**: If time_window is not (int, int) or if it is already in the list.
        """
        # check if time_window is (int, int, float) and if it is not already in the list
        if len(time_window) == 3 and isinstance(time_window[0], int) and isinstance(time_window[1], int) and isinstance(time_window[2], float) and time_window not in self.time_windows:
            self.time_windows.append(time_window)
        else:
            raise ValueError(
                f"int window {time_window} is not valid! It should be a tuple of (int, int) and cannot be already in the list")


class PersonalPoint(DeliveryPoint):
    """
    PersonalPoint class represents a personal point of the problem. It is a special type of delivery point.

    Parameters:
        loc: Location
            The location of the personal point.
        is_tracked: bool
            Indicates if the personal point is tracked or not.
        time_windows: list
            The list of time windows of the personal point.
        pickers: list
            The list of pickers of the personal point.
        id: str
            The id of the personal point. If no id is provided, a random uuid will be generated.

    Attributes:
        id : str
            The id of the personal point.
        loc: Location
            The location of the personal point.
        is_tracked: bool
            Indicates if the personal point is tracked or not.
        time_windows: list
            The list of time windows of the personal point.
        pickers: list
            The list of pickers of the personal point.
        probabilities: list
            The list of probabilities of the personal point.
    """

    def __init__(self, loc: Location, time_to_depot: float, is_tracked: bool = False, time_windows: list = [], pickers: list = [], id = None):
        super().__init__(DeliveryType.HOME, loc, time_to_depot, time_windows, -1, id)
        self.id = id
        self.is_tracked = is_tracked # TODO: define how will this be used (maybe it should be a list)
        self.pickers = pickers
        self.probabilities = None # TODO: define how will this be used (data structure)

        if self.id is None:
            self.id = str(uuid.uuid4())

        # check if pickers is a list of uuids
        if not all(isinstance(picker, uuid) for picker in self.pickers):
            raise ValueError(
                f"Pickers {self.pickers} are not valid! They should be a list of uuids")

    def __str__(self):
        return f"PersonalPoint(id={self.id}, loc={self.loc}, is_tracked={self.is_tracked}, time_windows={self.time_windows or None}, time_to_depot={self.time_to_depot}, pickers={self.pickers or None})"

    def __repr__(self):
        return self.__str__()

    def __eq__(self, other):
        return self.id == other.id and self.loc == other.loc

    def __hash__(self):
        return hash((self.id, self.loc))

    def set_is_tracked(self, is_tracked: bool):
        """
        Sets the is_tracked attribute of the personal point.

        Parameters:
            is_tracked: bool
                The value to be set.
        """
        self.is_tracked = is_tracked

    def add_picker(self, picker: tuple):
        """
        Adds a picker to the personal point.

        Parameters:
            picker: uuid
                The picker to be added.

        Raises:
            **ValueError**: If picker is not a Customer or if it is already in the list.
        """
        # check if picker is Customer and if it is not already in the list
        if isinstance(picker, Customer) and picker not in self.pickers:
            self.pickers.append(picker)
        else:
            raise ValueError(
                f"Picker {picker} is not valid! It should be a Customer and cannot be already in the list")


class Truck:
    """
    Truck class represents a truck of the problem.

    Parameters:
        capacity: int
            The capacity of the truck.
        cost: int
            The cost of the truck.
        id: str
            The id of the truck. If no id is provided, a random uuid will be generated.

    Attributes:
        id : str
            The id of the truck.
        capacity: int
            The capacity of the truck.
        cost: int
            The cost of the truck.
    """

    def __init__(self, cost: int, capacity: int = 20):
        self.id = str(uuid.uuid4())
        self.capacity = capacity
        self.cost = cost

    def __str__(self):
        return f"Truck(id={self.id}, capacity={self.capacity}, cost={self.cost})"

    def __repr__(self):
        return self.__str__()

    def __eq__(self, other):
        return self.id == other.id

    def __hash__(self):
        return hash(self.id)


class Depot:
    """
    Depot class represents the depot of the problem.

    Parameters:
        loc: Location
            The location of the depot.
        fleet: list
            The list of trucks of the depot.
        time_window: tuple
            The time window of the depot.
        id: str
            The id of the depot. If no id is provided, a random uuid will be generated.

    Attributes:
        id : str
            The id of the depot.
        loc: Location
            The location of the depot.
        fleet: list
            The list of trucks of the depot.
        time_window: tuple
            The time window of the depot.
    """

    def __init__(self, loc: Location, fleet: list = [], time_window: tuple = None):
        self.id = str(uuid.uuid4())
        self.loc = loc
        self.fleet = fleet
        self.time_window = time_window

        # check if fleet is a list of Trucks
        if not all(isinstance(truck, Truck) for truck in self.fleet):
            raise ValueError(
                f"Fleet {self.fleet} is not valid! It should be a list of Trucks")

    def __str__(self):
        return f"Depot(id={self.id}, loc={self.loc}, time_window={self.time_window or None})"

    def __repr__(self):
        return self.__str__()

    def __eq__(self, other):
        return self.id == other.id

    def __hash__(self):
        return hash(self.id)

    def set_fleet(self, fleet: list):
        self.fleet = fleet

    def add_truck(self, truck: Truck):
        """
        Adds a truck to the depot.

        Parameters:
            truck: Truck
                The truck to be added.

        Raises:
            **ValueError**: If truck is not a Truck or if it is already in the list.
        """
        # check if truck is not already in the fleet
        if truck not in self.fleet:
            self.fleet.append(truck)
        else:
            raise ValueError(
                f"Truck {truck} is not valid! It cannot be already in the fleet")


class Problem:
    """
    Problem class represents the problem.

    Parameters:
        depot: Depot
            The depot of the problem.
        customers: list
            The list of customers of the problem.
        orders: list
            The list of orders of the problem.
        delivery_points: list
            The list of delivery points of the problem (including personal points).
        truck_capacity: int
            The capacity of the truck.
        truck_cost: float
            The cost of the truck.
        km_cost: float
            The cost per km.

    Attributes:
        id : str
            The id of the problem.
        depot: Depot
            The depot of the problem.
        customers: list
            The list of customers of the problem.
        orders: list
            The list of orders of the problem.
        delivery_points: list
            The list of delivery points of the problem.
        truck_capacity: int
            The capacity of the truck.
        truck_cost: float
            The cost of the truck.
        km_cost: float
            The cost per km.
        truck_speed: float
            The speed of the truck.
    """

    def __init__(self, depot: Depot, truck_speed: int, truck_cost: float, km_cost: float, customers: list = [], delivery_points: list = [], orders: list = []):
        self.id = str(uuid.uuid4())
        self.depot = depot
        self.customers = customers
        self.delivery_points = delivery_points
        self.orders = orders
        self.truck_capacity = 100
        self.truck_cost = truck_cost
        self.km_cost = km_cost
        self.truck_speed = truck_speed

        # check if customers is a list of Customers
        if not all(isinstance(customer, Customer) for customer in self.customers):
            raise ValueError(
                f"Customers {self.customers} is not valid! It should be a list of Customers")

        # check if orders is a list of Orders
        if not all(isinstance(order, Order) for order in self.orders):
            raise ValueError(
                f"Orders {self.orders} is not valid! It should be a list of Orders")

        # check if delivery_points is a list of DeliveryPoints
        if not all(isinstance(delivery_point, DeliveryPoint) for delivery_point in self.delivery_points):
            raise ValueError(
                f"Delivery points {self.delivery_points} is not valid! It should be a list of DeliveryPoints")

        # check if depot is Depot
        if not isinstance(depot, Depot):
            raise ValueError(
                f"Depot {self.depot} is not valid! It should be a Depot")

    def __str__(self):
        # return f"Problem(id={self.id}, depot={self.depot}, customers={self.customers}, orders={self.orders}, delivery_points={self.delivery_points})"
        delivery_points = ""
        for dp in self.delivery_points:
            delivery_points += f"\t{dp},\n"
        customers = ""
        for cust in self.customers:
            customers += f"\t{cust},\n"
        orders = ""
        for order in self.orders:
            orders += f"\t{order},\n"
        return f"Problem {self.id}:\n depot={self.depot}\n truck_cost={self.truck_cost}\n km_cost={self.km_cost}\n customers=[\n{customers}]\n delivery_points=[\n{delivery_points} orders=[\n{orders}]\n ]"

    def __repr__(self):
        return self.__str__()

    def __eq__(self, other):
        return self.id == other.id

    def __hash__(self):
        return hash(self.id)

    def find_dp_by_id(self, id: str):
        """
        Finds a delivery point by its id.

        Parameters:
            id: str
                The id of the delivery point to be found.

        Returns:
            **DeliveryPoint | PersonalPoint | False**: The delivery point if it is found, False otherwise.
        """
        for dp in self.delivery_points:
            if dp.id == id:
                return dp
        return False

    def find_order_by_id(self, id: str):
        """
        Finds an order by its id.

        Parameters:
            id: str
                The id of the order to be found.

        Returns:
            **Order | False**: The order if it is found, False otherwise.
        """
        for o in self.orders:
            if o.id == id:
                return o
        return False

    def set_depot(self, depot: Depot):
        """
        Sets the depot of the problem.

        Parameters:
            depot: Depot
                The depot to be set.

        Raises:
            **ValueError**: If depot is not a Depot.
        """
        self.depot = depot

    def add_customer(self, customer: Customer):
        """
        Adds a customer to the problem.

        Parameters:
            customer: Customer
                The customer to be added.

        Raises:
            **ValueError**: If customer is not a Customer or if it is already in the list.
        """
        # check if customer is Customer and if it is not already in the list
        if isinstance(customer, Customer) and customer not in self.customers:
            self.customers.append(customer)
        else:
            raise ValueError(
                f"Customer {customer} is not valid! It should be a Customer and cannot be already in the list")

    def add_order(self, order: Order):
        """
        Adds an order to the problem.

        Parameters:
            order: Order
                The order to be added.

        Raises:
            **ValueError**: If order is not an Order or if it is already in the list.
        """
        # check if order is Order and if it is not already in the list
        if isinstance(order, Order) and order not in self.orders:
            self.orders.append(order)
        else:
            raise ValueError(
                f"Order {order} is not valid! It should be an Order and cannot be already in the list")

    def add_delivery_point(self, delivery_point: DeliveryPoint):
        """
        Adds a delivery point to the problem.

        Parameters:
            delivery_point: DeliveryPoint
                The delivery point to be added.

        Raises:
            **ValueError**: If delivery_point is not a DeliveryPoint or if it is already in the list.
        """
        # check if delivery_point is DeliveryPoint and if it is not already in the list
        if isinstance(delivery_point, DeliveryPoint) and delivery_point not in self.delivery_points:
            self.delivery_points.append(delivery_point)
        else:
            raise ValueError(
                f"Delivery point {delivery_point} is not valid! It should be a DeliveryPoint and cannot be already in the list")

    def create_dictionaries(self):
        """
        Create dictionaries with the information of the problem:

        Parameters:
            -

        Raises:
            -
        """
        self.dict_xy = {}
        self.dict_twe = {}
        self.dict_twl = {}
        self.dict_twp = {}
        self.dict_distance = {}
        self.dict_tw_distance = {}
        self.dict_capacity = {}
        self.dict_priority = {}
        self.dict_delivery_time = {
            'DEFAULT': 10,
            'HOME': 5,
            'LOCKER': 2,
            'SHOP': 2
        }
        
        for order in self.orders:
            for option in order.delivery_options:
                dp, priority = option[0], option[1]
                self.dict_xy[dp.id] = dp.loc.x, dp.loc.y
                self.dict_priority[(order.id, dp.id)] = priority
                tw = dp.time_windows
                if len(tw) == 1:
                    twe, twl, twp = tw[0]
                elif len(tw) == 0:
                    twe = -999999999
                    twl = 999999999
                    twp = 1.0
                else:
                    twe, twl, twp = [], [], []
                    for i in range(len(tw)):
                        twe.append(tw[i][0])
                        twl.append(tw[i][1])
                        twp.append(tw[i][2])

                self.dict_twe[dp.id] = twe
                self.dict_twl[dp.id] = twl
                self.dict_twp[dp.id] = twp

        for delivery_point in self.delivery_points:
            self.dict_capacity[delivery_point.id] = delivery_point.capacity

        self.dict_xy['DEPOT'] = (self.depot.loc.x, self.depot.loc.y)
        self.dict_twe['DEPOT'] = self.depot.time_window[0]
        self.dict_twl['DEPOT'] = self.depot.time_window[1]
        self.dict_capacity['DEPOT'] = -1

        for point1, xy1 in self.dict_xy.items():
            for point2, xy2 in self.dict_xy.items():
                self.dict_distance[point1, point2] = dist(xy1, xy2)
                if point1 != point2:
                    self.dict_tw_distance[point1, point2] = tw_dist(self.dict_twe[point1], self.dict_twl[point1], self.dict_twe[point2], self.dict_twl[point2])
                else:
                    self.dict_tw_distance[point1, point2] = 0
                    
    def create_list_of_options(self):
        """
        Create a list with every tuple (order,option)

        Parameters:
            -

        Returns:
            ** list_of_options **
        """
        self.list_of_options = []
        for order in self.orders:
            for option in order.delivery_options:
                dp = option[0]
                self.list_of_options.append((order.id, dp.id))
        return self.list_of_options

    def create_list_of_clustered_options(self, dict_clusters):
        """
        Create a list with every tuple (order,option)

        Parameters:
            -

        Returns:
            ** list_of_options **
        """
        self.list_of_clustered_options = []
        list_of_options = []
        for order in self.orders:
            for option in order.delivery_options:
                dp = option[0]
                list_of_options.append((order.id, dp.id))

        dopts_by_cluster = {cluster: [] for cluster in dict_clusters}
        for order, loc in list_of_options:
            for cluster, locs in dict_clusters.items():
                if loc in locs:
                    dopts_by_cluster[cluster].append((order, loc))

        for cluster, dopts in dopts_by_cluster.items():
            self.list_of_clustered_options += dopts

        return self.list_of_clustered_options


class Stop:
    """
    Stop class represents a stop of a route.

    Parameters:
        dp: DeliveryPoint | PersonalPoint
            The delivery point of the stop.
        orders: list
            The list of orders to be delivered in the stop.
        exec_time: int
            The execution time of the stop.

    Attributes:
        id : str
            The id of the stop.
        dp: DeliveryPoint | PersonalPoint
            The delivery point of the stop.
        orders: list
            The list of orders to be delivered in the stop.
        exec_time: int
            The execution time of the stop.
    """

    def __init__(self, dp: DeliveryPoint | PersonalPoint, orders: list = [], exec_time: int = None):
        self.id = str(uuid.uuid4())
        self.dp = dp
        self.orders = orders
        self.exec_time = exec_time

    def __str__(self):
        orders_str = "["
        for order in self.orders:
            orders_str += f"{order.id}, "
        orders_str = orders_str[:-2] + "]"

        return f"Stop({self.dp.id}, {orders_str})"

    def __repr__(self):
        return self.__str__()

    def __eq__(self, other):
        return self.id == other.id

    def __hash__(self):
        return hash(self.id)

    def add_order(self, order: Order):
        """
        Adds an order to the stop.

        Parameters:
            order: Order
                The order to be added.

        Raises:
            **ValueError**: If order is not an Order or if it is already in the list.
        """
        # check if order is Order and if it is not already in the list
        if isinstance(order, Order) and order not in self.orders:
            self.orders.append(order)
        else:
            raise ValueError(
                f"Order {order} is not valid! It should be an Order and cannot be already in the list")

    def set_exec_time(self, exec_time: int):
        """
        Sets the execution time of the stop.

        Parameters:
            exec_time: int
                The execution time to be set.
        """
        self.exec_time = exec_time


class Route:
    """
    Route class represents a route of the problem.

    Parameters:
        truck: Truck
            The truck of the route.
        stops: list
            The list of stops of the route.

    Attributes:
        id : str
            The id of the route. If no id is provided, a random uuid will be generated.
        truck: Truck
            The truck of the route.
        stops: list
            The list of stops of the route.
        distance: float
            The total distance of the route.
        cost: float
            The total cost of the route.
        sum_priority: int
            The sum of the priorities of the orders of the route.
        start_time: int
            The start time of the route.
        is_feasible: bool
            Indicates if the route is feasible or not.
    """

    def __init__(self, truck: Truck, stops: list = []):
        self.id = str(uuid.uuid4())
        self.truck = truck
        self.stops = stops
        self.distance = 0.0
        self.cost = 0.0
        self.sum_priority = 0
        self.start_time = None
        self.is_feasible_att = True

    def __str__(self):
        return f"Route(id={self.id}, stops={self.stops})"

    def __repr__(self):
        return self.__str__()

    def __eq__(self, other):
        return self.id == other.id

    def __hash__(self):
        return hash(self.id)

    def add_stop(self, stop: Stop, problem: Problem):
        """
        Adds a stop to the route.

        Parameters:
            stop: Stop
                The stop to be added.

        Raises:
            **ValueError**: If stop is not a Stop or is not feasible or if it is already in the list.

        """
        distance = 0.0
        if isinstance(stop, Stop) and stop not in self.stops and stop.dp.capacity:
            if len(self.stops) == 0:    # first stop
                distance = problem.depot.loc.distance(stop.dp.loc)
                spent_time = stop.dp.time_to_depot
                start = problem.depot.time_window[0]

                if stop.dp.delivery_type == DeliveryType.HOME or stop.dp.delivery_type == DeliveryType.SHOP:
                    # print(start + spent_time, stop.dp.time_windows, start + 2*spent_time, [problem.depot.time_window])
                    if check_time_windows(start + spent_time, stop.dp.time_windows) and check_time_windows(start + 2*spent_time, [problem.depot.time_window]):
                        self.start_time = start
                    else:
                        for tw in stop.dp.time_windows:
                            start = tw[0] - spent_time
                            self.start_time = start
                            if check_time_windows(start, [problem.depot.time_window]) and check_time_windows(start + 2*spent_time, [problem.depot.time_window]):
                                self.is_feasible_att = True
                                break
                            else: self.is_feasible_att = False

                elif stop.dp.delivery_type == DeliveryType.LOCKER:
                    self.start_time = start

                stop.exec_time = self.start_time + spent_time

            else:
                distance = self.stops[-1].dp.loc.distance(stop.dp.loc)
                travel_time = (distance / problem.truck_speed) * 60
                spent_time = self.stops[-1].exec_time + travel_time

                if stop.dp.delivery_type == DeliveryType.HOME or stop.dp.delivery_type == DeliveryType.SHOP:
                    if check_time_windows(spent_time, stop.dp.time_windows) and check_time_windows(spent_time + stop.dp.time_to_depot, [problem.depot.time_window]):
                        stop.exec_time = spent_time
                    else:
                        for tw in stop.dp.time_windows:
                            if spent_time <= tw[0] and check_time_windows(tw[0] + stop.dp.time_to_depot, [problem.depot.time_window]):
                                stop.exec_time = tw[0]
                                self.is_feasible_att = True
                                break
                            else: self.is_feasible_att = False

                elif stop.dp.delivery_type == DeliveryType.LOCKER:
                    if check_time_windows(spent_time + stop.dp.time_to_depot, [problem.depot.time_window]):
                        stop.exec_time = spent_time
                    else:
                        self.is_feasible_att = False

            if not self.is_feasible_att:
                # print(f"{stop} cannot be added to the route because the time windows do not match!")
                # print(f"Route: {self}")
                # print(f"Stop time windows: {stop.dp.time_windows}")
                # print(f"Depot time windows: {problem.depot.time_window}")
                # print('\n\n')
                raise ValueError(f"{stop} cannot be added to the route! It is not feasible")

            self.stops.append(stop)
            self.distance += distance
            self.cost += distance * problem.km_cost

        else:
            raise ValueError(f"{stop} is not valid! It should be a Stop and cannot be already in the list. Also, the delivery point should have enough capacity.")

    def set_distance(self, distance: int):
        """
        Sets the distance of the route.

        Parameters:
            distance: int
                The distance to be set.
        """
        self.distance = distance

    def set_cost(self, cost: int):
        """
        Sets the cost of the route.

        Parameters:
            cost: int
                The cost to be set.
        """
        self.cost = cost

    def set_start_time(self, start_time: int):
        """
        Sets the start time of the route.

        Parameters:
            start_time: int
                The start time to be set.
        """
        self.start_time = start_time

    def is_feasible(self, problem: Problem):
        """
        Checks if the route is feasible.

        Parameters:
            problem: Problem
                The problem to be solved.

        Returns:
            **bool**: True if the route is feasible, False otherwise.
        """
        vehicle_capacity = self.truck.capacity
        num_orders = sum([len(stop.orders) for stop in self.stops])

        if num_orders > vehicle_capacity:
            return False

        for i in range(len(self.stops)):
            spent_time = 0
            dp_tws = problem.reduced_tws[self.stops[i].dp.id]
            if i == 0:  # first stop
                spent_time += (problem.depot.loc.distance(self.stops[i].dp.loc) / problem.truck_speed) * 60
                if not check_time_windows(self.start_time + spent_time, dp_tws): return False
            elif i == len(self.stops) - 1:  # last stop
                spent_time += (self.stops[i-1].dp.loc.distance(self.stops[i].dp.loc) / problem.truck_speed) * 60
                if not check_time_windows(self.start_time + spent_time, dp_tws): return False
            else:  # intermediate stops
                spent_time += (self.stops[i-1].dp.loc.distance(self.stops[i].dp.loc) / problem.truck_speed) * 60
                if not check_time_windows(self.start_time + spent_time, dp_tws): return False

                spent_time += (self.stops[i].dp.loc.distance(problem.depot.loc) / problem.truck_speed) * 60
                if not check_time_windows(self.start_time + spent_time, problem.depot.time_window): return False

            if self.stops[i].dp.delivery_type == DeliveryType.SHOP and len(self.stops[i].orders) > self.stops[i].dp.capacity:
                return False

            if self.stops[i].dp.delivery_type == DeliveryType.LOCKER and len(self.stops[i].orders) > self.stops[i].dp.capacity:
                return False

            # if self.stops[i].dp.delivery_type == DeliveryType.HOME and self.stops[i].dp.is_tracked:
            #     return check_probabilities(self.stops[i].dp.probabilities)

        return True

    def clear(self):
        self.id = str(uuid.uuid4())
        self.stops = []
        self.distance = 0.0
        self.cost = 0.0
        self.sum_priority = 0
        self.start_time = None
        self.is_feasible = True


class Solution:
    """
    Solution class represents a solution of the problem.

    Parameters:
        routes: list
            The list of routes of the solution.

    Attributes:
        id : str
            The id of the solution. It is generated randomly.
        routes: list
            The list of routes of the solution.
        successProbability: float
            The success probability of the solution.
        customerSatisfaction: float
            The customer satisfaction of the solution.
    """
    def __init__(self, routes: list = []):
        self.id = str(uuid.uuid4())
        self.routes = routes
        self.successProbability = None
        self.customerSatisfaction = None

    def __str__(self):
        routes_str = "[\n"
        for route in self.routes:
            r_str = "\nRoute " + str(self.routes.index(route) + 1) + ": DEPOT -> "
            for stop in route.stops:
                r_str += stop.dp.id + " ["
                for order in stop.orders:
                    r_str += order.id + ", "
                r_str = r_str[:-2] + "] -> "
            r_str = r_str[:-4]
            routes_str += r_str + " -> DEPOT,\n"
        routes_str = routes_str[:-2] + "\n\n]"
        return f"Solution(id={self.id}, routes={routes_str})"

    def __repr__(self):
        return self.__str__()

    def __eq__(self, other):
        return self.id == other.id

    def __hash__(self):
        return hash(self.id)

    def add_route(self, route: Route):
        """
        Adds a route to the solution.

        Parameters:
            route: Route
                The route to be added.

        Raises:
            **ValueError**: If route is not a Route or if it is already in the list.
        """
        # check if route is Route and if it is not already in the list
        if isinstance(route, Route) and route not in self.routes:
            self.routes.append(route)
        else:
            raise ValueError(
                f"Route {route} is not valid! It should be a Route and cannot be already in the list")

    def set_routes(self, routes: list):
        """
        Sets the routes of the solution.

        Parameters:
            routes: list
                The routes to be set.

        Raises:
            **ValueError**: If routes is not a list of Routes.
        """
        # check if routes is a list of Routes
        if all(isinstance(route, Route) for route in routes):
            self.routes = routes
        else:
            raise ValueError(
                f"Routes {routes} is not valid! It should be a list of Routes")

    def is_feasible(self, problem: Problem):
        """
        Checks if the solution is feasible.

        Parameters:
            problem: Problem
                The problem to be solved.

        Returns:
            **bool**: True if the solution is feasible, False otherwise.
        """
        if len(self.routes) > len(problem.depot.fleet):
        # if len(self.routes) != len(problem.depot.fleet):
            return False

        for route in self.routes:
            if not route.is_feasible(problem):
                return False
        return True


class DeliveryType(Enum):
    """
    DeliveryType enum represents the type of delivery point.
    """
    HOME = 'home'
    LOCKER = 'locker'
    SHOP = 'shop'

    def __str__(self):
        return self.value

    def __repr__(self):
        return self.__str__()


def check_time_windows(time: int, time_windows: list):
    """
    Checks if a time is in a list of time windows.

    Parameters:
        time: int
            The time to be checked.
        time_windows: list
            The list of time windows.

    Returns:
        **bool**: True if the time is in the list of time windows, False otherwise.
    """
    for time_window in time_windows:
        if time_window[0] <= time <= time_window[1]:
            return True
    return False


def tw_dist(twe1: int | list, twl1: int | list, twe2: int | list, twl2: int | list):
    """
    Calculates the time distance between two delivery points.

    Parameters:
        twe1: int | list
            The earliest time(s) of the first delivery point.
        twl1: int | list
            The latest time(s) of the first delivery point.
        twe2: int | list
            The earliest time(s) of the second delivery point.
        twl2: int | list
            The latest time(s) of the second delivery point.

    Returns:
        **int**: The minimum time distance between the two delivery points.
    """

    # Each parameter can be a single int or a list of ints
    # If it is a single int, convert it to a list
    if isinstance(twe1, int):
        twe1 = [twe1]
    if isinstance(twl1, int):
        twl1 = [twl1]
    if isinstance(twe2, int):
        twe2 = [twe2]
    if isinstance(twl2, int):
        twl2 = [twl2]

    # Calculate the minimum distance between the two delivery points
    distances = []
    for tw1 in zip(twe1, twl1):
        for tw2 in zip(twe2, twl2):
            if tw1[0] > tw2[1]: # tw1 starts after tw2 ends
                distances.append(tw1[0] - tw2[1])
            elif tw2[0] > tw1[1]: # tw2 starts after tw1 ends
                distances.append(tw2[0] - tw1[1])
            else:
                distances.append(0)

    # Return the minimum distance
    return np.min(distances)