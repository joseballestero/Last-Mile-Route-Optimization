import random


def neighbourhoodSWAP1(current_solution, condition):
    
    if condition:
    
        i, j = random.sample(range(len(current_solution)), 2)
        neighbour = current_solution[:]
        neighbour[i], neighbour[j] = neighbour[j], neighbour[i]
        return neighbour
    else:
        neighbour = current_solution[:]  # Copia de la solución original
        random_index = random.randint(0, len(current_solution))  # Posición aleatoria para insertar
        random_element = random.choice(current_solution)  # Elemento aleatorio a insertar
        neighbour.insert(random_index, random_element)  # Inserción del elemento
        return neighbour

    
def neighbourhoodSWAP2(current_solution):
    

        i, j = random.sample(range(len(current_solution)), 2)
        neighbour = current_solution[:]
        neighbour[i], neighbour[j] = neighbour[j], neighbour[i]

        random_index = random.randint(0, len(current_solution))  # Posición aleatoria para insertar
        random_element = random.choice(neighbour)  # Elemento aleatorio a insertar
        neighbour.insert(random_index, random_element)  # Inserción del elemento
        
        return neighbour    