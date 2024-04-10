import random


def neighbourhoodSWAP(current_solution):
    neighbourhood = []
    for i in range(len(current_solution)):
        for j in range(i + 1, len(current_solution)):
            if i != j:
                neighbour = current_solution[:]  # Copia de la solución original
                neighbour[i], neighbour[j] = neighbour[j], neighbour[i]  # Intercambio de elementos
                neighbourhood.append(neighbour)
    return neighbourhood



def neighbourhoodINSERT(current_solution):
    neighbourhood = []
    for i in range(len(current_solution)):
        for j in range(len(current_solution)):
            if i != j:
                neighbour = current_solution[:]  # Copia de la solución original
                element = neighbour.pop(i)  # Sacamos el elemento en la posición i
                neighbour.insert(j, element)  # Insertamos el elemento en la posición j
                neighbourhood.append(neighbour)
    return neighbourhood




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