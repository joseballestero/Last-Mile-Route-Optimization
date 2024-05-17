import random

#Función para Swap
def neighbourhoodSWAP(current_solution):
    neighbourhood = []
    for i in range(len(current_solution)):
        for j in range(i + 1, len(current_solution)):
            if i != j:
                neighbour = current_solution[:]  # Copia de la solución original
                neighbour[i], neighbour[j] = neighbour[j], neighbour[i]  # Intercambio de elementos
                neighbourhood.append(neighbour)
    return neighbourhood




#Función para doble Swap, cada iteración realiza dos intercambios en un grupo de 4 elementos consecutivos
def neighbourhoodDoubleSWAP(current_solution):
    neighbourhood = []
    n = len(current_solution)

    i = 0
    while i < n - 3:
        neighbour = current_solution[:]  # Copia de la solución original

        # Intercambio de los primeros dos elementos
        neighbour[i], neighbour[i + 1] = neighbour[i + 1], neighbour[i]

        # Intercambio de los siguientes dos elementos
        if i + 2 < n and i + 3 < n:
            neighbour[i + 2], neighbour[i + 3] = neighbour[i + 3], neighbour[i + 2]

        neighbourhood.append(neighbour)
        i += 4  # Avanzar al siguiente grupo de 4 elementos

    return neighbourhood


#Función para Insert
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


#Función para hacer un Swap alternativo
def neighbourhoodAlternativeSWAP(current_solution):
    neighbourhood = []
    # Itera sobre todos los índices de la solución actual
    for i in range(len(current_solution)):
        # Itera sobre los índices restantes después de i en la solución actual
        for j in range(i + 1, len(current_solution)):
            # Itera sobre los índices restantes después de j en la solución actual
            for k in range(j + 1, len(current_solution)):
                # Verifica que i, j, k sean diferentes
                if i != j and j !=k:
                    neighbour = current_solution[:]  # Copia de la solución original
                    # Intercambio de elementos
                    neighbour[i], neighbour[j], neighbour[k] = neighbour[j], neighbour[k], neighbour[i]  
                    neighbourhood.append(neighbour)
    return neighbourhood



#Función para hacer Swap o Insert
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


#Función para hacer Swap e Insert en cada iteración    
def neighbourhoodSWAP2(current_solution):
    

        i, j = random.sample(range(len(current_solution)), 2)
        neighbour = current_solution[:]
        neighbour[i], neighbour[j] = neighbour[j], neighbour[i]

        random_index = random.randint(0, len(current_solution))  # Posición aleatoria para insertar
        random_element = random.choice(neighbour)  # Elemento aleatorio a insertar
        neighbour.insert(random_index, random_element)  # Inserción del elemento
        
        return neighbour    