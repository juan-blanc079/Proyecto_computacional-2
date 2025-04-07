import numpy as np
import matplotlib.pyplot as plt
from numpy import pi

#Constantes
C = 26.24
D = 1.44
tiempo = np.linspace(0, 0.001, 30000)
dt = tiempo[1] - tiempo[0]
Total_energy = -13.6

#Funciones

def potential_energy(x):
    return -D/x if x != 0 else 0

def wave_function(particle):

    history = []

    for i in range(len(tiempo)):
        
        yi_0 = particle["y_0"]
        values = np.array([i,yi_0])
        yi_1 = particle["y_1"]

        history.append(values)

        Potential_energy = potential_energy(i)

        y_2 = -C * (Total_energy - Potential_energy) * yi_0
        y_1 = yi_1+ y_2 * dt
        y_0 = yi_0 + y_1 * dt

        particle["y_0"] = y_0
        particle["y_1"] = y_1 
    
    particle["history"] = history

    return particle

#Gráficas

def graphs(initial_conditions):

    y_0 = initial_conditions[0]
    y_1 = initial_conditions[1]

    plt.figure(figsize=(10, 8))
    plt.title("Función de Onda")
    plt.xlabel("x (m)")
    plt.ylabel("f(x) (m)")

    for i in range(0,5):

        particle={"id": i,
                  "y_0": y_0,
                  "y_1": y_1}
        
        particle = wave_function(particle)

        history = particle["history"]
        xs = [p[0] for p in history]
        ys = [p[1] for p in history]
        plt.scatter(xs, ys, s=2, alpha=0.7, label=f"b={i:.1e} m")

#Iniciación

initial_conditions = np.array([0,0.5]) # [f(0), f´(0)= 0.5]

graphs(initial_conditions)