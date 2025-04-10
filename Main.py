import numpy as np
import matplotlib.pyplot as plt
from numpy import pi

#Constantes
C = 26.24
D = 1.44
tiempo = np.linspace(0,1000,10000)
dx = 0.001
dx_2 = dx/2

#Funciones

"""def phi_function(y_0, y_1):
    value = y_0 + y_1 * dx
    return value

def phi_derivative(y_1,y_2):
    value = y_1 * y_2 * dx
    return value

def phi_derivative_2(E,U, y_0):
    value = -C * (E + U) * y_0
    return value

"""

def potential_energy(x):
    potential_energy = -D/x if x!=0 else 0
    return potential_energy

def wave_function(particle):

    history = []

    for i in range(len(tiempo)):

        x = particle["x"]
        yi_0 = particle["y_0"]
        yi_1 = particle["y_1"]
        values = np.array([x,yi_0])

        history.append(values)

        E = particle["energy"]
        U = potential_energy(x)

        y_2 = -C *(E - U)*yi_0
        y_1 = yi_1 + y_2 * dx
        y_0 = yi_0 + y_1 * dx

        particle["y_0"] = y_0
        particle["y_1"] = y_1 
        particle["x"] = x + dx
    
    particle["history"] = history
    print(history)
    
    return particle

#Gráficas

def graphs(initial_conditions, energy_levels):

    y_0 = initial_conditions[0]
    y_1 = initial_conditions[1]

    plt.figure(figsize=(10, 8))
    plt.title("Función de Onda")
    plt.xlabel("x (m)")
    plt.ylabel("f(x) (m)")
    plt.ylim(-10, 10)
    plt.xlim(-0.01, 10)

    for i in range(len(energy_levels)):

        energy = energy_levels[i]
        particle={"id": i,
                  "x": 0,
                  "energy": energy,
                  "y_0": y_0,
                  "y_1": y_1}
        
        particle = wave_function(particle)

        history = particle["history"]
        xs = [p[0] for p in history]
        ys = [p[1] for p in history]
        plt.scatter(xs, ys, s=2, alpha=0.7, label=f"E={energy_levels[i]} eV")

    plt.legend()
    plt.grid(True)
    plt.show()
    

#Iniciación

initial_conditions = np.array([0,0.5]) # [f(0), f´(0)]
energy_levels = np.array([-13.5525, -3.3945, -1.5095, -0.8494])

graphs(initial_conditions, energy_levels)
