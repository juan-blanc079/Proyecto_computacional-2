import numpy as np
import matplotlib.pyplot as plt
from numpy import pi

# Constantes
C = 26.24
D = 1.44
tiempo = np.linspace(0, 1000, 10000)
dx = 0.001

def potential_energy(x):
    # Se evita la singularidad en x=0; aunque se contempla que nunca se evalúa en exactamente 0
    return -D/x if x != 0 else 0

def wave_function(particle):
    history = []
    for i in range(len(tiempo)):
        x = particle["x"]
        yi_0 = particle["y_0"]
        yi_1 = particle["y_1"]

        # Se guarda el par (x, ψ)
        history.append([x, yi_0])
        
        E = particle["energy"]
        U = potential_energy(x)
        
        # Ecuación de Schrödinger reescrita en forma diferencial de 2º orden:
        y_2 = -C * (E - U) * yi_0
        y_1 = yi_1 + y_2 * dx
        y_0 = yi_0 + y_1 * dx
        
        particle["y_0"] = y_0
        particle["y_1"] = y_1 
        particle["x"] = x + dx
        
    particle["history"] = history
    
    return particle

def graphs(initial_conditions, energy_levels):
    # Para evitar problemas con la singularidad en x = 0, iniciamos la integración en x = dx/2
    x0 = dx/2  
    psi0 = initial_conditions[1] * x0  # Aproximación: ψ(x0) ≈ ψ'(0)*x0
    psi1 = initial_conditions[1]
    x_lims = [1.25, 1.8, 2.5, 3.5]
    
    for i, energy in enumerate(energy_levels):
        plt.figure(figsize=(10, 8))
        plt.title(f"Función de Onda para E = {energy} eV")
        plt.xlabel("x (m)")
        plt.ylabel("ψ(x)")
        plt.ylim(-10, 10)
        plt.xlim(-0.01, 10)
        plt.grid(True)
        
        # Definir la partícula con las condiciones iniciales modificadas
        particle = {
            "id": i,
            "x": x0,
            "energy": energy,
            "y_0": psi0,
            "y_1": psi1
        }
        
        particle = wave_function(particle)
        history = particle["history"]
        xs = [p[0] for p in history]
        ys = [p[1] for p in history]
        plt.plot(xs, ys, lw=1, label=f"E = {energy} eV")
        plt.legend()
        plt.ylim(-0.05, 0.05)
        plt.xlim(-0.05, x_lims[i])
        plt.show()

# Iniciación
initial_conditions = np.array([0, 0.5])  # [ψ(0), ψ'(0)]
energy_levels = np.array([-13.606461780325, -3.401236, -1.511589, -0.85025])

graphs(initial_conditions, energy_levels)
