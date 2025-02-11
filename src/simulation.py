import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import time
from src.swarm import Swarm
from src.config import X_LIMIT, Y_LIMIT, Z_LIMIT, SIMULATION_STEPS

def run_simulation():
    """Exécute la simulation de l'essaim"""
    swarm = Swarm()

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    for _ in range(SIMULATION_STEPS):
        ax.clear()
        swarm.update_swarm()

        # Récupération des positions
        active_positions, failed_positions = swarm.get_positions()

        # Affichage des drones actifs (bleu) et en panne (rouge)
        if len(active_positions) > 0:
            ax.scatter(active_positions[:, 0], active_positions[:, 1], active_positions[:, 2], color='blue', label="Actifs")

        if len(failed_positions) > 0:
            ax.scatter(failed_positions[:, 0], failed_positions[:, 1], failed_positions[:, 2], color='red', label="En panne")

        # Configuration de l'affichage
        ax.set_xlim(X_LIMIT)
        ax.set_ylim(Y_LIMIT)
        ax.set_zlim(Z_LIMIT)
        ax.set_title("Simulation d'un Essaim de Drones")
        ax.legend()

        plt.pause(0.05)  # Pause pour l'animation

    plt.show()
