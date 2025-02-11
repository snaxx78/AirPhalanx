import numpy as np
from src.drone import Drone
from src.config import N_DRONES

class Swarm:
    def __init__(self):
        """Initialisation de l’essaim de drones"""
        self.drones = [Drone(np.random.randint(-10, 10), 
                             np.random.randint(-10, 10), 
                             np.random.randint(-10, 10), 
                             role="Leader" if i == 0 else "Follower") 
                       for i in range(N_DRONES)]
        self.iteration_count = 0  # Compteur d'itérations

    def update_swarm(self):
        """Met à jour tous les drones de l’essaim"""

        self.iteration_count += 1
        # Déclarer un drone en panne après 20 itérations 
        if self.iteration_count == 20:
            failing_drone = np.random.choice(self.drones)  # Choisir un drone au hasard
            failing_drone.set_failure()

        for drone in self.drones:
            drone.update_position()

    def get_positions(self):
        """Récupère les positions des drones actifs et en panne"""
        active_positions = [drone.position for drone in self.drones if drone.state == "Actif"]
        failed_positions = [drone.position for drone in self.drones if drone.state == "En panne"]
        return np.array(active_positions), np.array(failed_positions)
