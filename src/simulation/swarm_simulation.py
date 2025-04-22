import random
import numpy as np
import time

from src.models.drone import Drone
from src.models.enums import DroneRole, DroneStatus

class DecentralizedSwarmSimulation:
    """Simulation de l'essaim de drones décentralisé"""
    def __init__(self, num_drones=10, waypoints=None):
        # Passer les waypoints à l'initialisation
        self.waypoints = waypoints or [
            [100, 100, 60],   # Point 1
            [200, 50, 70],    # Point 2
            [50, 200, 55],    # Point 3
            [150, 150, 65]    # Point 4
        ]
        self.num_drones = num_drones
        self.drones = []
        self.start_time = time.time()
        self.network_graph = {}  # Pour suivre les connexions entre drones
        
        # Initialisation
        self.initialize_swarm()
    
    def initialize_swarm(self):
        """Initialise l'essaim de drones avec des positions de départ"""
        self.drones = []
        
        for i in range(self.num_drones):
            position = [
                random.uniform(0, 100),
                random.uniform(0, 100),
                random.uniform(50, 70)
            ]
            
            velocity = [
                random.uniform(-0.1, 0.1),
                random.uniform(-0.1, 0.1),
                random.uniform(-0.05, 0.05)
            ]
            
            # Créer le drone avec les waypoints
            drone = Drone(
                drone_id=i, 
                position=position, 
                velocity=velocity,
                waypoints=self.waypoints
            )
            self.drones.append(drone)
        
        # Désigner le premier drone comme leader initial
        if self.drones:
            self.drones[0].role = DroneRole.LEADER
            self.drones[0].leader_id = 0
            self.drones[0].color = 'green'
            
            # Initialiser la formation depuis le leader
            self.drones[0].broadcast_formation_update()
            print(f"Drone 0 désigné comme leader initial, waypoints: {self.waypoints}")
    
    def reset_simulation(self, event=None):
        """Réinitialise la simulation"""
        self.drones = []
        self.start_time = time.time()
        self.initialize_swarm()
        print("Simulation réinitialisée avec de nouveaux drones.")
    
    def add_drone(self, event=None):
        """Ajoute un nouveau drone à l'essaim"""
        # Trouver l'ID le plus élevé existant
        max_id = max([drone.id for drone in self.drones]) if self.drones else -1
        new_id = max_id + 1
        
        # Utiliser la position moyenne des drones actifs comme point de départ
        active_drones = [d for d in self.drones if d.status == DroneStatus.ACTIVE]
        if active_drones:
            avg_position = np.mean([d.position for d in active_drones], axis=0)
            position = avg_position + np.array([
                random.uniform(-20, 20),
                random.uniform(-20, 20),
                random.uniform(5, 15)
            ])
        else:
            # Position par défaut si aucun drone actif
            position = np.array([
                random.uniform(0, 100),
                random.uniform(0, 100),
                random.uniform(50, 70)
            ])
        
        velocity = [
            random.uniform(-0.1, 0.1),
            random.uniform(-0.1, 0.1),
            random.uniform(-0.05, 0.05)
        ]
        
        # Créer et ajouter le nouveau drone
        new_drone = Drone(
            drone_id=new_id, 
            position=position, 
            velocity=velocity,
            waypoints=self.waypoints
        )
        self.drones.append(new_drone)
        
        # Forcer le leader à recalculer la formation
        for drone in self.drones:
            if drone.role == DroneRole.LEADER and drone.status == DroneStatus.ACTIVE:
                formation_update = drone.broadcast_formation_update()
                if formation_update:
                    drone.relay_message(formation_update, self.drones)
                break
                
        print(f"Ajouté un nouveau drone avec l'ID {new_id}")
    
    def fail_leader(self, event=None):
        """Fait échouer le leader actuel pour tester l'élection"""
        for drone in self.drones:
            if drone.role == DroneRole.LEADER and drone.status == DroneStatus.ACTIVE:
                drone.status = DroneStatus.FAILED
                drone.color = 'black'
                print(f"Leader (Drone {drone.id}) mis en échec! Une nouvelle élection devrait commencer.")
                return
        
        print("Aucun leader actif trouvé à mettre en échec.")
    
    def update_network_graph(self):
        """Met à jour le graphe du réseau basé sur qui peut communiquer avec qui"""
        self.network_graph = {}
        
        # Initialiser le graphe
        for drone in self.drones:
            self.network_graph[drone.id] = []
        
        # Déterminer les connexions
        for i, drone1 in enumerate(self.drones):
            for drone2 in self.drones[i+1:]:
                if drone1.can_communicate_with(drone2) and drone2.can_communicate_with(drone1):
                    self.network_graph[drone1.id].append(drone2.id)
                    self.network_graph[drone2.id].append(drone1.id)
    
    def update(self):
        """Met à jour l'état global de la simulation"""
        # Mettre à jour chaque drone
        for drone in self.drones:
            drone.update(self.drones)
        
        # Mettre à jour le graphe du réseau
        self.update_network_graph()