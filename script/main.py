import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D
import random
import math
from enum import Enum

class DroneStatus(Enum):
    ACTIVE = 1
    FAILING = 2
    FAILED = 3

class Drone:
    def __init__(self, drone_id, position, velocity, is_leader=False):
        self.id = drone_id
        self.position = np.array(position, dtype=float)
        self.velocity = np.array(velocity, dtype=float)
        self.acceleration = np.zeros(3)
        self.is_leader = is_leader
        self.status = DroneStatus.ACTIVE
        self.communication_range = 50.0
        self.perception_range = 40.0
        self.neighbors = []
        self.last_known_positions = {}
        self.last_known_roles = {}
        self.leader_id = self.id if is_leader else None
        self.mission_target = None
        self.formation_position = None
        self.color = 'red' if is_leader else 'blue'
        self.failure_probability = 0.0005  # Probabilité de défaillance à chaque pas de temps
        self.failing_countdown = 20  # Temps avant défaillance complète
        
    def set_mission_target(self, target):
        self.mission_target = np.array(target)
        
    def distance_to(self, other_drone):
        return np.linalg.norm(self.position - other_drone.position)
    
    def can_communicate_with(self, other_drone):
        return self.distance_to(other_drone) <= self.communication_range
    
    def broadcast_status(self, swarm):
        """Diffuse son statut aux drones voisins"""
        self.neighbors = []
        for drone in swarm:
            if drone.id != self.id and self.can_communicate_with(drone) and drone.status != DroneStatus.FAILED:
                self.neighbors.append(drone.id)
                self.last_known_positions[drone.id] = drone.position.copy()
                self.last_known_roles[drone.id] = drone.is_leader
    
    def find_current_leader(self, swarm):
        """Identifie le leader actuel parmi les drones accessibles"""
        for drone in swarm:
            if drone.is_leader and drone.status != DroneStatus.FAILED and self.can_communicate_with(drone):
                self.leader_id = drone.id
                return drone.id
        return None
    
    def elect_new_leader(self, swarm):
        """Processus d'élection d'un nouveau leader basé sur l'ID (le plus petit ID devient leader)"""
        if self.status == DroneStatus.FAILED:
            return
            
        # Collecte les IDs des drones actifs dans le réseau de communication
        active_drones = [d.id for d in swarm if d.status != DroneStatus.FAILED and self.can_communicate_with(d)]
        active_drones.append(self.id)  # Inclure soi-même
        
        # Le drone avec l'ID le plus petit devient le nouveau leader
        new_leader_id = min(active_drones)
        
        # Si ce drone est élu leader
        if new_leader_id == self.id:
            self.is_leader = True
            self.color = 'red'
            print(f"Drone {self.id} devient le nouveau leader")
        else:
            self.is_leader = False
            self.color = 'blue'
            
        self.leader_id = new_leader_id
        
        # Propager l'information sur le nouveau leader
        for drone in swarm:
            if drone.id != self.id and self.can_communicate_with(drone) and drone.status != DroneStatus.FAILED:
                drone.leader_id = new_leader_id
                if drone.id == new_leader_id:
                    drone.is_leader = True
                    drone.color = 'red'
                else:
                    drone.is_leader = False
                    drone.color = 'blue'
    
    def check_leader_status(self, swarm):
        """Vérifie si le leader est toujours actif et lance une élection si nécessaire"""
        if self.leader_id is None:
            self.elect_new_leader(swarm)
            return
            
        leader_active = False
        for drone in swarm:
            if drone.id == self.leader_id and drone.status != DroneStatus.FAILED and self.can_communicate_with(drone):
                leader_active = True
                break
                
        if not leader_active:
            print(f"Leader {self.leader_id} n'est plus disponible. Élection d'un nouveau leader...")
            self.elect_new_leader(swarm)
    
    def get_current_leader(self, swarm):
        """Retourne le drone leader actuel ou None"""
        for drone in swarm:
            if drone.id == self.leader_id and drone.status != DroneStatus.FAILED:
                return drone
        return None
    
    def calculate_formation_position(self, swarm):
        """Calcule la position que le drone devrait occuper dans la formation"""
        if self.is_leader or self.status == DroneStatus.FAILED:
            return
            
        leader = self.get_current_leader(swarm)
        if leader is None:
            return
            
        # Récupérer les drones actifs dans l'ordre de leur ID
        active_followers = sorted([d.id for d in swarm 
                                if not d.is_leader and d.status != DroneStatus.FAILED])
        
        if self.id not in active_followers:
            return
            
        # Position dans la formation en fonction de l'index dans la liste triée
        idx = active_followers.index(self.id)
        num_followers = len(active_followers)
        
        # Calcul de la position dans une formation en V derrière le leader
        angle = math.pi / 4  # 45 degrés
        distance = 15.0
        
        # Alterner les positions gauche et droite dans la formation en V
        side = 1 if idx % 2 == 0 else -1
        row = (idx // 2) + 1
        
        offset_x = -row * distance * math.cos(angle)
        offset_y = side * row * distance * math.sin(angle)
        offset_z = -row * 5.0  # Légèrement plus bas que le leader
        
        # Position relative au leader
        relative_pos = np.array([offset_x, offset_y, offset_z])
        
        # Direction vers laquelle se dirige le leader
        if np.linalg.norm(leader.velocity) > 0:
            direction = leader.velocity / np.linalg.norm(leader.velocity)
        else:
            direction = np.array([1, 0, 0])  # Direction par défaut
            
        # Définir les axes de la formation en fonction de la direction du leader
        forward = direction
        
        # On s'assure que 'up' est toujours (0,0,1) pour maintenir une formation horizontale
        up = np.array([0, 0, 1])
        right = np.cross(forward, up)
        right = right / np.linalg.norm(right) if np.linalg.norm(right) > 0 else np.array([0, 1, 0])
        
        # Recalculer 'up' pour s'assurer qu'il est orthogonal
        up = np.cross(right, forward)
        up = up / np.linalg.norm(up) if np.linalg.norm(up) > 0 else np.array([0, 0, 1])
        
        # Matrice de transformation pour convertir les coordonnées relatives en coordonnées globales
        transform = np.array([forward, right, up]).T
        global_offset = transform @ np.array([offset_x, offset_y, offset_z])
        
        self.formation_position = leader.position + global_offset
    
    def calculate_steering_forces(self, swarm):
        """Calcule les forces de pilotage selon les règles de comportement en essaim"""
        if self.status == DroneStatus.FAILED:
            # Les drones défaillants tombent lentement
            self.acceleration = np.array([0, 0, -0.5])
            return np.array([0, 0, -0.5])
            
        if self.status == DroneStatus.FAILING:
            # Les drones en défaillance deviennent instables
            failing_acc = np.array([
                random.uniform(-0.5, 0.5),
                random.uniform(-0.5, 0.5),
                random.uniform(-0.5, 0.1)
            ])
            self.acceleration = failing_acc
            return failing_acc
        
        # Poids des différentes forces
        separation_weight = 1.5
        alignment_weight = 1.0
        cohesion_weight = 1.0
        formation_weight = 2.0
        mission_weight = 1.5
        obstacle_avoidance_weight = 2.0
        
        # Force résultante (initialement nulle)
        steering_force = np.zeros(3)
        
        # Liste des drones voisins
        neighbors = [d for d in swarm if d.id != self.id and self.can_communicate_with(d) and d.status != DroneStatus.FAILED]
        
        if not neighbors:
            # Si pas de voisins, essayer de retourner vers la dernière position connue du leader
            if self.leader_id in self.last_known_positions:
                return (self.last_known_positions[self.leader_id] - self.position) * 0.02
            return np.zeros(3)
        
        # 1. Force de séparation (éviter les collisions)
        separation_force = np.zeros(3)
        separation_count = 0
        
        for neighbor in neighbors:
            distance = self.distance_to(neighbor)
            if distance < 10.0:  # Distance minimale souhaitée
                repulsion = self.position - neighbor.position
                
                # Plus le drone est proche, plus la force de répulsion est grande
                if distance > 0:
                    repulsion = repulsion / distance
                    separation_force += repulsion
                    separation_count += 1
        
        if separation_count > 0:
            separation_force /= separation_count
            separation_force = self.normalize(separation_force)
            
        # 2. Force d'alignement (même direction que les voisins)
        alignment_force = np.zeros(3)
        alignment_count = 0
        
        for neighbor in neighbors:
            alignment_force += neighbor.velocity
            alignment_count += 1
        
        if alignment_count > 0:
            alignment_force /= alignment_count
            alignment_force = self.normalize(alignment_force)
        
        # 3. Force de cohésion (rester groupé)
        cohesion_force = np.zeros(3)
        cohesion_count = 0
        
        for neighbor in neighbors:
            cohesion_force += neighbor.position
            cohesion_count += 1
        
        if cohesion_count > 0:
            cohesion_force /= cohesion_count
            cohesion_force = self.normalize(cohesion_force - self.position)
        
        # 4. Force de formation (maintenir la position dans la formation)
        formation_force = np.zeros(3)
        
        if not self.is_leader and self.formation_position is not None:
            formation_force = self.formation_position - self.position
            formation_force = self.normalize(formation_force)
        
        # 5. Force de mission (atteindre l'objectif)
        mission_force = np.zeros(3)
        
        if self.is_leader and self.mission_target is not None:
            mission_force = self.mission_target - self.position
            mission_force = self.normalize(mission_force)
        
        # 6. Évitement d'obstacles (à implémenter si des obstacles sont ajoutés)
        obstacle_force = np.zeros(3)
        
        # Calculer la force résultante en appliquant les poids
        steering_force = (
            separation_force * separation_weight +
            alignment_force * alignment_weight +
            cohesion_force * cohesion_weight +
            formation_force * formation_weight +
            mission_force * mission_weight +
            obstacle_force * obstacle_avoidance_weight
        )
        
        return steering_force
    
    def normalize(self, v):
        """Normalise un vecteur"""
        norm = np.linalg.norm(v)
        if norm > 0:
            return v / norm
        return v
    
    def update_status(self):
        """Met à jour le statut du drone (simulation de défaillance)"""
        if self.status == DroneStatus.ACTIVE:
            if random.random() < self.failure_probability:
                self.status = DroneStatus.FAILING
                self.color = 'orange'
                print(f"Drone {self.id} commence à tomber en panne!")
                
        elif self.status == DroneStatus.FAILING:
            self.failing_countdown -= 1
            if self.failing_countdown <= 0:
                self.status = DroneStatus.FAILED
                self.color = 'black'
                print(f"Drone {self.id} est complètement en panne!")
    
    def update(self, swarm):
        """Met à jour la position, la vitesse et l'état du drone"""
        # Mise à jour du statut (simulation de défaillance)
        self.update_status()
        
        # Diffuser son statut aux voisins
        self.broadcast_status(swarm)
        
        # Vérifier l'état du leader et élire un nouveau si nécessaire
        self.check_leader_status(swarm)
        
        # Calculer la position dans la formation
        self.calculate_formation_position(swarm)
        
        # Calculer les forces de pilotage
        steering_force = self.calculate_steering_forces(swarm)
        
        # S'assurer que steering_force n'est pas None
        if steering_force is None:
            steering_force = np.zeros(3)
        
        # Appliquer l'accélération (avec une limite)
        self.acceleration = steering_force
        max_acceleration = 0.5
        acc_magnitude = np.linalg.norm(self.acceleration)
        if acc_magnitude > max_acceleration:
            self.acceleration = self.acceleration * (max_acceleration / acc_magnitude)
        
        # Mettre à jour la vitesse
        self.velocity += self.acceleration
        
        # Limiter la vitesse
        max_speed = 2.0 if self.status == DroneStatus.ACTIVE else 1.0
        speed = np.linalg.norm(self.velocity)
        if speed > max_speed:
            self.velocity = self.velocity * (max_speed / speed)
        
        # Mettre à jour la position
        self.position += self.velocity

class DroneSwarmSimulation:
    def __init__(self, num_drones=5):
        self.num_drones = num_drones
        self.drones = []
        self.mission_waypoints = [
            np.array([100, 100, 50]),
            np.array([150, 200, 70]),
            np.array([200, 150, 60]),
            np.array([250, 100, 80]),
            np.array([300, 200, 90])
        ]
        self.current_waypoint_index = 0
        self.fig = None
        self.ax = None
        
        # Initialiser l'essaim de drones
        self.initialize_swarm()
    
    def initialize_swarm(self):
        """Initialise l'essaim de drones avec un leader et des suiveurs"""
        # Créer le premier drone comme leader
        leader = Drone(
            drone_id=0,
            position=[0, 0, 50],  # Position initiale
            velocity=[0.5, 0.5, 0],  # Vitesse initiale
            is_leader=True
        )
        self.drones.append(leader)
        
        # Créer les drones suiveurs
        for i in range(1, self.num_drones):
            # Position aléatoire autour du leader
            position = [
                leader.position[0] + random.uniform(-10, 10),
                leader.position[1] + random.uniform(-10, 10),
                leader.position[2] + random.uniform(-5, 5)
            ]
            
            # Vitesse similaire au leader avec une petite variation
            velocity = [
                leader.velocity[0] + random.uniform(-0.1, 0.1),
                leader.velocity[1] + random.uniform(-0.1, 0.1),
                leader.velocity[2] + random.uniform(-0.1, 0.1)
            ]
            
            follower = Drone(
                drone_id=i,
                position=position,
                velocity=velocity,
                is_leader=False
            )
            self.drones.append(follower)
        
        # Définir la cible de mission pour le leader
        leader.set_mission_target(self.mission_waypoints[self.current_waypoint_index])
    
    def update_mission_target(self):
        """Met à jour la cible de mission si nécessaire"""
        leader = None
        for drone in self.drones:
            if drone.is_leader and drone.status != DroneStatus.FAILED:
                leader = drone
                break
        
        if leader is None:
            return
            
        current_target = self.mission_waypoints[self.current_waypoint_index]
        distance_to_target = np.linalg.norm(leader.position - current_target)
        
        # Si le leader est suffisamment proche de la cible, passer à la suivante
        if distance_to_target < 20.0:
            self.current_waypoint_index = (self.current_waypoint_index + 1) % len(self.mission_waypoints)
            new_target = self.mission_waypoints[self.current_waypoint_index]
            leader.set_mission_target(new_target)
            print(f"Nouveau waypoint: {self.current_waypoint_index}")
    
    def update(self):
        """Met à jour l'état de tous les drones"""
        # Mettre à jour la cible de mission
        self.update_mission_target()
        
        # Mettre à jour chaque drone
        for drone in self.drones:
            drone.update(self.drones)
    
    def setup_visualization(self):
        """Prépare la visualisation 3D"""
        self.fig = plt.figure(figsize=(10, 8))
        self.ax = self.fig.add_subplot(111, projection='3d')
        self.ax.set_xlim([-50, 350])
        self.ax.set_ylim([-50, 250])
        self.ax.set_zlim([0, 100])
        self.ax.set_xlabel('X')
        self.ax.set_ylabel('Y')
        self.ax.set_zlabel('Z')
        self.ax.set_title('Simulation d\'Essaim de Drones en 3D')
    
    def visualize(self):
        """Affiche l'état actuel de l'essaim"""
        self.ax.clear()
        self.ax.set_xlim([-50, 350])
        self.ax.set_ylim([-50, 250])
        self.ax.set_zlim([0, 100])
        self.ax.set_xlabel('X')
        self.ax.set_ylabel('Y')
        self.ax.set_zlabel('Z')
        self.ax.set_title('Simulation d\'Essaim de Drones en 3D')
        
        # Afficher les drones
        for drone in self.drones:
            self.ax.scatter(
                drone.position[0],
                drone.position[1],
                drone.position[2],
                color=drone.color,
                s=100,
                marker='o' if drone.status == DroneStatus.ACTIVE else 'x'
            )
            
            # Afficher une flèche indiquant la direction
            if drone.status != DroneStatus.FAILED:
                velocity_norm = np.linalg.norm(drone.velocity)
                if velocity_norm > 0:
                    direction = drone.velocity / velocity_norm * 5  # Longueur de la flèche
                    self.ax.quiver(
                        drone.position[0],
                        drone.position[1],
                        drone.position[2],
                        direction[0],
                        direction[1],
                        direction[2],
                        color=drone.color,
                        length=1.0,
                        normalize=True
                    )
        
        # Afficher les waypoints de mission
        for i, waypoint in enumerate(self.mission_waypoints):
            color = 'green' if i == self.current_waypoint_index else 'gray'
            self.ax.scatter(
                waypoint[0],
                waypoint[1],
                waypoint[2],
                color=color,
                s=100,
                marker='*'
            )
            
            # Relier les waypoints par une ligne
            if i < len(self.mission_waypoints) - 1:
                next_waypoint = self.mission_waypoints[i+1]
                self.ax.plot(
                    [waypoint[0], next_waypoint[0]],
                    [waypoint[1], next_waypoint[1]],
                    [waypoint[2], next_waypoint[2]],
                    color='gray',
                    linestyle='--',
                    alpha=0.5
                )
        
        # Afficher la légende
        self.ax.text2D(0.02, 0.95, "Rouge: Leader", transform=self.ax.transAxes, color='red')
        self.ax.text2D(0.02, 0.92, "Bleu: Suiveur", transform=self.ax.transAxes, color='blue')
        self.ax.text2D(0.02, 0.89, "Orange: Défaillant", transform=self.ax.transAxes, color='orange')
        self.ax.text2D(0.02, 0.86, "Noir: En panne", transform=self.ax.transAxes, color='black')
        self.ax.text2D(0.02, 0.83, f"Waypoint actuel: {self.current_waypoint_index+1}/{len(self.mission_waypoints)}", transform=self.ax.transAxes)
        
        # Compter les drones par statut
        active_count = sum(1 for d in self.drones if d.status == DroneStatus.ACTIVE)
        failing_count = sum(1 for d in self.drones if d.status == DroneStatus.FAILING)
        failed_count = sum(1 for d in self.drones if d.status == DroneStatus.FAILED)
        
        self.ax.text2D(0.02, 0.80, f"Drones actifs: {active_count}", transform=self.ax.transAxes)
        self.ax.text2D(0.02, 0.77, f"Drones défaillants: {failing_count}", transform=self.ax.transAxes)
        self.ax.text2D(0.02, 0.74, f"Drones en panne: {failed_count}", transform=self.ax.transAxes)
        
        plt.pause(0.01)
    
    def run_simulation(self, num_steps=1000):
        """Exécute la simulation pour un nombre défini de pas de temps"""
        self.setup_visualization()
        
        for step in range(num_steps):
            self.update()
            self.visualize()
            
            # Vérifier si tous les drones sont en panne
            if all(drone.status == DroneStatus.FAILED for drone in self.drones):
                print("Tous les drones sont en panne. Fin de la simulation.")
                break
        
        plt.show()

# Exemple d'utilisation
if __name__ == "__main__":
    simulation = DroneSwarmSimulation(num_drones=5)
    simulation.run_simulation(num_steps=1000)