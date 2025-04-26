import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D
import random
import math
from enum import Enum
import time

class DroneStatus(Enum):
    ACTIVE = 1
    FAILING = 2
    FAILED = 3

class ObstacleType(Enum):
    STATIC = 1
    MOVING = 2

class Obstacle:
    def __init__(self, position, size, obstacle_type=ObstacleType.STATIC, velocity=None):
        self.position = np.array(position, dtype=float)
        self.size = size  # Rayon de l'obstacle
        self.type = obstacle_type
        self.velocity = np.zeros(3) if velocity is None else np.array(velocity, dtype=float)
        
    def update(self):
        if self.type == ObstacleType.MOVING:
            self.position += self.velocity

class Drone:
    def __init__(self, drone_id, position, velocity):
        self.id = drone_id
        self.position = np.array(position, dtype=float)
        self.velocity = np.array(velocity, dtype=float)
        self.acceleration = np.zeros(3)
        self.status = DroneStatus.ACTIVE
        
        # Paramètres de communication et perception
        self.communication_range = 50.0
        self.perception_range = 40.0
        
        # Gestion du réseau et des voisins
        self.neighbors = []
        self.last_known_positions = {}
        self.network_graph = {}  # Pour stocker la topologie du réseau
        
        # Informations de consensus
        self.consensus_values = {}
        self.proposed_formation = None
        self.agreed_formation = None
        self.voted_formation = None
        self.votes_received = {}
        self.formation_position = None
        
        # Planification de trajectoire
        self.path = []
        self.current_waypoint_index = 0
        self.local_target = None
        self.global_target = None
        
        # État et apparence
        self.color = 'blue'
        self.failure_probability = 0.0005
        self.failing_countdown = 20
        
        # Paramètres de l'algorithme de consensus
        self.consensus_iterations = 3
        self.consensus_threshold = 0.8  # Pourcentage requis pour consensus

    def distance_to(self, other):
        """Calcule la distance entre ce drone et un autre objet (drone ou obstacle)"""
        return np.linalg.norm(self.position - other.position)
    
    def can_communicate_with(self, other_drone):
        """Vérifie si le drone peut communiquer avec un autre drone"""
        return (self.distance_to(other_drone) <= self.communication_range and 
                other_drone.status != DroneStatus.FAILED)
    
    def discover_neighbors(self, swarm):
        """Découvre les drones voisins dans le rayon de communication"""
        self.neighbors = []
        self.network_graph = {}
        
        for drone in swarm:
            if drone.id != self.id and self.can_communicate_with(drone):
                self.neighbors.append(drone.id)
                self.last_known_positions[drone.id] = drone.position.copy()
                
                # Collecte des informations sur le réseau pour la topologie
                self.network_graph[drone.id] = [d.id for d in swarm 
                                              if d.id != drone.id and drone.can_communicate_with(d)]
    
    def normalize(self, v):
        """Normalise un vecteur"""
        norm = np.linalg.norm(v)
        if norm > 0:
            return v / norm
        return v
    
    # ALGORITHME DE CONSENSUS
    def propose_formation(self, formation_type="V"):
        """Propose un type de formation basé sur l'environnement actuel"""
        # Actuellement supporte "V", "line", "circle"
        self.proposed_formation = formation_type
        return formation_type
    
    def vote_on_formation(self, swarm):
        """Vote pour une formation basée sur les propositions des voisins"""
        proposals = {}
        
        # Collecter toutes les propositions dans le voisinage
        for drone in swarm:
            if drone.id in self.neighbors and drone.proposed_formation is not None:
                if drone.proposed_formation not in proposals:
                    proposals[drone.proposed_formation] = 0
                proposals[drone.proposed_formation] += 1
        
        # Ajouter sa propre proposition
        if self.proposed_formation is not None:
            if self.proposed_formation not in proposals:
                proposals[self.proposed_formation] = 0
            proposals[self.proposed_formation] += 1
        
        # Voter pour la formation la plus populaire
        if proposals:
            most_popular = max(proposals, key=proposals.get)
            self.voted_formation = most_popular
        else:
            self.voted_formation = "V"  # Formation par défaut
        
        return self.voted_formation
    
    def reach_consensus(self, swarm):
        """Exécute l'algorithme de consensus pour décider de la formation"""
        # Initialiser le vote
        if self.proposed_formation is None:
            self.propose_formation()
            
        # Exécuter plusieurs itérations pour converger vers un consensus
        for _ in range(self.consensus_iterations):
            # Voter
            vote = self.vote_on_formation(swarm)
            
            # Partager le vote avec les voisins
            for drone in swarm:
                if drone.id in self.neighbors:
                    if drone.id not in self.votes_received:
                        self.votes_received[drone.id] = {}
                    self.votes_received[drone.id][self.id] = vote
            
            # Collecter les votes des voisins
            all_votes = []
            for drone_id, votes in self.votes_received.items():
                all_votes.extend(votes.values())
            
            # Ajouter son propre vote
            all_votes.append(vote)
            
            # Compter les votes
            vote_counts = {}
            for v in all_votes:
                if v not in vote_counts:
                    vote_counts[v] = 0
                vote_counts[v] += 1
            
            # Vérifier si le consensus est atteint
            if vote_counts and max(vote_counts.values()) / len(all_votes) >= self.consensus_threshold:
                self.agreed_formation = max(vote_counts, key=vote_counts.get)
                return self.agreed_formation
        
        # Si aucun consensus n'est atteint après toutes les itérations, utiliser le vote le plus populaire
        if self.voted_formation:
            self.agreed_formation = self.voted_formation
        else:
            self.agreed_formation = "V"  # Formation par défaut
        
        return self.agreed_formation
    
    # CONTRÔLE DE FORMATION
    def calculate_formation_position(self, swarm):
        """Calcule la position que le drone devrait occuper dans la formation convenue"""
        if self.status == DroneStatus.FAILED:
            return
            
        if self.agreed_formation is None:
            self.reach_consensus(swarm)
            
        # Trouver le centre de l'essaim
        active_drones = [d for d in swarm if d.status != DroneStatus.FAILED]
        if not active_drones:
            return
            
        # Calculer le centre de l'essaim
        center = np.mean([d.position for d in active_drones], axis=0)
        
        # Calculer la direction moyenne de déplacement
        avg_velocity = np.mean([d.velocity for d in active_drones], axis=0)
        direction = self.normalize(avg_velocity) if np.linalg.norm(avg_velocity) > 0 else np.array([1, 0, 0])
        
        # Obtenir l'index du drone parmi les drones actifs
        active_ids = sorted([d.id for d in active_drones])
        if self.id not in active_ids:
            return
            
        idx = active_ids.index(self.id)
        num_drones = len(active_ids)
        
        # Calculer la position en fonction du type de formation
        if self.agreed_formation == "V":
            # Formation en V
            angle = math.pi / 4  # 45 degrés
            distance = 15.0
            
            side = 1 if idx % 2 == 0 else -1
            row = (idx // 2) + 1
            
            offset_x = -row * distance * math.cos(angle)
            offset_y = side * row * distance * math.sin(angle)
            offset_z = -row * 5.0  # Légèrement plus bas
            
        elif self.agreed_formation == "line":
            # Formation en ligne
            distance = 10.0
            offset_x = -idx * distance
            offset_y = 0
            offset_z = 0
            
        elif self.agreed_formation == "circle":
            # Formation en cercle
            radius = 20.0
            angle = 2 * math.pi * idx / num_drones
            offset_x = radius * math.cos(angle)
            offset_y = radius * math.sin(angle)
            offset_z = 0
            
        else:
            # Formation par défaut (V)
            angle = math.pi / 4
            distance = 15.0
            side = 1 if idx % 2 == 0 else -1
            row = (idx // 2) + 1
            offset_x = -row * distance * math.cos(angle)
            offset_y = side * row * distance * math.sin(angle)
            offset_z = -row * 5.0
        
        # Convertir les offsets relatifs en position globale
        # Définir les axes de la formation en fonction de la direction moyenne
        forward = direction
        up = np.array([0, 0, 1])
        right = np.cross(forward, up)
        right = right / np.linalg.norm(right) if np.linalg.norm(right) > 0 else np.array([0, 1, 0])
        
        # Recalculer 'up' pour s'assurer qu'il est orthogonal
        up = np.cross(right, forward)
        up = up / np.linalg.norm(up) if np.linalg.norm(up) > 0 else np.array([0, 0, 1])
        
        # Matrice de transformation
        transform = np.column_stack([forward, right, up])
        relative_pos = np.array([offset_x, offset_y, offset_z])
        global_offset = transform @ relative_pos
        
        self.formation_position = center + global_offset
    
    # ÉVITEMENT DE COLLISION
    def detect_collisions(self, swarm, obstacles):
        """Détecte les collisions potentielles avec d'autres drones ou obstacles"""
        collision_vectors = []
        
        # Évitement des autres drones
        for drone in swarm:
            if drone.id != self.id and drone.status != DroneStatus.FAILED:
                distance = self.distance_to(drone)
                if distance < 10.0:  # Distance de sécurité
                    repulsion = self.position - drone.position
                    strength = (10.0 - distance) / 10.0  # Plus proche = plus forte répulsion
                    if distance > 0:
                        normalized = self.normalize(repulsion)
                        collision_vectors.append(normalized * strength)
        
        # Évitement des obstacles
        for obstacle in obstacles:
            distance = self.distance_to(obstacle) - obstacle.size  # Tenir compte de la taille
            if distance < 15.0:  # Distance de sécurité
                repulsion = self.position - obstacle.position
                strength = (15.0 - distance) / 15.0
                if distance > -obstacle.size:  # Éviter division par zéro et valeurs négatives
                    normalized = self.normalize(repulsion)
                    collision_vectors.append(normalized * strength * 2.0)  # Force d'évitement plus importante
        
        # Combiner tous les vecteurs d'évitement
        if collision_vectors:
            return np.mean(collision_vectors, axis=0)
        return np.zeros(3)
    
    # PLANIFICATION DE CHEMIN
    def plan_path(self, target, obstacles, swarm):
        """Planification de chemin simple avec évitement d'obstacles"""
        if self.status == DroneStatus.FAILED:
            return []
            
        # Direction de base vers la cible
        direction_to_target = target - self.position
        distance_to_target = np.linalg.norm(direction_to_target)
        
        if distance_to_target < 5.0:  # Si proche de la cible, pas besoin de planifier
            return [target]
            
        # Vérifier s'il y a des obstacles sur le chemin direct
        path_clear = True
        
        for obstacle in obstacles:
            # Vecteur de la position actuelle à l'obstacle
            to_obstacle = obstacle.position - self.position
            
            # Projection du vecteur obstacle sur la direction vers la cible
            projection_length = np.dot(to_obstacle, self.normalize(direction_to_target))
            
            # Si l'obstacle est devant nous et dans la direction de la cible
            if 0 < projection_length < distance_to_target:
                # Distance perpendiculaire à la ligne de vue
                perpendicular = to_obstacle - projection_length * self.normalize(direction_to_target)
                perp_distance = np.linalg.norm(perpendicular)
                
                # Si l'obstacle est proche de notre ligne de vue
                if perp_distance < obstacle.size + 10.0:
                    path_clear = False
                    break
        
        if path_clear:
            # Chemin direct vers la cible
            return [target]
        else:
            # Créer un point intermédiaire pour contourner les obstacles
            # Cette approche est simplifiée; un algorithme plus sophistiqué 
            # comme A* ou RRT serait nécessaire pour des environnements complexes
            avoidance_direction = self.detect_collisions(swarm, obstacles)
            
            if np.linalg.norm(avoidance_direction) > 0:
                # Créer un waypoint intermédiaire qui évite les obstacles
                intermediate = self.position + self.normalize(avoidance_direction) * 20.0
                
                # Ajouter une composante vers la cible pour ne pas s'éloigner trop
                intermediate += self.normalize(direction_to_target) * 10.0
                
                return [intermediate, target]
            else:
                # Pas d'obstacle détecté malgré tout, chemin direct
                return [target]
    
    def update_path(self, target, obstacles, swarm):
        """Met à jour le chemin planifié en fonction des changements d'environnement"""
        self.global_target = target
        self.path = self.plan_path(target, obstacles, swarm)
        self.current_waypoint_index = 0
        
        if self.path:
            self.local_target = self.path[0]
    
    def follow_path(self):
        """Suit le chemin planifié"""
        if not self.path or self.status == DroneStatus.FAILED:
            return np.zeros(3)
            
        # Vérifier si nous avons atteint le waypoint actuel
        current_waypoint = self.path[self.current_waypoint_index]
        distance = np.linalg.norm(self.position - current_waypoint)
        
        if distance < 5.0:  # Waypoint atteint
            self.current_waypoint_index = min(self.current_waypoint_index + 1, len(self.path) - 1)
            
        # Diriger vers le waypoint actuel
        if self.current_waypoint_index < len(self.path):
            target = self.path[self.current_waypoint_index]
            direction = target - self.position
            return self.normalize(direction)
            
        return np.zeros(3)
    
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
    
    def calculate_steering_forces(self, swarm, obstacles, target):
        """Calcule toutes les forces de pilotage selon les règles de comportement"""
        if self.status == DroneStatus.FAILED:
            # Les drones défaillants tombent
            return np.array([0, 0, -0.5])
            
        if self.status == DroneStatus.FAILING:
            # Les drones en défaillance sont instables
            return np.array([
                random.uniform(-0.5, 0.5),
                random.uniform(-0.5, 0.5),
                random.uniform(-0.5, 0.1)
            ])
        
        # Poids des différentes forces
        separation_weight = 1.5     # Éviter les collisions avec d'autres drones
        alignment_weight = 1.0      # Aligner sa direction avec celle des voisins
        cohesion_weight = 1.0       # Rester proche du groupe
        formation_weight = 2.0      # Maintenir la formation
        path_weight = 1.5           # Suivre le chemin planifié
        obstacle_weight = 2.5       # Éviter les obstacles (priorité élevée)
        
        # Initialiser la force résultante
        steering_force = np.zeros(3)
        
        # Mettre à jour la liste des voisins
        self.discover_neighbors(swarm)
        
        # Obtenir les drones voisins
        neighbors = [d for d in swarm if d.id in self.neighbors]
        
        if not neighbors:
            # Si isolé, tenter de revenir vers la dernière position connue d'un voisin
            if self.last_known_positions:
                closest_pos = min(self.last_known_positions.values(), 
                                key=lambda p: np.linalg.norm(p - self.position))
                return self.normalize(closest_pos - self.position) * 0.5
        
        # 1. Force de séparation (éviter les collisions avec d'autres drones)
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
        
        if self.formation_position is not None:
            formation_force = self.formation_position - self.position
            formation_force = self.normalize(formation_force)
        
        # 5. Force de suivi de chemin
        path_force = self.follow_path()
        
        # 6. Force d'évitement d'obstacles
        obstacle_force = self.detect_collisions(swarm, obstacles)
        
        # Calculer la force résultante en appliquant les poids
        steering_force = (
            separation_force * separation_weight +
            alignment_force * alignment_weight +
            cohesion_force * cohesion_weight +
            formation_force * formation_weight +
            path_force * path_weight +
            obstacle_force * obstacle_weight
        )
        
        return steering_force
    
    def update(self, swarm, obstacles, target):
        """Met à jour l'état du drone en fonction de l'environnement"""
        # Mise à jour du statut (simulation de défaillance)
        self.update_status()
        
        # Calcul de la position dans la formation par consensus
        self.calculate_formation_position(swarm)
        
        # Mise à jour du chemin si nécessaire
        if random.random() < 0.05:  # Mise à jour périodique du chemin
            self.update_path(target, obstacles, swarm)
        
        # Calculer les forces de pilotage
        steering_force = self.calculate_steering_forces(swarm, obstacles, target)
        
        # Appliquer l'accélération (avec limite)
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
    def __init__(self, num_drones=10):
        self.num_drones = num_drones
        self.drones = []
        self.obstacles = []
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
        self.start_time = time.time()
        self.simulation_stats = {
            'mean_distance': [],
            'consensus_time': [],
            'formation_quality': [],
            'mission_progress': 0
        }
        
        # Initialiser l'environnement
        self.initialize_environment()
    
    def initialize_environment(self):
        """Initialise l'environnement avec des obstacles et l'essaim de drones"""
        # Créer quelques obstacles statiques
        self.obstacles = [
            Obstacle([75, 75, 50], 15),
            Obstacle([175, 175, 60], 20),
            Obstacle([225, 125, 70], 15)
        ]
        
        # Ajouter des obstacles mobiles
        self.obstacles.append(
            Obstacle([120, 150, 65], 10, ObstacleType.MOVING, velocity=[0.3, 0.2, 0])
        )
        
        # Initialiser l'essaim de drones dans une région de départ
        for i in range(self.num_drones):
            # Position initiale aléatoire dans une petite région
            position = [
                random.uniform(-10, 10),
                random.uniform(-10, 10),
                random.uniform(45, 55)
            ]
            
            # Vitesse initiale faible et aléatoire
            velocity = [
                random.uniform(-0.2, 0.2),
                random.uniform(-0.2, 0.2),
                random.uniform(-0.1, 0.1)
            ]
            
            # Créer le drone
            drone = Drone(drone_id=i, position=position, velocity=velocity)
            self.drones.append(drone)
    
    def update_environment(self):
        """Met à jour l'état de l'environnement"""
        # Mettre à jour les obstacles mobiles
        for obstacle in self.obstacles:
            obstacle.update()
            
            # Faire rebondir les obstacles mobiles aux limites
            if obstacle.type == ObstacleType.MOVING:
                for i in range(3):
                    if obstacle.position[i] < -50 or obstacle.position[i] > 350:
                        obstacle.velocity[i] *= -1
    
    def update_mission_target(self):
        """Met à jour la cible de mission"""
        current_target = self.mission_waypoints[self.current_waypoint_index]
        
        # Vérifier si suffisamment de drones sont proches de la cible
        drones_at_target = 0
        active_drones = [d for d in self.drones if d.status != DroneStatus.FAILED]
        
        for drone in active_drones:
            if np.linalg.norm(drone.position - current_target) < 20.0:
                drones_at_target += 1
        
        # Si plus de 60% des drones actifs ont atteint la cible, passer à la suivante
        if drones_at_target >= len(active_drones) * 0.6:
            self.current_waypoint_index = (self.current_waypoint_index + 1) % len(self.mission_waypoints)
            print(f"Nouveau waypoint: {self.current_waypoint_index+1}")
    
    def update(self):
        """Met à jour l'état global de la simulation"""
        # Mettre à jour l'environnement
        self.update_environment()
        
        # Mettre à jour la cible de mission
        self.update_mission_target()
        
        # Cible actuelle
        target = self.mission_waypoints[self.current_waypoint_index]
        
        # Mettre à jour chaque drone
        for drone in self.drones:
            drone.update(self.drones, self.obstacles, target)
    
    def setup_visualization(self):
        """Prépare la visualisation 3D"""
        self.fig = plt.figure(figsize=(12, 10))
        self.ax = self.fig.add_subplot(111, projection='3d')
        self.configure_axes()
    
    def configure_axes(self):
        """Configure les axes du graphique"""
        self.ax.set_xlim([-50, 350])
        self.ax.set_ylim([-50, 250])
        self.ax.set_zlim([0, 100])
        self.ax.set_xlabel('X')
        self.ax.set_ylabel('Y')
        self.ax.set_zlabel('Z')
        self.ax.set_title('Simulation d\'Essaim de Drones Décentralisé en 3D')
    
    def collect_statistics(self):
        """Collecte des statistiques sur l'essaim"""
        active_drones = [d for d in self.drones if d.status != DroneStatus.FAILED]
        if not active_drones:
            return
            
        # Calculer la distance moyenne entre drones
        total_distance = 0
        count = 0
        for i, drone1 in enumerate(active_drones):
            for drone2 in active_drones[i+1:]:
                total_distance += np.linalg.norm(drone1.position - drone2.position)
                count += 1
                
        if count > 0:
            mean_distance = total_distance / count
            self.simulation_stats['mean_distance'].append(mean_distance)
        
        # Évaluer la qualité de la formation
        # Une mesure simple: écart moyen par rapport à la position idéale dans la formation
        formation_error = 0
        for drone in active_drones:
            if drone.formation_position is not None:
                error = np.linalg.norm(drone.position - drone.formation_position)
                formation_error += error
                
        if active_drones:
            avg_formation_error = formation_error / len(active_drones)
            formation_quality = max(0, 1 - (avg_formation_error / 20))  # Normaliser
            self.simulation_stats['formation_quality'].append(formation_quality)
        
        # Progression de mission (pourcentage de waypoints atteints)
        self.simulation_stats['mission_progress'] = (self.current_waypoint_index / len(self.mission_waypoints)) * 100
    
    def visualize(self):
        """Affiche l'état actuel de la simulation"""
        self.ax.clear()
        self.configure_axes()
        
        # Collecter des statistiques
        self.collect_statistics()
        
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
                    direction = drone.velocity / velocity_norm * 5
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
            
            # Afficher le chemin planifié
            if drone.path and len(drone.path) > 1:
                path_points = np.array([drone.position] + drone.path)
                self.ax.plot(
                    path_points[:, 0],
                    path_points[:, 1],
                    path_points[:, 2],
                    color=drone.color,
                    linestyle=':',
                    alpha=0.3
                )
        
        # Afficher les obstacles
        for obstacle in self.obstacles:
            if obstacle.type == ObstacleType.STATIC:
                color = 'gray'
            else:
                color = 'red'
                
            # Sphere simple pour représenter l'obstacle
            u = np.linspace(0, 2 * np.pi, 10)
            v = np.linspace(0, np.pi, 10)
            x = obstacle.position[0] + obstacle.size * np.outer(np.cos(u), np.sin(v))
            y = obstacle.position[1] + obstacle.size * np.outer(np.sin(u), np.sin(v))
            z = obstacle.position[2] + obstacle.size * np.outer(np.ones(np.size(u)), np.cos(v))
            self.ax.plot_surface(x, y, z, color=color, alpha=0.3)
            
        # Afficher les waypoints de mission
        for i, waypoint in enumerate(self.mission_waypoints):
            color = 'green' if i == self.current_waypoint_index else 'gray'
            self.ax.scatter(
                waypoint[0],
                waypoint[1],
                waypoint[2],
                color=color,
                s=150,
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
        
        # Afficher la légende et les statistiques
        elapsed_time = time.time() - self.start_time
        
        # Compter les drones par statut
        active_count = sum(1 for d in self.drones if d.status == DroneStatus.ACTIVE)
        failing_count = sum(1 for d in self.drones if d.status == DroneStatus.FAILING)
        failed_count = sum(1 for d in self.drones if d.status == DroneStatus.FAILED)
        
        # Obtenir la formation dominante
        formations = [d.agreed_formation for d in self.drones if d.agreed_formation]
        dominant_formation = max(set(formations), key=formations.count) if formations else "Aucune"
        
        # Afficher les informations
        info_text = [
            f"Temps: {elapsed_time:.1f}s",
            f"Waypoint: {self.current_waypoint_index+1}/{len(self.mission_waypoints)}",
            f"Drones actifs: {active_count}",
            f"Drones défaillants: {failing_count}",
            f"Drones en panne: {failed_count}",
            f"Formation: {dominant_formation}"
        ]
        
        # Afficher les statistiques si disponibles
        if self.simulation_stats['formation_quality']:
            info_text.append(f"Qualité formation: {self.simulation_stats['formation_quality'][-1]:.2f}")
            
        if self.simulation_stats['mean_distance']:
            info_text.append(f"Distance moyenne: {self.simulation_stats['mean_distance'][-1]:.1f}")
            
        # Ajouter des explications de couleur
        info_text.extend([
            "",
            "Bleu: Drone actif",
            "Orange: Drone défaillant",
            "Noir: Drone en panne",
            "Vert: Waypoint actuel",
            "Gris: Obstacle statique",
            "Rouge: Obstacle mobile"
        ])
        
        # Afficher chaque ligne d'information
        for i, text in enumerate(info_text):
            y_pos = 0.95 - i * 0.03
            self.ax.text2D(0.02, y_pos, text, transform=self.ax.transAxes)


    def run_simulation(self, num_steps=1000):
            """Exécute la simulation pour un nombre défini de pas de temps"""
            self.setup_visualization()
            
            # Animation function pour FuncAnimation
            def update_frame(frame):
                self.update()
                self.visualize()
                return self.ax,
            
            # Créer l'animation
            ani = FuncAnimation(self.fig, update_frame, frames=num_steps, 
                                interval=50, blit=False, repeat=False)
            
            plt.show()
    
    def run_simulation_step_by_step(self, num_steps=1000):
        """Exécute la simulation pas à pas (variante sans animation pour le débogage)"""
        self.setup_visualization()
        
        for step in range(num_steps):
            self.update()
            self.visualize()
            
            # Vérifier si tous les drones sont en panne
            if all(drone.status == DroneStatus.FAILED for drone in self.drones):
                print("Tous les drones sont en panne. Fin de la simulation.")
                break
                
            plt.pause(0.05)
        
        plt.show()


# Exemple d'utilisation
if __name__ == "__main__":
    # Pour l'analyse des performances (désactiver la visualisation)
    # results = analyze_swarm_performance()
    
    # Pour la visualisation interactive
    simulation = DroneSwarmSimulation(num_drones=10)
    simulation.run_simulation(num_steps=1000)
    
    # Pour une exécution pas à pas (utile pour le débogage)
    # simulation = DroneSwarmSimulation(num_drones=10)
    # simulation.run_simulation_step_by_step(num_steps=1000)