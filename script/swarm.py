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

class Drone:
    def __init__(self, drone_id, position, velocity):
        self.id = drone_id
        self.position = np.array(position, dtype=float)
        self.velocity = np.array(velocity, dtype=float)
        self.acceleration = np.zeros(3)
        self.status = DroneStatus.ACTIVE
        
        # Paramètres de communication et perception
        self.communication_range = 70.0  # Augmenté pour améliorer la coordination
        self.perception_range = 60.0
        
        # Gestion du réseau et des voisins
        self.neighbors = []
        self.last_known_positions = {}
        
        # Informations de consensus
        self.proposed_formation = "V"
        self.agreed_formation = "V"
        self.voted_formation = None
        self.votes_received = {}
        self.formation_position = None
        
        # État et apparence
        self.color = 'blue'
        self.failure_probability = 0.0002  # Réduit pour moins de défaillances
        self.failing_countdown = 20
        
        # Paramètres de l'algorithme de consensus
        self.consensus_iterations = 2
        self.consensus_threshold = 0.7

    def distance_to(self, other):
        """Calcule la distance entre ce drone et un autre drone"""
        return np.linalg.norm(self.position - other.position)
    
    def can_communicate_with(self, other_drone):
        """Vérifie si le drone peut communiquer avec un autre drone"""
        return (self.distance_to(other_drone) <= self.communication_range and 
                other_drone.status != DroneStatus.FAILED)
    
    def discover_neighbors(self, swarm):
        """Découvre les drones voisins dans le rayon de communication"""
        self.neighbors = []
        
        for drone in swarm:
            if drone.id != self.id and self.can_communicate_with(drone):
                self.neighbors.append(drone.id)
                self.last_known_positions[drone.id] = drone.position.copy()
    
    def normalize(self, v):
        """Normalise un vecteur"""
        norm = np.linalg.norm(v)
        if norm > 0:
            return v / norm
        return v
    
    # ALGORITHME DE CONSENSUS OPTIMISÉ
    def propose_formation_s(self, swarm):
        """Propose un type de formation basé sur la taille de l'essaim et la mission actuelle"""
        active_drones = [d for d in swarm if d.status != DroneStatus.FAILED]
        num_drones = len(active_drones)
        
        # Ajout d'une chance de proposer la formation "shoal" (banc)
        if random.random() < 0.3:  # 30% de chance de proposer shoal
            self.proposed_formation = "shoal"
            return self.proposed_formation
        
        # Logique existante pour les autres formations
        if num_drones <= 5:
            self.proposed_formation = "V"
        elif num_drones <= 10:
            positions = [d.position for d in active_drones]
            dispersion = np.std(positions, axis=0)
            if np.mean(dispersion) > 20:
                self.proposed_formation = "line"
            else:
                self.proposed_formation = "V"
        else:
            self.proposed_formation = "circle"
            
        return self.proposed_formation
    
    def propose_formation(self, swarm):
        self.proposed_formation = "V"
        return self.proposed_formation
       
    
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
            self.propose_formation(swarm)
            
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
        
        # Si aucun consensus n'est atteint, utiliser le vote le plus populaire
        if self.voted_formation:
            self.agreed_formation = self.voted_formation
        else:
            self.agreed_formation = "V"  # Formation par défaut
        
        return self.agreed_formation
    
    # CONTRÔLE DE FORMATION AMÉLIORÉ
    def calculate_formation_position(self, swarm, target):
        """Calcule la position que le drone devrait occuper dans la formation, avec option banc de poissons"""
        if self.status == DroneStatus.FAILED:
            return
            
        # S'assurer qu'un consensus sur la formation a été atteint
        if self.agreed_formation is None:
            self.reach_consensus(swarm)
            
        # Trouver les drones actifs
        active_drones = [d for d in swarm if d.status != DroneStatus.FAILED]
        if not active_drones:
            return
            
        # Calculer le centre de l'essaim
        center = np.mean([d.position for d in active_drones], axis=0)
        
        # Direction vers la cible à partir du centre
        direction_to_target = target - center
        direction = self.normalize(direction_to_target)
        
        # Obtenir l'index du drone parmi les drones actifs
        active_ids = sorted([d.id for d in active_drones])
        if self.id not in active_ids:
            return
            
        idx = active_ids.index(self.id)
        num_drones = len(active_ids)
        
        # Ajustement pour petits essaims: réduire les offsets verticaux
        vertical_adjustment = min(1.0, num_drones / 10.0)
        
        # Facteur d'espacement global
        spacing_factor = 2.0
        
        # NOUVEAU: Formation "shoal" (banc de poissons)
        if self.agreed_formation == "shoal":
            # Utiliser l'ID du drone comme graine pour le générateur aléatoire
            # pour obtenir une position cohérente pour chaque drone
            random.seed(self.id)
            
            # Distance maximale au centre (pour contrôler la dispersion)
            max_radius = 40.0
            
            # Créer un offset aléatoire mais stable pour chaque drone
            # avec une tendance à rester plus près du centre
            radius = random.random() * max_radius
            
            # Angle aléatoire sur le plan horizontal
            theta = random.uniform(0, 2 * math.pi)
            
            # Angle vertical (plus restreint pour éviter trop de dispersion verticale)
            phi = random.uniform(-math.pi/6, math.pi/6)
            
            # Convertir en coordonnées cartésiennes
            offset_x = radius * math.cos(theta) * math.cos(phi)
            offset_y = radius * math.sin(theta) * math.cos(phi)
            offset_z = radius * math.sin(phi) * 0.5  # Réduit la dispersion verticale
            
            # Réinitialiser la graine aléatoire pour ne pas affecter d'autres comportements
            random.seed()
            
        # Formations structurées existantes
        elif self.agreed_formation == "V":
            angle = math.pi / 4
            distance = 25.0 * spacing_factor
            
            side = 1 if idx % 2 == 0 else -1
            row = (idx // 2) + 1
            
            offset_x = -row * distance * math.cos(angle)
            offset_y = side * row * distance * math.sin(angle)
            offset_z = -row * 3.0 * vertical_adjustment
            
        elif self.agreed_formation == "line":
            distance = 20.0 * spacing_factor
            offset_x = -idx * distance
            offset_y = 0
            offset_z = 0
            
        elif self.agreed_formation == "circle":
            radius = 35.0 * spacing_factor
            angle = 2 * math.pi * idx / max(1, num_drones)
            offset_x = radius * math.cos(angle)
            offset_y = radius * math.sin(angle)
            offset_z = 0
            
        else:
            # Formation par défaut (V)
            angle = math.pi / 4
            distance = 25.0 * spacing_factor
            side = 1 if idx % 2 == 0 else -1
            row = (idx // 2) + 1
            offset_x = -row * distance * math.cos(angle)
            offset_y = side * row * distance * math.sin(angle)
            offset_z = -row * 3.0 * vertical_adjustment
        
        # Pour la formation en banc, ajouter une composante dynamique
        if self.agreed_formation == "shoal":
            # Ajouter une petite variation temporelle à la position
            # pour créer un mouvement subtil similaire à celui d'un banc de poissons
            time_factor = time.time() % 10  # Cycle de 10 secondes
            
            # Variation sinusoïdale basée sur le temps et l'ID (chaque drone a un pattern différent)
            wave_x = math.sin(time_factor + self.id * 0.7) * 5.0
            wave_y = math.cos(time_factor + self.id * 1.3) * 5.0
            wave_z = math.sin(time_factor + self.id * 2.1) * 2.0
            
            # Ajouter ces variations à la position
            offset_x += wave_x
            offset_y += wave_y
            offset_z += wave_z
        
        # Matrice de transformation pour les formations orientées
        if self.agreed_formation != "shoal":
            # Pour les formations structurées, orienter selon la direction cible
            forward = direction
            up = np.array([0, 0, 0.5])
            right = np.cross(forward, up)
            right = right / np.linalg.norm(right) if np.linalg.norm(right) > 0 else np.array([0, 1, 0])
            
            up = np.cross(right, forward)
            up = up / np.linalg.norm(up) if np.linalg.norm(up) > 0 else np.array([0, 0, 0.5])
            
            transform = np.column_stack([forward, right, up])
            relative_pos = np.array([offset_x, offset_y, offset_z])
            global_offset = transform @ relative_pos
        else:
            # Pour la formation en banc, pas besoin de transformation complexe
            # car les offsets sont déjà en coordonnées globales
            global_offset = np.array([offset_x, offset_y, offset_z])
            
            # Légère tendance à orienter le banc vers la cible
            global_offset = global_offset + direction * 5.0
        
        # Position de formation finale
        formation_pos = center + global_offset
        
        # S'assurer que la formation ne descend pas trop bas
        min_height = 40.0
        if formation_pos[2] < min_height:
            formation_pos[2] = min_height
        
        self.formation_position = formation_pos

    def calculate_steering_forces(self, swarm, target):
        """Calcule les forces de pilotage avec correction pour l'attraction vers le bas"""
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
        separation_weight = 1.2     # Éviter les collisions
        alignment_weight = 0.8      # Aligner sa direction
        cohesion_weight = 0.8       # Rester groupé
        formation_weight = 2.0      # Formation
        target_weight = 2.0         # Suivi de cible
        
        # AJOUT: Force de maintien d'altitude
        altitude_weight = 1.5       # Maintien d'altitude
        
        # Mettre à jour la liste des voisins
        self.discover_neighbors(swarm)
        
        # Obtenir les drones voisins
        neighbors = [d for d in swarm if d.id in self.neighbors]
        
        if not neighbors:
            # Si isolé, se diriger directement vers la cible
            return self.normalize(target - self.position) * 1.5
        
        # 1. Force de séparation
        separation_force = np.zeros(3)
        separation_count = 0
        
        for neighbor in neighbors:
            distance = self.distance_to(neighbor)
            if distance < 10.0:
                repulsion = self.position - neighbor.position
                
                if distance > 0:
                    repulsion = repulsion / distance
                    separation_force += repulsion
                    separation_count += 1
        
        if separation_count > 0:
            separation_force /= separation_count
            separation_force = self.normalize(separation_force)
            
        # 2. Force d'alignement
        alignment_force = np.zeros(3)
        alignment_count = 0
        
        for neighbor in neighbors:
            alignment_force += neighbor.velocity
            alignment_count += 1
        
        if alignment_count > 0:
            alignment_force /= alignment_count
            alignment_force = self.normalize(alignment_force)
        
        # 3. Force de cohésion
        cohesion_force = np.zeros(3)
        cohesion_count = 0
        
        for neighbor in neighbors:
            dist_to_target = np.linalg.norm(neighbor.position - target)
            if dist_to_target < 50.0:
                cohesion_force += neighbor.position * 2
                cohesion_count += 2
            else:
                cohesion_force += neighbor.position
                cohesion_count += 1
        
        if cohesion_count > 0:
            cohesion_force /= cohesion_count
            cohesion_force = self.normalize(cohesion_force - self.position)
        
        # 4. Force de formation
        formation_force = np.zeros(3)
        
        if self.formation_position is not None:
            formation_force = self.formation_position - self.position
            formation_force = self.normalize(formation_force)
        
        # 5. Force vers la cible
        dist_to_target = np.linalg.norm(self.position - target)
        target_force = self.normalize(target - self.position)
        if dist_to_target > 100.0:
            target_weight *= 1.5
        
        # NOUVEAU: 6. Force de maintien d'altitude
        # Cette force pousse le drone vers le haut s'il descend trop bas
        altitude_force = np.zeros(3)
        min_altitude = 40.0
        
        if self.position[2] < min_altitude:
            # Plus on est bas sous l'altitude minimale, plus la force vers le haut est forte
            altitude_force = np.array([0, 0, 1]) * (min_altitude - self.position[2]) / 10.0
        
        # Calculer la force résultante en appliquant les poids
        steering_force = (
            separation_force * separation_weight +
            alignment_force * alignment_weight +
            cohesion_force * cohesion_weight +
            formation_force * formation_weight +
            target_force * target_weight +
            altitude_force * altitude_weight  # Ajout de la force d'altitude
        )
        
        return steering_force
    
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
    
    def update(self, swarm, target):
        """Met à jour l'état du drone en fonction de l'environnement"""
        # Mise à jour du statut
        self.update_status()
        
        # Calcul de la position dans la formation par consensus
        self.calculate_formation_position(swarm, target)
        
        # Calculer les forces de pilotage
        steering_force = self.calculate_steering_forces(swarm, target)
        
        # Appliquer l'accélération (avec limite)
        self.acceleration = steering_force
        max_acceleration = 0.3  # Augmenté pour plus de réactivité
        acc_magnitude = np.linalg.norm(self.acceleration)
        if acc_magnitude > max_acceleration:
            self.acceleration = self.acceleration * (max_acceleration / acc_magnitude)
        
        # Mettre à jour la vitesse
        self.velocity += self.acceleration
        
        # Limiter la vitesse
        max_speed = 1.0 if self.status == DroneStatus.ACTIVE else 0.5  # Augmenté pour plus de vitesse
        speed = np.linalg.norm(self.velocity)
        if speed > max_speed:
            self.velocity = self.velocity * (max_speed / speed)
        
        # Mettre à jour la position
        self.position += self.velocity

class DroneSwarmSimulation:
    def __init__(self, num_drones=10):
        # Garder le reste des initialisations comme avant
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
        self.start_time = time.time()
        self.simulation_stats = {
            'mean_distance': [],
            'formation_quality': [],
            'mission_progress': 0
        }
        
        # Ajouter un attribut pour la formation actuelle de l'essaim
        self.current_formation = "V"  # Formation par défaut
        
        # Initialiser l'essaim de drones
        self.initialize_swarm()

    def change_formation(self, formation_type):
        """Change la formation actuelle de l'essaim"""
        if formation_type in ["V", "line", "circle", "shoal"]:
            self.current_formation = formation_type
            print(f"Formation changée pour: {formation_type}")
            
            # Appliquer la nouvelle formation à tous les drones
            for drone in self.drones:
                drone.proposed_formation = formation_type
                drone.agreed_formation = formation_type
                # Réinitialiser les votes pour éviter des conflits
                drone.votes_received = {}
        else:
            print(f"Formation {formation_type} non reconnue")
    
    def initialize_swarm(self):
        """Initialise l'essaim de drones avec une formation banc de poissons par défaut"""
        for i in range(self.num_drones):
            position = [
                random.uniform(-10, 10),
                random.uniform(-10, 10),
                random.uniform(50, 60)
            ]
            
            velocity = [
                random.uniform(-0.2, 0.2),
                random.uniform(-0.2, 0.2),
                random.uniform(0, 0.1)
            ]
            
            drone = Drone(drone_id=i, position=position, velocity=velocity)
            # Définir la formation banc par défaut
            drone.proposed_formation = "shoal"
            drone.agreed_formation = "shoal"
            self.drones.append(drone)
    
    def update_mission_target(self):
        """Met à jour la cible de mission avec une meilleure détection d'arrivée"""
        current_target = self.mission_waypoints[self.current_waypoint_index]
        
        # Vérifier si suffisamment de drones sont proches de la cible
        drones_at_target = 0
        active_drones = [d for d in self.drones if d.status != DroneStatus.FAILED]
        
        # Calculer le centre de l'essaim
        if active_drones:
            swarm_center = np.mean([d.position for d in active_drones], axis=0)
            distance_center_to_target = np.linalg.norm(swarm_center - current_target)
            
            # Si le centre de l'essaim est proche de la cible
            if distance_center_to_target < 30.0:  # Plus permissif
                self.current_waypoint_index = (self.current_waypoint_index + 1) % len(self.mission_waypoints)
                print(f"Nouveau waypoint: {self.current_waypoint_index+1}")
                return
        
        # Vérification individuelle des drones (méthode originale)
        for drone in active_drones:
            if np.linalg.norm(drone.position - current_target) < 25.0:  # Distance augmentée
                drones_at_target += 1
        
        # Si plus de 40% des drones actifs ont atteint la cible, passer à la suivante
        # Seuil réduit pour faciliter la progression
        if drones_at_target >= len(active_drones) * 0.4:  
            self.current_waypoint_index = (self.current_waypoint_index + 1) % len(self.mission_waypoints)
            print(f"Nouveau waypoint: {self.current_waypoint_index+1}")
    
    def update(self):
        """Met à jour l'état global de la simulation"""
        # Mettre à jour la cible de mission
        self.update_mission_target()
        
        # Cible actuelle
        target = self.mission_waypoints[self.current_waypoint_index]
        
        # Mettre à jour chaque drone
        for drone in self.drones:
            drone.update(self.drones, target)
    
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
        formation_error = 0
        for drone in active_drones:
            if drone.formation_position is not None:
                error = np.linalg.norm(drone.position - drone.formation_position)
                formation_error += error
                
        if active_drones:
            avg_formation_error = formation_error / len(active_drones)
            formation_quality = max(0, 1 - (avg_formation_error / 20))  # Normaliser
            self.simulation_stats['formation_quality'].append(formation_quality)
        
        # Progression de mission
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
                s=30,
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
            
            # Visualiser la position dans la formation
            if drone.formation_position is not None and drone.status == DroneStatus.ACTIVE:
                self.ax.plot(
                    [drone.position[0], drone.formation_position[0]],
                    [drone.position[1], drone.formation_position[1]],
                    [drone.position[2], drone.formation_position[2]],
                    color='lightblue',
                    linestyle=':',
                    alpha=0.3
                )
            
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
            f"Formation actuelle: {self.current_formation}",  # Afficher la formation actuellement sélectionnée
            f"Contrôles: v-V, l-Ligne, c-Cercle, s-Banc"  # Rappel des contrôles
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
            "Vert: Waypoint actuel"
        ])
        
        # Afficher chaque ligne d'information
        for i, text in enumerate(info_text):
            y_pos = 0.95 - i * 0.03
            self.ax.text2D(0.02, y_pos, text, transform=self.ax.transAxes)
    
    def run_simulation(self, num_steps=1000):
        """Exécute la simulation avec animation"""
        self.setup_visualization()
        
        # Animation function
        def update_frame(frame):
            self.update()
            self.visualize()
            return self.ax,
        
        # Créer l'animation avec un intervalle plus court pour plus de vitesse
        ani = FuncAnimation(self.fig, update_frame, frames=num_steps, 
                           interval=30, blit=False, repeat=False)
        
        plt.show()
    
    def run_simulation_fast(self, num_steps=1000):
        """Exécute la simulation en mode rapide sans animation complexe"""
        self.setup_visualization()
        
        for step in range(num_steps):
            self.update()
            
            # Visualiser moins fréquemment pour accélérer la simulation
            if step % 5 == 0:
                self.visualize()
                plt.pause(0.01)  # Pause très courte
            
            # Vérifier si tous les drones sont en panne
            if all(drone.status == DroneStatus.FAILED for drone in self.drones):
                print("Tous les drones sont en panne. Fin de la simulation.")
                break
        plt.show()

    def run_simulation_interactive(self, num_steps=1000):
        """Exécute la simulation avec contrôle interactif de la formation"""
        self.setup_visualization()
        
        # Fonction de gestion des événements clavier
        def on_key(event):
            if event.key == 'v':
                self.change_formation("V")
            elif event.key == 'l':
                self.change_formation("line")
            elif event.key == 'c':
                self.change_formation("circle")
            elif event.key == 's':
                self.change_formation("shoal")
        
        # Connecter l'événement clavier
        self.fig.canvas.mpl_connect('key_press_event', on_key)
        
        # Animation function
        def update_frame(frame):
            self.update()
            self.visualize()
            return self.ax,
        
        # Afficher les instructions
        print("\nContrôles de formation:")
        print("  'v' - Formation en V")
        print("  'l' - Formation en ligne")
        print("  'c' - Formation en cercle")
        print("  's' - Formation en banc de poissons (shoal)")
        
        # Créer l'animation
        ani = FuncAnimation(self.fig, update_frame, frames=num_steps, 
                           interval=30, blit=False, repeat=False)
        
        plt.show()

# Exemple d'utilisation
if __name__ == "__main__":
    # Créer la simulation avec plus de drones pour mieux voir les formations
    simulation = DroneSwarmSimulation(num_drones=9)
    
    # Utiliser le mode rapide
    simulation.run_simulation_fast(num_steps=1000)
    
    # Alternative: mode normal
    # simulation.run_simulation(num_steps=1000)