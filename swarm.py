import numpy as np
import random
import time
import math
from enum import Enum
import sys
from PyQt5 import QtWidgets, QtCore
import pyqtgraph as pg
import pyqtgraph.opengl as gl

class DroneStatus(Enum):
    ACTIVE = 1
    FAILING = 2
    FAILED = 3

class DroneRole(Enum):
    LEADER = 1
    FOLLOWER = 2

class WaypointStatus(Enum):
    NAVIGATING = 1
    REACHED = 2

class Message:
    """Message échangé entre les drones"""
    def __init__(self, sender_id, msg_type, data=None):
        self.sender_id = sender_id
        self.msg_type = msg_type  # position, formation_update, leader_heartbeat, etc.
        self.data = data if data is not None else {}
        self.timestamp = time.time()
        self.ttl = 5  # Time to live (nombre de sauts maximum) - augmenté pour meilleure propagation

class Drone:
    """Représente un drone individuel avec son comportement autonome"""
    def __init__(self, drone_id, position, velocity=None, waypoints=None):
        # Identité et statut
        self.id = drone_id
        self.status = DroneStatus.ACTIVE
        self.role = DroneRole.FOLLOWER
        self.leader_id = None
        
        # État physique
        self.position = np.array(position, dtype=float)
        self.velocity = np.array(velocity if velocity is not None else [0, 0, 0], dtype=float)
        self.acceleration = np.zeros(3)
        self.color = 'blue'
        
        # Communication et perception
        self.communication_range = 60.0
        self.neighbors = {}  # {drone_id: last_update_time}
        self.received_messages = []
        self.message_cache = {}  # Pour éviter de traiter plusieurs fois le même message
        self.known_positions = {}  # {drone_id: (position, timestamp)}
        self.formation_position = None
        
        # Paramètres de formation
        self.formation_grid_indices = None  # [x, y] indices dans la grille logique
        self.formation_spacing = 20.0  # Espacement entre drones dans la formation
        self.formation_grid = {}  # {drone_id: [x, y]} - positions logiques des drones connues
        self.formation_center = None  # Centre estimé de la formation
        self.formation_updated_time = 0  # Dernière mise à jour de formation
        
        # Leader election
        self.leader_last_seen = 0
        self.leader_timeout = 5.0  # secondes
        self.election_in_progress = False
        self.election_votes = {}
        # Priorité inversement proportionnelle à l'ID (plus petit ID = plus haute priorité)
        self.my_election_priority = random.random() * (100 - self.id)
        
        # État interne
        self.failure_probability = 0.00001
        self.failing_countdown = 20
        self.heartbeat_interval = 0.5  # Intervalle d'envoi des battements de cœur
        self.last_heartbeat = 0
        self.formation_update_interval = 1.0  # Intervalle pour envoyer des mises à jour de formation
        self.last_formation_update = 0
        
        # Système de waypoints
        self.waypoints = waypoints if waypoints is not None else [
            [100, 100, 60],   # Point 1
            [200, 50, 70],    # Point 2
            [50, 200, 55],    # Point 3
            [150, 150, 65]    # Point 4
        ]
        self.current_waypoint_index = 0
        self.waypoint_status = WaypointStatus.NAVIGATING
        self.waypoint_reached_time = 0
        self.waypoint_pause_duration = 0.0  # Pause de 2 secondes entre les waypoints
        self.waypoint_threshold = 10.0  # Distance pour considérer un waypoint atteint

    def get_current_target(self):
        """Obtenir le waypoint courant"""
        return np.array(self.waypoints[self.current_waypoint_index])
    
    def update_waypoint_progress(self):
        """Mettre à jour la progression des waypoints"""
        current_time = time.time()
        
        # Si le leader est proche du waypoint
        if self.role == DroneRole.LEADER:
            distance_to_waypoint = np.linalg.norm(self.position - self.get_current_target())
            
            if distance_to_waypoint <= self.waypoint_threshold:
                # Marquer comme atteint si pas déjà marqué
                if self.waypoint_status == WaypointStatus.NAVIGATING:
                    self.waypoint_status = WaypointStatus.REACHED
                    self.waypoint_reached_time = current_time
                
                # Passer au prochain waypoint après la pause
                if (current_time - self.waypoint_reached_time) >= self.waypoint_pause_duration:
                    self.current_waypoint_index = (self.current_waypoint_index + 1) % len(self.waypoints)
                    self.waypoint_status = WaypointStatus.NAVIGATING
                    print(f"Drone {self.id} (Leader) passe au waypoint {self.current_waypoint_index}")

    def distance_to(self, other_position):
        """Calcule la distance entre ce drone et une position"""
        try:
            if np.isnan(self.position).any() or np.isnan(other_position).any():
                return float('inf')
            return np.linalg.norm(self.position - other_position)
        except Exception as e:
            print(f"Erreur de calcul de distance pour le drone {self.id}: {e}")
            return float('inf')

    def can_communicate_with(self, other_drone):
        """Vérifie si le drone peut communiquer avec un autre drone"""
        if other_drone.id == self.id or other_drone.status == DroneStatus.FAILED:
            return False
        return self.distance_to(other_drone.position) <= self.communication_range

    def send_message(self, msg_type, data=None):
        """Crée un message à envoyer aux voisins"""
        message = Message(self.id, msg_type, data)
        return message

    def broadcast_position(self):
        """Diffuse la position du drone à ses voisins"""
        return self.send_message("position", {
            "position": self.position.tolist(),
            "velocity": self.velocity.tolist(),
            "status": self.status.name,
            "role": self.role.name,
            "grid_indices": self.formation_grid_indices
        })
    
    def broadcast_leader_heartbeat(self):
        """Si leader, diffuse un battement de cœur"""
        if self.role != DroneRole.LEADER:
            return None
            
        return self.send_message("leader_heartbeat", {
            "leader_id": self.id,
            "timestamp": time.time()
        })
    
    def broadcast_formation_update(self):
        """Si leader, diffuse une mise à jour de la formation (attribution des indices de grille)"""
        if self.role != DroneRole.LEADER:
            return None
        
        # Collecter les drones connus actifs
        current_time = time.time()
        active_drones = []
        
        for drone_id, (position, timestamp) in self.known_positions.items():
            if current_time - timestamp < 3.0:  # Positions récentes seulement
                active_drones.append((drone_id, position))
        
        # Ajouter ma position
        active_drones.append((self.id, self.position))
        
        # Trier par ID pour que tous les drones aient la même vue
        sorted_drones = sorted(active_drones, key=lambda x: x[0])
        
        # Calculer la taille de la grille
        grid_size = math.ceil(math.sqrt(len(sorted_drones)))
        
        # Générer les positions de grille
        grid_positions = {}
        for idx, (drone_id, _) in enumerate(sorted_drones):
            row = idx // grid_size
            col = idx % grid_size
            grid_positions[drone_id] = [col, row]
        
        # Stocker ma position dans la grille
        self.formation_grid_indices = grid_positions.get(self.id)
        
        # Stocker toute la grille
        self.formation_grid = grid_positions
        
        # Mettre à jour le temps
        self.last_formation_update = current_time
        
        # Envoyer la mise à jour de formation à tous
        return self.send_message("formation_update", {
            "grid_positions": grid_positions,
            "spacing": self.formation_spacing
        })
    
    def start_leader_election(self):
        """Démarre une élection de leader"""
        if self.election_in_progress:
            return None
            
        self.election_in_progress = True
        print(f"Drone {self.id}: Démarrage d'élection de leader")
        
        # Envoyer une candidature avec ma priorité
        return self.send_message("leader_election", {
            "candidate_id": self.id,
            "priority": self.my_election_priority
        })
    
    def vote_for_leader(self, candidate_id, priority):
        """Vote pour un candidat leader"""
        return self.send_message("leader_vote", {
            "candidate_id": candidate_id,
            "priority": priority
        })
    
    def declare_as_leader(self):
        """Se déclare comme nouveau leader après élection"""
        self.role = DroneRole.LEADER
        self.leader_id = self.id
        self.color = 'green'
        print(f"Drone {self.id}: Je suis le nouveau leader!")

        self.election_in_progress = False
        self.election_votes = {}

        self.leader_last_seen = time.time()
        
        # Envoyer un message de résultat d'élection
        return self.send_message("leader_elected", {
            "leader_id": self.id,
            "election_timestamp": time.time()
        })
    
    def process_received_messages(self, all_drones):
        """Traite tous les messages reçus"""
        for message in self.received_messages:
            # Éviter de traiter plusieurs fois le même message
            msg_id = f"{message.sender_id}_{message.msg_type}_{message.timestamp}"
            if msg_id in self.message_cache:
                continue
                
            self.message_cache[msg_id] = True
            
            # Traiter selon le type de message
            if message.msg_type == "position":
                self.handle_position_message(message)
            elif message.msg_type == "leader_heartbeat":
                self.handle_leader_heartbeat(message)
            elif message.msg_type == "formation_update":
                self.handle_formation_update(message)
            elif message.msg_type == "leader_election":
                self.handle_leader_election(message, all_drones)
            elif message.msg_type == "leader_vote":
                self.handle_leader_vote(message)
            elif message.msg_type == "leader_elected":
                self.handle_leader_elected(message, all_drones)
            
            # Relayer le message aux voisins si TTL > 0
            if message.ttl > 0:
                message.ttl -= 1
                self.relay_message(message, all_drones)
                
        # Vider la liste des messages
        self.received_messages = []
        
        # Nettoyer le cache des messages (garder seulement les 100 derniers)
        if len(self.message_cache) > 100:
            oldest_keys = sorted(self.message_cache.keys())[:50]
            for key in oldest_keys:
                del self.message_cache[key]
    
    def relay_message(self, message, all_drones):
        """Relaye un message aux voisins"""
        # Trouver tous les voisins directs
        for drone in all_drones:
            if drone.id != self.id and drone.id != message.sender_id and self.can_communicate_with(drone):
                # Dans un système réel, envoyer le message au drone
                # Dans notre simulation, l'ajouter à sa liste de messages
                drone.received_messages.append(message)
    
    def handle_position_message(self, message):
        """Traite un message de position"""
        sender_id = message.sender_id
        position_data = message.data
        
        # Vérifier le statut du drone émetteur
        sender_status = DroneStatus.ACTIVE  # Par défaut
        if "status" in position_data:
            sender_status_name = position_data["status"]
            try:
                sender_status = DroneStatus[sender_status_name]
            except KeyError:
                # Si le nom du statut n'est pas valide, on garde la valeur par défaut
                pass
        
        # Ne pas traiter les messages des drones en panne (FAILED)
        if sender_status == DroneStatus.FAILED:
            return
        
        # Mettre à jour la connaissance des positions
        if "position" in position_data:
            position = np.array(position_data["position"])
            self.known_positions[sender_id] = (position, message.timestamp)
            
            # Marquer si le drone est un leader
            if "role" in position_data and position_data["role"] == DroneRole.LEADER.name:
                self.leader_id = sender_id
                self.leader_last_seen = message.timestamp
            
            # Mise à jour des indices de grille connus
            if "grid_indices" in position_data and position_data["grid_indices"] is not None:
                if sender_id not in self.formation_grid:
                    self.formation_grid[sender_id] = position_data["grid_indices"]
            
        # Mettre à jour la liste des voisins
        self.neighbors[sender_id] = message.timestamp
    
    def handle_leader_heartbeat(self, message):
        """Traite un battement de cœur du leader"""
        leader_data = message.data
        leader_id = leader_data.get("leader_id")
        
        if leader_id is not None:
            # Mettre à jour le leader connu
            self.leader_id = leader_id
            self.leader_last_seen = message.timestamp
            
            # Fin d'élection si en cours
            if self.election_in_progress:
                self.election_in_progress = False
    
    def handle_formation_update(self, message):
        """Traite une mise à jour de formation envoyée par le leader"""
        formation_data = message.data
        grid_positions = formation_data.get("grid_positions", {})
        spacing = formation_data.get("spacing", self.formation_spacing)
        
        # Mettre à jour ma position dans la grille
        if self.id in grid_positions:
            self.formation_grid_indices = grid_positions[self.id]
            
        # Mettre à jour toutes les positions connues
        self.formation_grid = grid_positions
        self.formation_spacing = spacing
        
        # Marquer la formation comme mise à jour
        self.formation_updated_time = message.timestamp
        
    def handle_leader_election(self, message, all_drones):
        """Traite un message d'élection de leader"""
        election_data = message.data
        candidate_id = election_data.get("candidate_id")
        priority = election_data.get("priority", 0)
        
        # Marquer l'élection en cours
        self.election_in_progress = True
        
        # Si ma priorité est plus élevée, je propose ma candidature
        if self.my_election_priority > priority and self.status == DroneStatus.ACTIVE:
            # Je vote pour moi-même
            my_vote = self.vote_for_leader(self.id, self.my_election_priority)
            self.relay_message(my_vote, all_drones)
        else:
            # Je vote pour le candidat
            my_vote = self.vote_for_leader(candidate_id, priority)
            self.relay_message(my_vote, all_drones)
    
    def handle_leader_vote(self, message):
        """Traite un vote pour l'élection du leader"""
        vote_data = message.data
        candidate_id = vote_data.get("candidate_id")
        priority = vote_data.get("priority", 0)
        
        # Compter les votes
        if candidate_id not in self.election_votes:
            self.election_votes[candidate_id] = {"count": 0, "priority": priority}
        
        self.election_votes[candidate_id]["count"] += 1
        
        # Vérifier si j'ai reçu suffisamment de votes pour moi
        # Réduire le seuil pour faciliter l'élection avec peu de drones
        enough_votes = self.election_votes.get(self.id, {"count": 0})["count"] >= 2
        
        # Également, vérifier si je suis le drone actif avec la priorité la plus élevée
        highest_priority = True
        for cand_id, vote_info in self.election_votes.items():
            if (cand_id != self.id and 
                vote_info["priority"] > self.my_election_priority and 
                vote_info["count"] >= 2):
                highest_priority = False
                break
        
        # Si j'ai reçu assez de votes ou que je suis l'unique candidat valide après un délai
        if (enough_votes and highest_priority) or (
                time.time() - self.leader_last_seen > self.leader_timeout + 3.0 and 
                self.my_election_priority > 0 and 
                highest_priority):
            self.declare_as_leader()

    def detect_leaderless_state(self, all_drones):
        """Détecte si l'essaim est sans leader et démarre une élection si nécessaire"""
        # Ne s'applique qu'aux drones actifs
        if self.status != DroneStatus.ACTIVE:
            return
        
        current_time = time.time()
        
        # Vérifier si moi ou quelqu'un d'autre est actuellement leader
        leader_exists = False
        for drone in all_drones:
            if (drone.status == DroneStatus.ACTIVE and 
                drone.role == DroneRole.LEADER):
                leader_exists = True
                break
        
        # Si aucun leader n'existe et qu'il n'y a pas d'élection en cours
        if not leader_exists and not self.election_in_progress:
            # Attendre un délai aléatoire pour éviter que tous les drones démarrent une élection
            # en même temps (utiliser l'ID pour créer une séquence déterministe)
            delay = (self.id % 5) * 0.2  # Délai entre 0 et 0.8 seconde
            
            # Enregistrer un timer pour démarrer l'élection
            self.leader_last_seen = current_time - self.leader_timeout + delay
    
    def handle_leader_elected(self, message, all_drones):
        """Traite l'annonce d'un nouveau leader"""
        leader_data = message.data
        new_leader_id = leader_data.get("leader_id")
        election_timestamp = leader_data.get("election_timestamp", 0)  # AJOUT: récupérer le timestamp
        
        if new_leader_id is not None:
            # Mettre à jour le leader
            self.leader_id = new_leader_id
            self.leader_last_seen = message.timestamp
            
            # Si ce n'est pas moi, je suis un suiveur
            if new_leader_id != self.id:
                self.role = DroneRole.FOLLOWER
                self.color = 'blue'
            
            # AJOUT: Réinitialisation explicite et complète de l'état d'élection
            self.election_in_progress = False
            self.election_votes = {}
            
            # AJOUT: Forcer une mise à jour de formation après l'élection
            if self.role == DroneRole.LEADER:
                self.last_formation_update = 0  # Pour forcer une mise à jour immédiate
                
            # AJOUT: Propager l'annonce pour s'assurer que tous les drones sont informés
            if message.ttl <= 0:
                # Si le TTL est épuisé, créer un nouveau message avec TTL frais pour propagation
                new_msg = self.send_message("leader_elected", {
                    "leader_id": new_leader_id,
                    "election_timestamp": election_timestamp
                })
                if new_msg:
                    self.relay_message(new_msg, all_drones)  # Nécessite all_drones comme paramètre
    
    def check_leader_status(self, all_drones):
        """Vérifie si le leader est toujours actif, sinon lance une élection"""
        current_time = time.time()

        if self.election_in_progress and current_time - self.leader_last_seen > self.leader_timeout * 2:
            print(f"Drone {self.id}: Élection bloquée, réinitialisation forcée")
            self.election_in_progress = False
            self.election_votes = {}
        
        # Si je suis le leader, envoyer un battement de cœur périodiquement
        if self.role == DroneRole.LEADER:
            if current_time - self.last_heartbeat > self.heartbeat_interval:
                heartbeat = self.broadcast_leader_heartbeat()
                if heartbeat:
                    self.relay_message(heartbeat, all_drones)
                    self.last_heartbeat = current_time
            
            # Envoyer périodiquement des mises à jour de formation
            if current_time - self.last_formation_update > self.formation_update_interval:
                formation_update = self.broadcast_formation_update()
                if formation_update:
                    self.relay_message(formation_update, all_drones)
                    self.last_formation_update = current_time
            
            return
            
        # Si le leader n'a pas été vu depuis longtemps, démarrer une élection
        if (self.leader_id is not None and 
            current_time - self.leader_last_seen > self.leader_timeout and 
            not self.election_in_progress and
            self.status == DroneStatus.ACTIVE):
            
            print(f"Drone {self.id}: Le leader {self.leader_id} semble être hors ligne")
            
            # Vérifier si je peux encore le voir directement ou indirectement
            leader_visible = False
            for drone in all_drones:
                if drone.id == self.leader_id and drone.status != DroneStatus.FAILED:
                    if self.can_communicate_with(drone):
                        leader_visible = True
                        self.leader_last_seen = current_time
                        break
            
            if not leader_visible:
                # Démarrer une élection avec plus de persistance
                election_msg = self.start_leader_election()
                if election_msg:
                    # Diffuser le message plusieurs fois pour garantir la propagation
                    for _ in range(3):  # Répéter 3 fois pour plus de fiabilité
                        self.relay_message(election_msg, all_drones)
                        
                    # Voter pour moi-même immédiatement
                    vote_msg = self.vote_for_leader(self.id, self.my_election_priority)
                    if vote_msg:
                        self.relay_message(vote_msg, all_drones)

    def calculate_formation_position(self):
        """Calcule la position physique dans l'espace à partir des indices logiques de grille"""
        if self.formation_grid_indices is None:
            return None
        
        # Calculer le centre de formation à partir des drones ACTIFS connus
        current_time = time.time()
        positions = []
        for drone_id, (position, timestamp) in self.known_positions.items():
            if current_time - timestamp < 3.0:  # Données récentes
                # Vérifier si le drone est actif (pas défaillant ou en panne)
                # On vérifie dans les messages reçus récemment si le drone est marqué comme actif
                active = True
                for msg in self.received_messages:
                    if msg.msg_type == "position" and msg.sender_id == drone_id:
                        if "status" in msg.data and msg.data["status"] != DroneStatus.ACTIVE.name:
                            active = False
                            break
                
                if active:
                    positions.append(position)
        
        # Ajouter ma position si je suis actif
        if self.status == DroneStatus.ACTIVE:
            positions.append(self.position)
        
        if not positions:
            return None
        
        # Centre estimé de l'essaim actif
        swarm_center = np.mean(positions, axis=0)
        self.formation_center = swarm_center
        
        # Calculer le décalage par rapport au centre basé sur les indices de grille
        grid_x, grid_y = self.formation_grid_indices
        
        # Trouver la taille de la grille pour centrer correctement
        all_indices = list(self.formation_grid.values())
        if not all_indices:
            return None
            
        max_x = max(idx[0] for idx in all_indices)
        max_y = max(idx[1] for idx in all_indices)
        
        # Calculer le décalage par rapport au centre
        offset_x = (grid_x - max_x/2) * self.formation_spacing
        offset_y = (grid_y - max_y/2) * self.formation_spacing
        
        # Position finale
        formation_position = np.array([
            swarm_center[0] + offset_x,
            swarm_center[1] + offset_y,
            swarm_center[2]
        ])
        
        # Assurer une hauteur minimale
        min_height = 40.0
        if formation_position[2] < min_height:
            formation_position[2] = min_height
            
        return formation_position


    def calculate_emergent_v_formation(self):
        """Formation en V avec le leader toujours à la pointe"""
        current_time = time.time()
        
        # Collecter uniquement les voisins directs
        neighbors = []
        for drone_id, last_seen in self.neighbors.items():
            if current_time - last_seen < 2.0:
                if drone_id in self.known_positions:
                    pos, _ = self.known_positions[drone_id]
                    neighbors.append((drone_id, pos))
        
        if not neighbors:
            return None
        
        # Trouver le leader global ou local
        leader_id = self.leader_id
        leader_position = None
        leader_velocity = None
        
        # Déterminer le leader et sa position
        if leader_id is None or leader_id not in [n[0] for n in neighbors] + [self.id]:
            # Pas de leader global connu, utiliser le leader local (ID le plus bas)
            leader_id = min([n[0] for n in neighbors] + [self.id])
        
        # Si je suis le leader, je reste à ma position (toujours devant)
        if leader_id == self.id:
            return self.position
        
        # Trouver la position du leader
        for drone_id, pos in neighbors:
            if drone_id == leader_id:
                leader_position = pos
                break
        
        if leader_position is None:
            # Leader non trouvé parmi les voisins, essayer d'utiliser la dernière position connue
            if leader_id in self.known_positions:
                leader_position, _ = self.known_positions[leader_id]
            else:
                return None  # Impossible de déterminer la position du leader
        
        # Déterminer la direction de déplacement de la formation
        # Toujours utiliser la direction entre le waypoint actuel et le leader
        # pour garantir que la formation avance dans la bonne direction
        if self.waypoints:
            current_waypoint = np.array(self.waypoints[self.current_waypoint_index])
            formation_direction = self.normalize(current_waypoint - leader_position)
        else:
            # Si pas de waypoint connu, utiliser l'axe X comme direction par défaut
            formation_direction = np.array([1.0, 0.0, 0.0])
        
        # Déterminer ma position dans la formation en V
        # Trier les drones par ID pour une assignation stable
        all_ids = sorted([n[0] for n in neighbors] + [self.id])
        # Exclure le leader de la liste pour l'assignation des positions
        if leader_id in all_ids:
            all_ids.remove(leader_id)
        
        # Trouver mon rang dans la liste des suiveurs
        my_rank = all_ids.index(self.id) if self.id in all_ids else 0
        
        # Déterminer de quel côté du V je me place
        # Alternance équilibrée : les ID pairs à droite, impairs à gauche
        side = 1 if my_rank % 2 == 0 else -1
        
        # Calculer le vecteur perpendiculaire à la direction de déplacement
        perpendicular = self.normalize(np.cross(formation_direction, np.array([0, 0, 1])))
        
        # Calculer la distance derrière le leader
        distance_back = (my_rank // 2 + 1) * self.formation_spacing
        
        # Calculer la distance latérale (ouverture du V)
        distance_side = (my_rank // 2 + 1) * self.formation_spacing * 0.5 * side
        
        # Calculer ma position cible dans la formation
        formation_position = (
            leader_position - 
            formation_direction * distance_back + 
            perpendicular * distance_side
        )
        
        # Assurer une hauteur minimale
        if formation_position[2] < 40.0:
            formation_position[2] = 40.0
        
        return formation_position


    def calculate_steering_forces(self):
        """Calcule les forces de direction basées sur les règles de l'essaim"""
        # Initialiser les forces
        separation_force = np.zeros(3)
        alignment_force = np.zeros(3)
        cohesion_force = np.zeros(3)
        formation_force = np.zeros(3)
        
        # Poids des forces
        weights = {
            'separation': 1.5,
            'alignment': 0.8,
            'cohesion': 0.7,
            'formation': 3.0  # Augmenté pour donner plus d'importance à la formation
        }
        
        # Collecter les voisins connus récents ET ACTIFS
        neighbors_data = []
        current_time = time.time()
        
        for drone_id, (position, timestamp) in self.known_positions.items():
            if current_time - timestamp < 2.0:  # Données récentes seulement
                # Vérifier si le drone est actif (pas défaillant ou en panne)
                # On utilise les derniers messages reçus pour déterminer le statut
                is_active = True
                for msg in self.received_messages:
                    if msg.msg_type == "position" and msg.sender_id == drone_id:
                        if "status" in msg.data and msg.data["status"] != DroneStatus.ACTIVE.name:
                            is_active = False
                            break
                
                if not is_active:
                    continue  # Ignorer les drones non actifs
                    
                # Vérifier si dans la portée de communication
                distance = self.distance_to(position)
                if distance <= self.communication_range:
                    neighbors_data.append((drone_id, position, distance))
        
        if not neighbors_data:
            # Pas de voisins, retourner une petite force aléatoire
            return np.array([
                random.uniform(-0.1, 0.1),
                random.uniform(-0.1, 0.1),
                random.uniform(0, 0.1)  # Légère tendance vers le haut
            ])
        
        # 1. Force de séparation
        close_neighbors = [(neighbor_id, pos, dist) for neighbor_id, pos, dist in neighbors_data if dist < 15.0]
        if close_neighbors:
            for _, neighbor_pos, dist in close_neighbors:
                if dist > 0:  # Éviter division par zéro
                    # Vecteur de répulsion
                    repulsion = self.position - neighbor_pos
                    repulsion = repulsion / (dist * dist)  # Force inversement proportionnelle au carré de la distance
                    separation_force += repulsion
            
            separation_force = self.normalize(separation_force)
        
        # 2. Force d'alignement (basée sur les positions connues)
        # Nous n'avons pas de vélocité des voisins directement, approximation basée sur positions
        alignment_force = np.zeros(3)
        
        # 3. Force de cohésion
        if neighbors_data:
            center_of_mass = np.mean([pos for _, pos, _ in neighbors_data], axis=0)
            cohesion_force = self.normalize(center_of_mass - self.position)
        
        # 4. Force de formation
        self.formation_position = self.calculate_formation_position()
        if self.formation_position is not None:
            formation_force = self.normalize(self.formation_position - self.position)
            
            # Augmenter le poids si loin de la position de formation
            distance_to_formation = np.linalg.norm(self.position - self.formation_position)
            if distance_to_formation > 30.0:
                weights['formation'] = 4.0
            elif distance_to_formation > 15.0:
                weights['formation'] = 3.5
        
        # Si je suis le leader, je veux me diriger vers le waypoint courant
        if self.role == DroneRole.LEADER:
            current_target = self.get_current_target()
            target_force = self.normalize(current_target - self.position)
            
            # Calculer une force liée à la cohésion de l'essaim
            swarm_cohesion_force = np.zeros(3)
            followers_positions = []
            
            # Collecter les positions des suiveurs actifs
            for drone_id, (position, timestamp) in self.known_positions.items():
                if (current_time - timestamp < 2.0 and 
                    drone_id != self.id):
                    followers_positions.append(position)
            
            if followers_positions:
                # Calculer le centre de masse des suiveurs
                swarm_center = np.mean(followers_positions, axis=0)
                # Calculer la distance au centre des suiveurs
                distance_to_swarm = np.linalg.norm(self.position - swarm_center)
                
                # Si les suiveurs sont trop loin, créer une force qui ralentit le leader
                if distance_to_swarm > 40.0:  # Seuil à ajuster
                    swarm_cohesion_force = self.normalize(swarm_center - self.position)
                    # Ajuster la pondération en fonction de la distance
                    swarm_weight = min(5.0, distance_to_swarm / 8.0)
                else:
                    swarm_weight = 0.0
            else:
                swarm_weight = 0.0
            
            # Équilibrer entre objectif et cohésion
            target_weight = 3.0 if swarm_weight > 0 else 5.0
            
            steering_force = (
                target_force * target_weight +
                swarm_cohesion_force * swarm_weight +
                separation_force * (weights['separation'] * 0.5) +  # Augmenté pour éviter les collisions
                alignment_force * (weights['alignment'] * 0.1) +
                cohesion_force * (weights['cohesion'] * 0.1) +
                formation_force * (weights['formation'] * 0.5)
            )
        else:
            # Les suiveurs gardent leur comportement original
            steering_force = (
                separation_force * weights['separation'] +
                alignment_force * weights['alignment'] +
                cohesion_force * weights['cohesion'] +
                formation_force * weights['formation']
            )
        
        # Ajouter une petite force vers le haut pour éviter le sol
        if self.position[2] < 40.0:
            altitude_force = np.array([0, 0, 1.0]) * (40.0 - self.position[2]) / 10.0
            steering_force += altitude_force
        
        return steering_force
    
    def clean_known_positions(self):
        """Nettoie la liste des positions connues en supprimant les drones défaillants ou inactifs"""
        current_time = time.time()
        positions_to_remove = []
        
        # Identifier les drones à supprimer
        for drone_id, (_, timestamp) in self.known_positions.items():
            # Supprimer les positions trop anciennes
            if current_time - timestamp > 5.0:
                positions_to_remove.append(drone_id)
                continue
                
            # Vérifier dans les messages récents si le drone est en panne
            for msg in self.received_messages:
                if msg.msg_type == "position" and msg.sender_id == drone_id:
                    if "status" in msg.data and msg.data["status"] == DroneStatus.FAILED.name:
                        positions_to_remove.append(drone_id)
                        break
        
        # Supprimer les positions identifiées
        for drone_id in positions_to_remove:
            if drone_id in self.known_positions:
                del self.known_positions[drone_id]
            if drone_id in self.formation_grid:
                del self.formation_grid[drone_id]
            if drone_id in self.neighbors:
                del self.neighbors[drone_id]
        
        # Si le leader est parmi les drones supprimés, on lance une élection
        if self.leader_id in positions_to_remove:
            self.leader_id = None
    
    def normalize(self, v):
        """Normalise un vecteur de manière sécurisée"""
        # Vérifier les NaN
        if np.isnan(v).any():
            return np.zeros_like(v)
            
        # Calculer la norme en toute sécurité
        norm = np.linalg.norm(v)
        if norm > 1e-10:
            return v / norm
        else:
            return np.zeros_like(v)
    
    def update_status(self):
        """Met à jour le statut du drone (simulation de défaillances)"""
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

    def update(self, all_drones):
        """Mise à jour de l'état du drone"""
        try:
            # Mettre à jour le statut
            self.update_status()
            
            # Traiter les messages reçus
            self.process_received_messages(all_drones)
            
            # Nettoyer les positions connues périodiquement
            self.clean_known_positions()
            
            # Vérifier l'état du leader
            self.check_leader_status(all_drones)

            self.detect_leaderless_state(all_drones)
            
            # Mettre à jour la progression des waypoints
            self.update_waypoint_progress()
            
            # Diffuser ma position
            position_msg = self.broadcast_position()
            self.relay_message(position_msg, all_drones)
            
            # En cas de panne, limiter le mouvement
            if self.status == DroneStatus.FAILED:
                # Drone en panne: chute lente
                self.acceleration = np.array([0, 0, -0.05])
                self.velocity = self.velocity * 0.95 + self.acceleration
                self.position += self.velocity
                
                # Limiter la vitesse
                max_speed = 0.5
                speed = np.linalg.norm(self.velocity)
                if speed > max_speed:
                    self.velocity = self.velocity * (max_speed / speed)
                
                return
            
            # Calculer les forces de direction pour les drones actifs
            steering_force = self.calculate_steering_forces()
            
            # Appliquer l'accélération
            self.acceleration = steering_force
            
            # Limiter l'accélération
            max_acceleration = 0.1
            acc_magnitude = np.linalg.norm(self.acceleration)
            if acc_magnitude > max_acceleration:
                self.acceleration = self.acceleration * (max_acceleration / acc_magnitude)
            
            # Amortissement pour éviter les oscillations
            damping = 0.92
            self.velocity = self.velocity * damping + self.acceleration
            
            # Limiter la vitesse
            max_speed = 1.5 if self.status == DroneStatus.ACTIVE else 0.7
            speed = np.linalg.norm(self.velocity)
            if speed > max_speed:
                self.velocity = self.velocity * (max_speed / speed)
            
            # Mettre à jour la position
            self.position += self.velocity
            
            # Garder les drones dans des limites raisonnables
            bounds = {
                'min_x': -100, 'max_x': 300,
                'min_y': -100, 'max_y': 300,
                'min_z': 20, 'max_z': 150
            }
            
            # Appliquer des contraintes de limites
            for i, (min_val, max_val) in enumerate([
                (bounds['min_x'], bounds['max_x']),
                (bounds['min_y'], bounds['max_y']),
                (bounds['min_z'], bounds['max_z'])
            ]):
                if self.position[i] < min_val:
                    self.position[i] = min_val
                    self.velocity[i] *= -0.5  # Rebond avec perte d'énergie
                elif self.position[i] > max_val:
                    self.position[i] = max_val
                    self.velocity[i] *= -0.5  # Rebond avec perte d'énergie
        except Exception as e:
            print(f"Erreur de mise à jour du drone {self.id}: {e}")
            # Réinitialiser le drone à un état stable
            if self.status != DroneStatus.FAILED:
                self.position = np.array([
                    random.uniform(0, 50),
                    random.uniform(0, 50),
                    random.uniform(50, 60)
                ])
                self.velocity = np.array([0.0, 0.0, 0.0])
                self.acceleration = np.array([0.0, 0.0, 0.1])

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
        self.fig = None
        self.ax = None
        self.reset_button_ax = None
        self.reset_button = None
        self.add_drone_button_ax = None
        self.add_drone_button = None
        self.fail_leader_button_ax = None
        self.fail_leader_button = None
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
    
    def configure_axes(self):
        """Configure les axes du graphique"""
        self.ax.set_xlim([-50, 250])
        self.ax.set_ylim([-50, 250])
        self.ax.set_zlim([0, 150])
        self.ax.set_xlabel('X')
        self.ax.set_ylabel('Y')
        self.ax.set_zlabel('Z')
        self.ax.set_title('Simulation d\'Essaim de Drones Décentralisée avec Formation Coordonnée')
    
    def reset_simulation(self, event):
        """Réinitialise la simulation"""
        self.drones = []
        self.start_time = time.time()
        self.initialize_swarm()
        print("Simulation réinitialisée avec de nouveaux drones.")
    
    def add_drone(self, event):
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
    
    def fail_leader(self, event):
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
    
    
    def visualize(self):
        """Affiche l'état actuel de la simulation"""
        try:
            self.ax.clear()
            self.configure_axes()
            
            # Dessiner la grille de formation si un leader existe
            leader = None
            grid_size = 0
            for drone in self.drones:
                if drone.role == DroneRole.LEADER and drone.status == DroneStatus.ACTIVE:
                    leader = drone
                    break
            
            if leader and leader.formation_center is not None:
                # Trouver la taille maximale de la grille
                max_x = 0
                max_y = 0
                for grid_indices in leader.formation_grid.values():
                    if grid_indices:
                        max_x = max(max_x, grid_indices[0])
                        max_y = max(max_y, grid_indices[1])
                
                # Calculer la taille de la grille
                grid_size = max(max_x, max_y) + 1
                
                # Dessiner une grille légère pour visualiser la formation
                center = leader.formation_center
                spacing = leader.formation_spacing
                
                # Points de la grille - plus léger que des lignes complètes
                for i in range(grid_size):
                    for j in range(grid_size):
                        # Calculer la position dans l'espace
                        x = center[0] + (i - max_x/2) * spacing
                        y = center[1] + (j - max_y/2) * spacing
                        z = center[2]
                        
                        # Dessiner un petit point pour marquer la position de la grille
                        self.ax.scatter(x, y, z, color='lightgray', s=5, alpha=0.3)
            
            # Afficher les drones
            for drone in self.drones:
                # Ignorer si la position contient NaN
                if np.isnan(drone.position).any() or np.isinf(drone.position).any():
                    continue
                
                # Marqueur de position du drone
                marker_size = 30 if drone.role == DroneRole.LEADER else 20
                marker_style = 'o' if drone.status == DroneStatus.ACTIVE else 'x'
                
                self.ax.scatter(
                    drone.position[0],
                    drone.position[1],
                    drone.position[2],
                    color=drone.color,
                    s=marker_size,
                    marker=marker_style,
                    alpha=0.8
                )
                
                # Afficher l'ID du drone
                self.ax.text(
                    drone.position[0], 
                    drone.position[1], 
                    drone.position[2] + 5, 
                    str(drone.id),
                    color='black',
                    fontsize=8
                )
                
                # Vecteur de vitesse
                if drone.status != DroneStatus.FAILED:
                    velocity_norm = np.linalg.norm(drone.velocity)
                    if velocity_norm > 0.01:  # Afficher uniquement pour les mouvements significatifs
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
                            normalize=True,
                            alpha=0.6
                        )
                
                # Indicateur de position dans la formation
                if (drone.formation_position is not None and 
                    drone.status == DroneStatus.ACTIVE and 
                    not np.isnan(drone.formation_position).any() and 
                    not np.isinf(drone.formation_position).any()):
                    
                    # Calculer l'erreur de formation pour ce drone
                    error = np.linalg.norm(drone.position - drone.formation_position)
                    # Gradient de couleur du rouge (mauvais) au vert (bon)
                    error_normalized = min(1.0, error / 30.0)  # Plafonner à 1.0
                    formation_color = (
                        error_normalized,  # Rouge: plus pour une erreur plus élevée
                        1 - error_normalized,  # Vert: plus pour une erreur plus faible
                        0.5  # Composante bleue fixe
                    )
                    
                    # Ligne reliant le drone à la position de formation
                    self.ax.plot(
                        [drone.position[0], drone.formation_position[0]],
                        [drone.position[1], drone.formation_position[1]],
                        [drone.position[2], drone.formation_position[2]],
                        color=formation_color,
                        linestyle=':',
                        alpha=0.3
                    )
                    
                    # Petit marqueur pour la position de formation
                    self.ax.scatter(
                        drone.formation_position[0],
                        drone.formation_position[1],
                        drone.formation_position[2],
                        color=formation_color,
                        s=10,
                        marker='.',
                        alpha=0.5
                    )
                    
                    # Afficher l'indice de grille si disponible
                    if drone.formation_grid_indices is not None:
                        grid_x, grid_y = drone.formation_grid_indices
                        grid_text = f"[{grid_x},{grid_y}]"
                        self.ax.text(
                            drone.formation_position[0],
                            drone.formation_position[1],
                            drone.formation_position[2] + 5,
                            grid_text,
                            color='darkgray',
                            fontsize=7
                        )
            
            # Afficher les connexions du réseau
            for drone_id, neighbors in self.network_graph.items():
                # Obtenir la position du drone
                drone = next((d for d in self.drones if d.id == drone_id), None)
                if drone is None or drone.status == DroneStatus.FAILED:
                    continue
                
                for neighbor_id in neighbors:
                    neighbor = next((d for d in self.drones if d.id == neighbor_id), None)
                    if neighbor is None or neighbor.status == DroneStatus.FAILED:
                        continue
                    
                    # Dessiner une ligne fine pour la connexion
                    self.ax.plot(
                        [drone.position[0], neighbor.position[0]],
                        [drone.position[1], neighbor.position[1]],
                        [drone.position[2], neighbor.position[2]],
                        color='gray',
                        linestyle='-',
                        linewidth=0.5,
                        alpha=0.2
                    )
            
            # Afficher les informations de simulation
            elapsed_time = time.time() - self.start_time
            
            # Compter les drones par statut
            active_count = sum(1 for d in self.drones if d.status == DroneStatus.ACTIVE)
            failing_count = sum(1 for d in self.drones if d.status == DroneStatus.FAILING)
            failed_count = sum(1 for d in self.drones if d.status == DroneStatus.FAILED)
            
            # Trouver le leader actuel
            leader_id = None
            for drone in self.drones:
                if drone.role == DroneRole.LEADER and drone.status == DroneStatus.ACTIVE:
                    leader_id = drone.id
                    break
            
            # Calculer la qualité de formation
            formation_quality = 0.0
            valid_drones = 0
            
            for drone in self.drones:
                if (drone.status == DroneStatus.ACTIVE and 
                    drone.formation_position is not None and 
                    not np.isnan(drone.formation_position).any()):
                    
                    error = np.linalg.norm(drone.position - drone.formation_position)
                    formation_quality += error
                    valid_drones += 1
            
            if valid_drones > 0:
                avg_formation_error = formation_quality / valid_drones
                formation_quality = max(0, 1 - (avg_formation_error / 30))
                quality_status = "Excellente" if formation_quality > 0.9 else "Bonne" if formation_quality > 0.7 else "Moyenne" if formation_quality > 0.5 else "Faible"
            else:
                formation_quality = 0
                quality_status = "Indéterminée"
            
            # Préparer les informations de waypoint
            if leader and leader.waypoints:
                current_waypoint = leader.waypoints[leader.current_waypoint_index]
                waypoint_info = f"Waypoint: {leader.current_waypoint_index} {current_waypoint}"
            else:
                waypoint_info = "Pas de waypoint"
            
            # Afficher les informations
            info_text = [
                f"Temps: {elapsed_time:.1f}s",
                f"Formation: Carrée",
                f"Leader actuel: {leader_id if leader_id is not None else 'Aucun'}",
                waypoint_info,
                f"Taille de grille: {grid_size}x{grid_size}",
                f"Drones actifs: {active_count}",
                f"Drones en panne: {failing_count}",
                f"Drones en échec: {failed_count}",
                f"Qualité de formation: {formation_quality:.2f} ({quality_status})"
            ]
            
            # Ajouter une légende de couleurs
            info_text.extend([
                "",
                "Bleu: Drone Suiveur",
                "Vert: Drone Leader",
                "Orange: Drone en Panne",
                "Noir: Drone en Échec",
                "Vert → Rouge: Bonne → Mauvaise Formation",
                "Gris: Communications",
                "X Rouge: Waypoints"
            ])
            
            # Afficher chaque ligne d'information
            for i, text in enumerate(info_text):
                y_pos = 0.95 - i * 0.03
                self.ax.text2D(1.05, y_pos, text, transform=self.ax.transAxes)
            
            # Afficher les waypoints
            if leader and leader.waypoints:
                for i, waypoint in enumerate(leader.waypoints):
                    color = 'red'
                    marker = 'x'
                    
                    self.ax.scatter(
                        waypoint[0],
                        waypoint[1],
                        waypoint[2],
                        color=color,
                        marker=marker,
                        s=100
                    )
        
        except Exception as e:
            print(f"Erreur dans la visualisation: {e}")

# Nouvelle classe pour la visualisation avec PyQtGraph
class PyQtGraphSwarmVisualization:
    def __init__(self, swarm_simulation, trail_length=20):
        self.swarm_simulation = swarm_simulation
        self.trail_length = trail_length
        
        # Configuration de l'application PyQt
        self.app = QtWidgets.QApplication([])
        
        # Création de la fenêtre principale
        self.window = QtWidgets.QMainWindow()
        self.window.setWindowTitle('AirPhalanx')
        self.window.resize(1200, 800)
        
        # Widget central qui contiendra la vue 3D et les contrôles
        self.central_widget = QtWidgets.QWidget()
        self.window.setCentralWidget(self.central_widget)
        
        # Layout principal
        self.main_layout = QtWidgets.QHBoxLayout()
        self.central_widget.setLayout(self.main_layout)
        
        # Zone de visualisation 3D
        self.view_3d = gl.GLViewWidget()
        self.main_layout.addWidget(self.view_3d, 3)  # Prend 3/4 de la largeur
        
        # Panneau de contrôle et de métriques à droite
        self.control_panel = QtWidgets.QVBoxLayout()
        self.main_layout.addLayout(self.control_panel, 1)  # Prend 1/4 de la largeur
        
        # Graphiques de métriques
        self.setup_metric_plots()
        
        # Données visuelles
        self.drone_markers = {}
        self.drone_trails = {}
        self.formation_lines = []
        self.communication_links = []
        self.drone_position_history = {}
        self.terrain = None
        self.waypoints = []
        self.landscape_objects = []
        
        # Paramètres d'affichage - DÉPLACÉ ICI AVANT setup_controls()
        self.show_trails = True
        self.show_formation = True
        self.show_communication = True
        self.show_terrain = True
        
        # Boutons de contrôle
        self.setup_controls()
        
        # Timer pour les mises à jour
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.update_simulation)
        self.timer.start(33)  # ~30 FPS
        
        # Initialiser les affichages
        self.setup_visualization()
        
    def setup_metric_plots(self):
        """Configure les graphiques de métriques"""
        # Graphique de qualité de formation
        self.quality_plot = pg.PlotWidget(title="Qualité de formation (%)")
        self.quality_plot.setYRange(0, 100)
        self.quality_plot.setBackground((240, 240, 240))
        self.quality_curve = self.quality_plot.plot(pen=(0, 200, 0))
        self.quality_data = np.zeros(100)
        self.control_panel.addWidget(self.quality_plot)
        
        # Graphique d'état des drones
        self.drone_status_plot = pg.PlotWidget(title="État des drones")
        self.drone_status_plot.setBackground((240, 240, 240))
        self.active_curve = self.drone_status_plot.plot(pen=(0, 0, 255), name="Actifs")
        self.failing_curve = self.drone_status_plot.plot(pen=(255, 165, 0), name="Défaillants")
        self.failed_curve = self.drone_status_plot.plot(pen=(255, 0, 0), name="En panne")
        self.drone_status_plot.addLegend()
        self.active_data = np.zeros(100)
        self.failing_data = np.zeros(100)
        self.failed_data = np.zeros(100)
        self.control_panel.addWidget(self.drone_status_plot)
        
        # Graphique de connectivité réseau
        self.connectivity_plot = pg.PlotWidget(title="Connectivité du réseau (%)")
        self.connectivity_plot.setYRange(0, 100)
        self.connectivity_plot.setBackground((240, 240, 240))
        self.connectivity_curve = self.connectivity_plot.plot(pen=(0, 100, 200))
        self.connectivity_data = np.zeros(100)
        self.control_panel.addWidget(self.connectivity_plot)
    
    def setup_controls(self):
        """Configure les boutons de contrôle"""
        # Groupe de boutons
        button_layout = QtWidgets.QHBoxLayout()
        
        # Bouton de réinitialisation
        reset_button = QtWidgets.QPushButton("Réinitialiser")
        reset_button.clicked.connect(self.swarm_simulation.reset_simulation)
        button_layout.addWidget(reset_button)
        
        # Bouton pour ajouter un drone
        add_drone_button = QtWidgets.QPushButton("+ Drone")
        add_drone_button.clicked.connect(self.swarm_simulation.add_drone)
        button_layout.addWidget(add_drone_button)
        
        # Bouton pour faire échouer le leader
        fail_leader_button = QtWidgets.QPushButton("Échec Leader")
        fail_leader_button.clicked.connect(self.swarm_simulation.fail_leader)
        button_layout.addWidget(fail_leader_button)
        
        self.control_panel.addLayout(button_layout)
        
        # Cases à cocher pour les options d'affichage
        display_options_group = QtWidgets.QGroupBox("Options d'affichage")
        display_options_layout = QtWidgets.QVBoxLayout()
        
        # Case à cocher pour les traînées
        self.trails_checkbox = QtWidgets.QCheckBox("Traînées")
        self.trails_checkbox.setChecked(self.show_trails)
        self.trails_checkbox.stateChanged.connect(self.toggle_trails)
        display_options_layout.addWidget(self.trails_checkbox)
        
        # Case à cocher pour la formation
        self.formation_checkbox = QtWidgets.QCheckBox("Formation")
        self.formation_checkbox.setChecked(self.show_formation)
        self.formation_checkbox.stateChanged.connect(self.toggle_formation)
        display_options_layout.addWidget(self.formation_checkbox)
        
        # Case à cocher pour les communications
        self.communication_checkbox = QtWidgets.QCheckBox("Communications")
        self.communication_checkbox.setChecked(self.show_communication)
        self.communication_checkbox.stateChanged.connect(self.toggle_communication)
        display_options_layout.addWidget(self.communication_checkbox)
        
        # Case à cocher pour le terrain
        self.terrain_checkbox = QtWidgets.QCheckBox("Terrain")
        self.terrain_checkbox.setChecked(self.show_terrain)
        self.terrain_checkbox.stateChanged.connect(self.toggle_terrain)
        display_options_layout.addWidget(self.terrain_checkbox)
        
        display_options_group.setLayout(display_options_layout)
        self.control_panel.addWidget(display_options_group)
        
        # Zone d'informations
        info_group = QtWidgets.QGroupBox("Informations")
        info_layout = QtWidgets.QVBoxLayout()
        self.info_label = QtWidgets.QLabel("Drones: 0\nLeader: aucun")
        info_layout.addWidget(self.info_label)
        info_group.setLayout(info_layout)
        self.control_panel.addWidget(info_group)
    
    def toggle_trails(self, state):
        self.show_trails = state == QtCore.Qt.Checked
        self.update_visualization()
    
    def toggle_formation(self, state):
        self.show_formation = state == QtCore.Qt.Checked
        self.update_visualization()
    
    def toggle_communication(self, state):
        self.show_communication = state == QtCore.Qt.Checked
        self.update_visualization()
    
    def toggle_terrain(self, state):
        self.show_terrain = state == QtCore.Qt.Checked
        if self.terrain:
            self.terrain.setVisible(self.show_terrain)
        
        # Modifier également la visibilité des objets de paysage
        if hasattr(self, 'landscape_objects'):
            for obj in self.landscape_objects:
                obj.setVisible(self.show_terrain)
    
    def setup_visualization(self):
        """Configure l'environnement 3D"""
        # Configuration de la vue 3D
        self.view_3d.setCameraPosition(distance=400, elevation=30, azimuth=45)
        self.view_3d.setBackgroundColor((0, 0, 0))  # Couleur de fond bleu clair
        
        # Créer le terrain si activé
        if self.show_terrain:
            self.create_terrain()
        
        # Créer les waypoints
        self.create_waypoints()
        
        # Initialiser les visualisations des drones
        for drone in self.swarm_simulation.drones:
            self.create_drone_visual(drone)


    # Voici comment modifier la méthode create_terrain de la classe PyQtGraphSwarmVisualization
    # pour y ajouter des sapins et des montagnes:

    def create_terrain(self):
        """Crée un terrain 3D réaliste sans sapins ni montagnes"""
        # Définir la grille du terrain avec une résolution plus élevée
        x = np.linspace(-100, 300, 80)  # Doublé la résolution
        y = np.linspace(-100, 300, 80)
        x_mesh, y_mesh = np.meshgrid(x, y)
        z_mesh = np.zeros_like(x_mesh)
        
        # Générer un terrain avec des ondulations plus complexes
        # Grandes ondulations (montagnes et vallées)
        z_mesh += 12 * np.sin(x_mesh/250) * np.cos(y_mesh/200)
        
        # Moyennes ondulations (collines)
        z_mesh += 6 * np.sin(x_mesh/80 + 1.5) * np.cos(y_mesh/70 + 0.5)
        z_mesh += 4 * np.sin(x_mesh/50 - 0.8) * np.cos(y_mesh/60 + 1.2)
        
        # Petites ondulations (terrain accidenté)
        z_mesh += 2 * np.sin(x_mesh/20 + 3.0) * np.cos(y_mesh/25 + 2.0)
        
        # Bruit de terrain (micro-détails)
        noise = np.random.random(z_mesh.shape) * 1.2
        smoothed_noise = np.zeros_like(noise)
        
        # Lissage simple du bruit pour un aspect plus naturel
        for i in range(1, noise.shape[0]-1):
            for j in range(1, noise.shape[1]-1):
                smoothed_noise[i, j] = np.mean(noise[i-1:i+2, j-1:j+2])
        
        z_mesh += smoothed_noise
        
        # Rivière ou cours d'eau serpentant
        river_x = 100 + 80 * np.sin(y_mesh / 100)
        river_mask = np.exp(-0.01 * (x_mesh - river_x)**2)
        river_depth = 8.0 * river_mask
        z_mesh -= river_depth
        
        # S'assurer que les zones d'eau sont lisses
        water_level = 2.0
        water_mask = z_mesh < water_level
        z_mesh[water_mask] = water_level - 0.2 * smoothed_noise[water_mask]
        
        # Créer un objet de surface pour le terrain
        terrain_mesh = gl.GLSurfacePlotItem(x=x, y=y, z=z_mesh, shader='shaded')
        
        # Définir la couleur du terrain avec plus de variation
        colors = np.zeros((z_mesh.shape[0], z_mesh.shape[1], 4), dtype=np.float32)
        
        for i in range(z_mesh.shape[0]):
            for j in range(z_mesh.shape[1]):
                height = z_mesh[i, j]
                
                # Calcul de la pente pour déterminer le type de terrain
                if i > 0 and i < z_mesh.shape[0]-1 and j > 0 and j < z_mesh.shape[1]-1:
                    dx = (z_mesh[i, j+1] - z_mesh[i, j-1]) / (x[j+1] - x[j-1])
                    dy = (z_mesh[i+1, j] - z_mesh[i-1, j]) / (y[i+1] - y[i-1])
                    slope = np.sqrt(dx*dx + dy*dy)
                else:
                    slope = 0
                
                # Eau profonde et peu profonde
                if height < water_level - 0.5:
                    # Eau profonde (bleu foncé)
                    colors[i, j] = [0.0, 0.2, 0.5, 1.0]
                elif height < water_level:
                    # Eau peu profonde (bleu clair)
                    colors[i, j] = [0.2, 0.5, 0.8, 1.0]
                # Plage/berge
                elif height < water_level + 0.7:
                    # Sable/limon
                    colors[i, j] = [0.8, 0.7, 0.5, 1.0]
                # Plaines et prairies
                elif height < 6.0:
                    if slope > 0.3:
                        # Sol nu/rocheux sur les pentes
                        colors[i, j] = [0.6, 0.5, 0.4, 1.0]
                    else:
                        # Variation de vert pour prairies
                        green_var = 0.5 + 0.3 * smoothed_noise[i, j] / 1.5
                        colors[i, j] = [0.2, green_var, 0.2, 1.0]
                # Collines et pentes
                elif height < 12.0:
                    if slope > 0.5:
                        # Rochers sur les pentes abruptes
                        rock_color = 0.4 + 0.2 * smoothed_noise[i, j] / 1.5
                        colors[i, j] = [rock_color, rock_color * 0.9, rock_color * 0.8, 1.0]
                    else:
                        # Mélange d'herbe et de roche
                        mix = (height - 6.0) / 6.0  # 0 à 1
                        green = 0.5 - 0.3 * mix
                        colors[i, j] = [0.3 + 0.3 * mix, green, 0.2 * (1 - mix), 1.0]
                # Haute montagne
                else:
                    snow_line = 15.0 + 2.0 * np.sin(x_mesh[i, j] / 30 + y_mesh[i, j] / 30)
                    if height > snow_line:
                        # Neige avec variations
                        snow_intensity = min(1.0, 0.8 + (height - snow_line) / 10.0)
                        snow_color = 0.8 + 0.2 * smoothed_noise[i, j] / 1.5
                        colors[i, j] = [snow_color, snow_color, snow_color, 1.0]
                    else:
                        # Roche de montagne
                        rock_color = 0.3 + 0.2 * smoothed_noise[i, j] / 1.5
                        colors[i, j] = [rock_color, rock_color * 0.85, rock_color * 0.7, 1.0]
        
        terrain_mesh.setData(colors=colors)
        self.view_3d.addItem(terrain_mesh)
        self.terrain = terrain_mesh
        
        # Initialiser la liste des objets de paysage comme vide
        self.landscape_objects = []
        
        print("Terrain réaliste créé sans sapins ni montagnes")
    
    def create_waypoints(self):
        """Crée les visualisations des waypoints"""
        if not self.swarm_simulation.drones:
            return
        
        waypoints = self.swarm_simulation.drones[0].waypoints
        if not waypoints:
            return
        
        # Créer un point pour chaque waypoint
        for i, wp in enumerate(waypoints):
            # Point du waypoint (étoile rouge)
            waypoint_item = gl.GLScatterPlotItem(
                pos=np.array([wp]),
                color=(1, 0, 0, 1),
                size=5,
                pxMode=True
            )
            self.view_3d.addItem(waypoint_item)
            self.waypoints.append(waypoint_item)
            
            # Texte du waypoint
            #text = gl.GLTextItem(pos=np.array(wp) + np.array([0, 0, 5]), text=f'WP {i}')
            #self.view_3d.addItem(text)
            #self.waypoints.append(text)
        
        # Ligne connectant les waypoints
        path_points = np.array(waypoints + [waypoints[0]])  # Fermer la boucle
        path_line = gl.GLLinePlotItem(
            pos=path_points,
            color=(1, 0, 0, 0.5),
            width=2,
            antialias=True
        )
        #self.view_3d.addItem(path_line)
        #self.waypoints.append(path_line)
    
    def create_drone_visual(self, drone):
        """Crée une représentation visuelle pour un drone"""
        # Déterminer la couleur en fonction du statut et du rôle
        color = self.get_drone_color(drone)
        
        # Créer un marqueur pour le drone
        marker = gl.GLScatterPlotItem(
            pos=np.array([drone.position]),
            color=color,
            size=15 if drone.role == DroneRole.LEADER else 10,
            pxMode=True
        )
        self.view_3d.addItem(marker)
        self.drone_markers[drone.id] = marker
        
        # Initialiser la traînée vide
        trail = gl.GLLinePlotItem(
            pos=np.array([drone.position, drone.position]),  # Deux points identiques = pas de ligne visible
            color=color,
            width=1.5,
            antialias=True
        )
        self.view_3d.addItem(trail)
        self.drone_trails[drone.id] = trail
        
        # Initialiser l'historique des positions
        self.drone_position_history[drone.id] = deque(maxlen=self.trail_length)
        self.drone_position_history[drone.id].append(drone.position.copy())
    
    def get_drone_color(self, drone):
        """Détermine la couleur d'un drone selon son statut et rôle"""
        if drone.status == DroneStatus.FAILED:
            return (0.8, 0.2, 0.2, 1.0)  # Rouge
        elif drone.status == DroneStatus.FAILING:
            return (1.0, 0.6, 0.0, 1.0)  # Orange
        elif drone.role == DroneRole.LEADER:
            return (0.2, 0.8, 0.2, 1.0)  # Vert
        else:
            return (0.2, 0.2, 0.8, 1.0)  # Bleu
    
    def update_drone_visuals(self, drone):
        """Met à jour les visualisations d'un drone"""
        # Récupérer la couleur actuelle
        color = self.get_drone_color(drone)
        
        # Mettre à jour le marqueur
        if drone.id in self.drone_markers:
            self.drone_markers[drone.id].setData(
                pos=np.array([drone.position]),
                color=color,
                size=15 if drone.role == DroneRole.LEADER else 10
            )
        
        # Mettre à jour la traînée
        if self.show_trails and drone.id in self.drone_trails:
            # Ajouter la position actuelle à l'historique
            self.drone_position_history[drone.id].append(drone.position.copy())
            
            # Mettre à jour la visualisation de la traînée si assez de points
            history = list(self.drone_position_history[drone.id])
            if len(history) >= 2:
                self.drone_trails[drone.id].setData(
                    pos=np.array(history),
                    color=color
                )
    
    def update_formation_lines(self):
        """Met à jour les lignes de formation"""
        # Supprimer les anciennes lignes
        for line in self.formation_lines:
            self.view_3d.removeItem(line)
        self.formation_lines = []
        
        if not self.show_formation:
            return
        
        # Trouver le leader
        leader = None
        for drone in self.swarm_simulation.drones:
            if drone.role == DroneRole.LEADER and drone.status == DroneStatus.ACTIVE:
                leader = drone
                break
        
        # Modification ici: vérifier si formation_center est None ou si c'est un array
        if leader is None or leader.formation_center is None:
            return
        
        # Pour chaque drone, dessiner une ligne vers sa position cible
        for drone in self.swarm_simulation.drones:
            if (drone.status == DroneStatus.ACTIVE and 
                drone.formation_position is not None and 
                not np.isnan(drone.formation_position).any()):
                
                # Calculer la qualité de la position
                distance = np.linalg.norm(drone.position - drone.formation_position)
                quality = max(0, 1 - distance / 30.0)  # 0 = mauvais, 1 = parfait
                
                # Couleur basée sur la qualité (rouge à vert)
                line_color = (1-quality, quality, 0.3, 0.5)
                
                # Créer une ligne entre la position actuelle et la position cible
                line = gl.GLLinePlotItem(
                    pos=np.array([drone.position, drone.formation_position]),
                    color=line_color,
                    width=1.5,
                    antialias=True
                )
                #self.view_3d.addItem(line)
                #self.formation_lines.append(line)
        
        # Visualiser le centre de formation
        center_point = gl.GLScatterPlotItem(
            pos=np.array([leader.formation_center]),
            color=(0.7, 0.3, 0.7, 0.7),  # Violet
            size=10,
            pxMode=True
        )
        #self.view_3d.addItem(center_point)
        #self.formation_lines.append(center_point)
    
    def update_communication_links(self):
        """Met à jour les liens de communication entre drones"""
        # Supprimer les anciens liens
        for link in self.communication_links:
            self.view_3d.removeItem(link)
        self.communication_links = []
        
        if not self.show_communication:
            return
        
        # Créer de nouveaux liens
        for i, drone1 in enumerate(self.swarm_simulation.drones):
            if drone1.status == DroneStatus.FAILED:
                continue
                
            for drone2 in self.swarm_simulation.drones[i+1:]:
                if drone2.status == DroneStatus.FAILED:
                    continue
                    
                if drone1.can_communicate_with(drone2) and drone2.can_communicate_with(drone1):
                    # Créer une ligne fine entre les drones
                    link = gl.GLLinePlotItem(
                        pos=np.array([drone1.position, drone2.position]),
                        color=(0.7, 0.7, 0.7, 0.2),  # Gris transparent
                        width=0.5,
                        antialias=True
                    )
                    self.view_3d.addItem(link)
                    self.communication_links.append(link)
    
    def update_metrics(self):
        """Met à jour les métriques de simulation"""
        # 1. Qualité de formation
        formation_quality = 0.0
        valid_drones = 0
        
        for drone in self.swarm_simulation.drones:
            if (drone.status == DroneStatus.ACTIVE and 
                drone.formation_position is not None and 
                not np.isnan(drone.formation_position).any()):
                
                error = np.linalg.norm(drone.position - drone.formation_position)
                formation_quality += error
                valid_drones += 1
        
        if valid_drones > 0:
            avg_formation_error = formation_quality / valid_drones
            formation_quality = max(0, 1 - (avg_formation_error / 30)) * 100  # En pourcentage
        else:
            formation_quality = 0
        
        # Décaler les données d'un point
        self.quality_data[:-1] = self.quality_data[1:]
        self.quality_data[-1] = formation_quality
        self.quality_curve.setData(self.quality_data)
        
        # 2. État des drones
        active_count = sum(1 for d in self.swarm_simulation.drones if d.status == DroneStatus.ACTIVE)
        failing_count = sum(1 for d in self.swarm_simulation.drones if d.status == DroneStatus.FAILING)
        failed_count = sum(1 for d in self.swarm_simulation.drones if d.status == DroneStatus.FAILED)
        
        self.active_data[:-1] = self.active_data[1:]
        self.active_data[-1] = active_count
        self.active_curve.setData(self.active_data)
        
        self.failing_data[:-1] = self.failing_data[1:]
        self.failing_data[-1] = failing_count
        self.failing_curve.setData(self.failing_data)
        
        self.failed_data[:-1] = self.failed_data[1:]
        self.failed_data[-1] = failed_count
        self.failed_curve.setData(self.failed_data)
        
        # 3. Connectivité du réseau
        connections = 0
        active_drones = [d for d in self.swarm_simulation.drones if d.status != DroneStatus.FAILED]
        
        for i, drone1 in enumerate(active_drones):
            for drone2 in active_drones[i+1:]:
                if drone1.can_communicate_with(drone2) and drone2.can_communicate_with(drone1):
                    connections += 1
        
        if len(active_drones) > 1:
            max_possible = (len(active_drones) * (len(active_drones) - 1)) / 2
            connectivity_ratio = connections / max_possible * 100  # En pourcentage
        else:
            connectivity_ratio = 0
        
        self.connectivity_data[:-1] = self.connectivity_data[1:]
        self.connectivity_data[-1] = connectivity_ratio
        self.connectivity_curve.setData(self.connectivity_data)
        
        # Mettre à jour les informations textuelles
        leader_id = "aucun"
        for drone in self.swarm_simulation.drones:
            if drone.role == DroneRole.LEADER and drone.status == DroneStatus.ACTIVE:
                leader_id = str(drone.id)
                break
        
        info_text = f"Drones: {len(self.swarm_simulation.drones)}\n"
        info_text += f"Leader: {leader_id}\n"
        info_text += f"Actifs: {active_count}, Défaillants: {failing_count}, En panne: {failed_count}\n"
        info_text += f"Qualité de formation: {formation_quality:.1f}%\n"
        info_text += f"Connectivité: {connectivity_ratio:.1f}%"
        
        self.info_label.setText(info_text)
    
    def update_simulation(self):
        """Met à jour la simulation et la visualisation"""
        # Mettre à jour la simulation plusieurs fois par image pour fluidité
        for _ in range(3):
            self.swarm_simulation.update()
        
        # Mettre à jour les visualisations des drones
        for drone in self.swarm_simulation.drones:
            # Ignorer les positions invalides
            if np.isnan(drone.position).any() or np.isinf(drone.position).any():
                continue
                
            self.update_drone_visuals(drone)
        
        # Mettre à jour les lignes de formation
        self.update_formation_lines()
        
        # Mettre à jour les liens de communication
        self.update_communication_links()
        
        # Mettre à jour les métriques
        self.update_metrics()
    
    def update_visualization(self):
        """Force une mise à jour complète de tous les éléments visuels"""
        # Mettre à jour chaque élément visuel
        self.update_formation_lines()
        self.update_communication_links()
        
        # Mettre à jour la visibilité des traînées
        for drone_id, trail in self.drone_trails.items():
            trail.setVisible(self.show_trails)
    
    def run(self):
        """Lance l'application"""
        self.window.show()
        return self.app.exec_()

# Classe combinée
class GPUAcceleratedSwarmSimulation:
    """Simulation d'essaim avec visualisation GPU accélérée (PyQtGraph)"""
    def __init__(self, num_drones=10, waypoints=None):
        # Créer la simulation
        self.simulation = DecentralizedSwarmSimulation(num_drones, waypoints)
        
        # Créer la visualisation
        self.visualization = PyQtGraphSwarmVisualization(self.simulation)
    
    def run(self):
        """Lance la simulation et la visualisation"""
        return self.visualization.run()

# Exécution de l'application
if __name__ == "__main__":
    # Analyser les arguments de ligne de commande
    import argparse
    from collections import deque
    
    parser = argparse.ArgumentParser(description='Simulation GPU d\'Essaim de Drones Décentralisée')
    parser.add_argument('--drones', type=int, default=10, help='Nombre de drones dans l\'essaim')
    parser.add_argument('--waypoints', type=float, nargs='+', 
                     default=[
                         0, 0, 60,      # Coin bas-gauche
                         150, 100, 80,    # Coin bas-droite
                         200, 200, 20,  # Coin haut-droite
                         0, 200, 100,    # Coin haut-gauche
                         0, 0, 60       # Retour au point de départ
                     ], 
                     help='Liste des coordonnées des waypoints (x1, y1, z1, x2, y2, z2, ...)')
    
    args = parser.parse_args()
    
    # Convertir les waypoints
    waypoints = []
    for i in range(0, len(args.waypoints), 3):
        if i+2 < len(args.waypoints):
            waypoints.append(args.waypoints[i:i+3])
    
    print(f"Initialisation de l'essaim de drones décentralisé avec {args.drones} drones...")
    print(f"Waypoints: {waypoints}")
    
    # Créer et lancer la simulation
    simulation = GPUAcceleratedSwarmSimulation(
        num_drones=args.drones, 
        waypoints=waypoints
    )
    
    sys.exit(simulation.run())