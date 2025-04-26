import numpy as np
import random
import time
import math

from src.models.enums import DroneStatus, DroneRole, WaypointStatus
from src.models.messages import Message
from utils.vector_utils import normalize_vector

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
        self.formation_grid = {}
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
        election_timestamp = leader_data.get("election_timestamp", 0)
        
        if new_leader_id is not None:
            # Mettre à jour le leader
            self.leader_id = new_leader_id
            self.leader_last_seen = message.timestamp
            
            # Si ce n'est pas moi, je suis un suiveur
            if new_leader_id != self.id:
                self.role = DroneRole.FOLLOWER
                self.color = 'blue'
            
            # Réinitialisation explicite et complète de l'état d'élection
            self.election_in_progress = False
            self.election_votes = {}
            
            # Forcer une mise à jour de formation après l'élection
            if self.role == DroneRole.LEADER:
                self.last_formation_update = 0  # Pour forcer une mise à jour immédiate
                
            # Propager l'annonce pour s'assurer que tous les drones sont informés
            if message.ttl <= 0:
                # Si le TTL est épuisé, créer un nouveau message avec TTL frais pour propagation
                new_msg = self.send_message("leader_elected", {
                    "leader_id": new_leader_id,
                    "election_timestamp": election_timestamp
                })
                if new_msg:
                    self.relay_message(new_msg, all_drones)
    
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
            formation_direction = normalize_vector(current_waypoint - leader_position)
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
        perpendicular = normalize_vector(np.cross(formation_direction, np.array([0, 0, 1])))
        
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
            
            separation_force = normalize_vector(separation_force)
        
        # 2. Force d'alignement (basée sur les positions connues)
        # Nous n'avons pas de vélocité des voisins directement, approximation basée sur positions
        alignment_force = np.zeros(3)
        
        # 3. Force de cohésion
        if neighbors_data:
            center_of_mass = np.mean([pos for _, pos, _ in neighbors_data], axis=0)
            cohesion_force = normalize_vector(center_of_mass - self.position)
        
        # 4. Force de formation
        self.formation_position = self.calculate_formation_position()
        if self.formation_position is not None:
            formation_force = normalize_vector(self.formation_position - self.position)
            
            # Augmenter le poids si loin de la position de formation
            distance_to_formation = np.linalg.norm(self.position - self.formation_position)
            if distance_to_formation > 30.0:
                weights['formation'] = 4.0
            elif distance_to_formation > 15.0:
                weights['formation'] = 3.5
        
        # Si je suis le leader, je veux me diriger vers le waypoint courant
        if self.role == DroneRole.LEADER:
            current_target = self.get_current_target()
            target_force = normalize_vector(current_target - self.position)
            
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
                    swarm_cohesion_force = normalize_vector(swarm_center - self.position)
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