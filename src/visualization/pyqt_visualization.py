import numpy as np
from collections import deque
from PyQt5 import QtWidgets, QtCore
import pyqtgraph as pg
import pyqtgraph.opengl as gl

from src.models.enums import DroneRole, DroneStatus

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
        
        # Paramètres d'affichage
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
        self.view_3d.setBackgroundColor((0, 0, 0))  # Couleur de fond noire
        
        # Créer le terrain si activé
        if self.show_terrain:
            self.create_terrain()
        
        # Créer les waypoints
        self.create_waypoints()
        
        # Initialiser les visualisations des drones
        for drone in self.swarm_simulation.drones:
            self.create_drone_visual(drone)
            
    def create_terrain(self):
        """Crée un terrain 3D réaliste"""
        # Définir la grille du terrain avec une résolution plus élevée
        x = np.linspace(-100, 300, 80)
        y = np.linspace(-100, 300, 80)
        x_mesh, y_mesh = np.meshgrid(x, y)
        z_mesh = np.zeros_like(x_mesh)
        
        # Générer un terrain avec des ondulations complexes
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
        
        print("Terrain réaliste créé")
    
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
        
        # Ligne connectant les waypoints
        path_points = np.array(waypoints + [waypoints[0]])  # Fermer la boucle
        path_line = gl.GLLinePlotItem(
            pos=path_points,
            color=(1, 0, 0, 0.5),
            width=2,
            antialias=True
        )
        self.view_3d.addItem(path_line)
        self.waypoints.append(path_line)
    
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
        
        # Vérifier si formation_center est None ou si c'est un array
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
                self.view_3d.addItem(line)
                self.formation_lines.append(line)
        
        # Visualiser le centre de formation
        center_point = gl.GLScatterPlotItem(
            pos=np.array([leader.formation_center]),
            color=(0.7, 0.3, 0.7, 0.7),  # Violet
            size=10,
            pxMode=True
        )
        self.view_3d.addItem(center_point)
        self.formation_lines.append(center_point)
    
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