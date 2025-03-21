import numpy as np
import random
import time
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import RadioButtons, Button, CheckButtons

from config import DroneStatus, FormationType, DEFAULT_FORMATION_PARAMS, DEFAULT_WAYPOINTS
from drone import Drone

class DecentralizedSwarm:
    def __init__(self, num_drones=15):
        self.num_drones = num_drones
        self.drones = []
        self.fig = None
        self.ax = None
        self.radio_ax = None
        self.radio = None
        self.reset_button_ax = None
        self.reset_button = None
        self.add_drone_button_ax = None
        self.add_drone_button = None
        self.add_waypoint_button_ax = None
        self.add_waypoint_button = None
        self.toggle_waypoint_button_ax = None  # Pour pause/reprise de la navigation
        self.toggle_waypoint_button = None
        self.start_time = time.time()
        self.simulation_stats = {
            'mean_distance': [],
            'formation_quality': [],
            'mission_progress': 0
        }
        
        # Waypoints system
        self.waypoints = [np.array(waypoint) for waypoint in DEFAULT_WAYPOINTS]
        self.current_waypoint_index = 0
        self.waypoint_radius = 40.0  # Radius to consider waypoint reached
        self.waypoint_navigation_active = True  # État d'activation de la navigation
        
        # Formation stabilization parameters
        self.stabilization_mode = True  # Start with formation stabilization mode
        self.min_quality_to_move = 0.7  # Minimum formation quality to move to next waypoint
        self.formation_stabilization_time = 0  # Counter for tracking stabilization time
        self.stabilization_threshold = 60  # Frames to wait for stabilization before accepting less optimal formation
        
        # Current formation type
        self.current_formation = FormationType.V
        self.previous_formation = None
        self.formation_change_time = time.time()
        
        # Formation parameters
        self.formation_params = DEFAULT_FORMATION_PARAMS.copy()
        
        # Initialize swarm
        self.initialize_swarm()
        
    def initialize_swarm(self):
        """Initialize swarm of drones with starting positions"""
        for i in range(self.num_drones):
            position = [
                random.uniform(80, 120),  # Positions closer to first waypoint
                random.uniform(80, 120),
                random.uniform(50, 70)
            ]
            
            velocity = [
                random.uniform(-0.1, 0.1),
                random.uniform(-0.1, 0.1),
                random.uniform(-0.05, 0.05)
            ]
            
            drone = Drone(drone_id=i, position=position, velocity=velocity)
            self.drones.append(drone)

    def update_mission_progress(self):
        """Update mission progress by checking if current waypoint is reached"""
        # Si la navigation est désactivée, ne rien faire
        if not self.waypoint_navigation_active:
            return
            
        # Count active drones
        active_drones = [d for d in self.drones if d.status != DroneStatus.FAILED]
        if not active_drones:
            return
            
        # Calculate swarm center
        swarm_center = np.mean([d.position for d in active_drones], axis=0)
        
        # Get current waypoint
        current_waypoint = self.waypoints[self.current_waypoint_index]
        
        # Calculate distance to waypoint
        distance_to_waypoint = np.linalg.norm(swarm_center - current_waypoint)
        
        # Count drones that consider the waypoint reached
        drones_reached = sum(1 for d in active_drones if d.waypoint_reached)
        
        # Evaluate current formation quality
        formation_quality = 0.0
        if self.simulation_stats['formation_quality']:
            formation_quality = self.simulation_stats['formation_quality'][-1]
        
        # Determine if swarm is close enough to waypoint
        is_near_waypoint = (distance_to_waypoint < self.waypoint_radius or 
                           drones_reached > len(active_drones) * 0.7)
        
        # Formation quality status for display
        quality_status = "Excellent" if formation_quality > 0.9 else "Good" if formation_quality > 0.7 else "Fair" if formation_quality > 0.5 else "Poor"
        
        # Log formation status periodically
        if len(self.simulation_stats['formation_quality']) % 30 == 0:  # Log periodically
            print(f"Current formation quality: {formation_quality:.2f} ({quality_status})")
            if is_near_waypoint:
                print(f"Near waypoint. Stabilization time: {self.formation_stabilization_time}")
        
        # If we're near the waypoint, track stabilization time
        if is_near_waypoint:
            self.formation_stabilization_time += 1
        else:
            self.formation_stabilization_time = 0
            
        # Evaluate if we should move to next waypoint
        should_move = False
        
        # Case 1: Formation quality is good
        if formation_quality >= self.min_quality_to_move and is_near_waypoint:
            should_move = True
            move_reason = f"good formation quality ({formation_quality:.2f})"
            
        # Case 2: We've waited too long trying to stabilize
        elif self.formation_stabilization_time > self.stabilization_threshold and is_near_waypoint:
            # Even with lower quality, move on after waiting some time
            should_move = True
            move_reason = f"timeout with acceptable formation ({formation_quality:.2f})"
            
        # Move to next waypoint if conditions are met
        if should_move:
            # Move to next waypoint
            self.current_waypoint_index = (self.current_waypoint_index + 1) % len(self.waypoints)
            print(f"Waypoint reached ({move_reason})! Moving to waypoint: {self.current_waypoint_index+1}")
            
            # Reset waypoint reached status for all drones
            for drone in self.drones:
                drone.waypoint_reached = False
                drone.perceived_target = None  # Force perception update
            
            # Reset stabilization parameters
            self.formation_stabilization_time = 0
            
            # Update mission progress
            self.simulation_stats['mission_progress'] = (self.current_waypoint_index / len(self.waypoints)) * 100
    
    def update(self):
        """Update global simulation state"""
        # Update mission progress
        self.update_mission_progress()
        
        # Current waypoint - si la navigation est désactivée, mettre à None
        current_waypoint = None
        if self.waypoint_navigation_active:
            current_waypoint = self.waypoints[self.current_waypoint_index]
        
        # Evaluate current formation quality for adaptive behavior
        formation_quality = 0.0
        if self.simulation_stats['formation_quality']:
            formation_quality = self.simulation_stats['formation_quality'][-1]
        
        # Adjust formation parameters based on quality
        self._adjust_formation_weights(formation_quality)
            
        # Update each drone
        for drone in self.drones:
            drone.update(self.drones, current_waypoint, self.current_formation, self.formation_params)
    
    def _adjust_formation_weights(self, formation_quality):
        """Dynamically adjust formation weights based on quality"""
        # Base weights
        base_weights = {
            'separation': 1.2,
            'alignment': 1.5,
            'cohesion': 1.0,
            'formation': 8.0,
            'target': 2.5,
            'altitude': 1.2
        }
        
        # Adjust formation weight based on quality
        if formation_quality < 0.5:  # Poor formation
            # Significantly increase formation weight, reduce target weight
            formation_factor = 1.5
            target_factor = 0.6
        elif formation_quality < 0.7:  # Fair formation
            # Moderately increase formation weight, slightly reduce target
            formation_factor = 1.3
            target_factor = 0.8
        elif formation_quality < 0.85:  # Good formation
            # Slightly increase formation, normal target
            formation_factor = 1.1
            target_factor = 1.0
        else:  # Excellent formation
            # Normal formation, can emphasize target more
            formation_factor = 1.0
            target_factor = 1.1
        
        # Update weights
        self.formation_params['weights']['formation'] = base_weights['formation'] * formation_factor
        self.formation_params['weights']['target'] = base_weights['target'] * target_factor
    
    def setup_visualization(self):
        """Setup 3D visualization with interactive controls"""
        self.fig = plt.figure(figsize=(14, 10))
        
        # Main 3D plot area
        self.ax = self.fig.add_subplot(111, projection='3d')
        
        # Add radio button for formation selection
        self.radio_ax = plt.axes([0.02, 0.5, 0.15, 0.35])
        formation_options = [f.value for f in FormationType]
        self.radio = RadioButtons(self.radio_ax, formation_options, active=0)
        self.radio.on_clicked(self.formation_selected)
        
        # Add reset button
        self.reset_button_ax = plt.axes([0.02, 0.4, 0.15, 0.05])
        self.reset_button = Button(self.reset_button_ax, 'Reset Simulation')
        self.reset_button.on_clicked(self.reset_simulation)
        
        # Add drone button
        self.add_drone_button_ax = plt.axes([0.02, 0.32, 0.15, 0.05])
        self.add_drone_button = Button(self.add_drone_button_ax, 'Add Drone')
        self.add_drone_button.on_clicked(self.add_drone)
        
        # Add waypoint button
        self.add_waypoint_button_ax = plt.axes([0.02, 0.24, 0.15, 0.05])
        self.add_waypoint_button = Button(self.add_waypoint_button_ax, 'Add Waypoint')
        self.add_waypoint_button.on_clicked(self.add_waypoint)
        
        # Add toggle waypoint navigation button
        self.toggle_waypoint_button_ax = plt.axes([0.02, 0.16, 0.15, 0.05])
        self.toggle_waypoint_button = Button(self.toggle_waypoint_button_ax, 
                                           'Pause Navigation' if self.waypoint_navigation_active else 'Reprendre Navigation')
        self.toggle_waypoint_button.on_clicked(self.toggle_waypoint_navigation)
        
        self.configure_axes()
    
    def configure_axes(self):
        """Configure plot axes"""
        self.ax.set_xlim([-50, 350])
        self.ax.set_ylim([-50, 250])
        self.ax.set_zlim([0, 150])
        self.ax.set_xlabel('X')
        self.ax.set_ylabel('Y')
        self.ax.set_zlabel('Z')
        self.ax.set_title('Decentralized Drone Swarm Simulation - Select Formation')
    
    def formation_selected(self, label):
        """Handle formation selection from radio button"""
        for formation_type in FormationType:
            if formation_type.value == label:
                self.current_formation = formation_type
                print(f"Formation changed to: {self.current_formation.value}")
                break
    
    def reset_simulation(self, event):
        """Reset the simulation"""
        self.drones = []
        self.current_waypoint_index = 0
        self.start_time = time.time()
        self.simulation_stats = {
            'mean_distance': [],
            'formation_quality': [],
            'mission_progress': 0
        }
        self.initialize_swarm()
        print("Simulation reset with new drones.")
    
    def add_drone(self, event):
        """Add a new drone to the swarm"""
        # Find highest existing ID
        max_id = max([drone.id for drone in self.drones]) if self.drones else -1
        new_id = max_id + 1
        
        # Use average position of active drones as starting point
        active_drones = [d for d in self.drones if d.status == DroneStatus.ACTIVE]
        if active_drones:
            avg_position = np.mean([d.position for d in active_drones], axis=0)
            position = avg_position + np.array([
                random.uniform(-10, 10),
                random.uniform(-10, 10),
                random.uniform(5, 15)
            ])
        else:
            # Default position near first waypoint
            position = self.waypoints[0] + np.array([
                random.uniform(-20, 20),
                random.uniform(-20, 20),
                random.uniform(0, 20)
            ])
        
        velocity = [
            random.uniform(-0.1, 0.1),
            random.uniform(-0.1, 0.1),
            random.uniform(-0.05, 0.05)
        ]
        
        # Create and add the new drone
        new_drone = Drone(drone_id=new_id, position=position, velocity=velocity)
        self.drones.append(new_drone)
        print(f"Added new drone with ID {new_id}")
    
    def add_waypoint(self, event):
        """Add a new waypoint near the current one"""
        if not self.waypoints:
            # If no waypoints, add one at the center of active drones
            active_drones = [d for d in self.drones if d.status == DroneStatus.ACTIVE]
            if active_drones:
                center = np.mean([d.position for d in active_drones], axis=0)
                new_waypoint = center + np.array([50, 50, 10])
            else:
                new_waypoint = np.array([150, 150, 70])
        else:
            # Add a waypoint near the last one
            last_waypoint = self.waypoints[-1]
            new_waypoint = last_waypoint + np.array([
                random.uniform(-50, 50),
                random.uniform(-50, 50),
                random.uniform(-10, 10)
            ])
            # Ensure minimum height
            if new_waypoint[2] < 40:
                new_waypoint[2] = 40
        
        self.waypoints.append(new_waypoint)
        print(f"Added new waypoint at {new_waypoint}. Total waypoints: {len(self.waypoints)}")
        
    def collect_statistics(self):
        """Collect statistics about the swarm with error handling"""
        try:
            # Filter out drones with valid positions
            active_drones = [d for d in self.drones if d.status != DroneStatus.FAILED 
                            and not np.isnan(d.position).any() 
                            and not np.isinf(d.position).any()]
            
            if not active_drones:
                return
                
            # Calculate average distance between drones
            total_distance = 0
            count = 0
            for i, drone1 in enumerate(active_drones):
                for drone2 in active_drones[i+1:]:
                    try:
                        distance = np.linalg.norm(drone1.position - drone2.position)
                        if not np.isnan(distance) and not np.isinf(distance):
                            total_distance += distance
                            count += 1
                    except Exception as e:
                        print(f"Error calculating distance between drones {drone1.id} and {drone2.id}: {e}")
                    
            if count > 0:
                mean_distance = total_distance / count
                if not np.isnan(mean_distance) and not np.isinf(mean_distance):
                    self.simulation_stats['mean_distance'].append(mean_distance)
            
            # Evaluate formation quality
            formation_error = 0
            valid_drones = 0
            for drone in active_drones:
                try:
                    if (drone.formation_position is not None and 
                        not np.isnan(drone.formation_position).any() and 
                        not np.isinf(drone.formation_position).any()):
                        
                        error = np.linalg.norm(drone.position - drone.formation_position)
                        if not np.isnan(error) and not np.isinf(error):
                            formation_error += error
                            valid_drones += 1
                except Exception as e:
                    print(f"Error calculating formation error for drone {drone.id}: {e}")
                    
            if valid_drones > 0:
                avg_formation_error = formation_error / valid_drones
                if not np.isnan(avg_formation_error) and not np.isinf(avg_formation_error):
                    formation_quality = max(0, 1 - (avg_formation_error / 30))  # Normalize
                    self.simulation_stats['formation_quality'].append(formation_quality)
                        
        except Exception as e:
            print(f"Error collecting statistics: {e}")
            # Continue without updating statistics this frame
    
    def run_simulation(self, num_steps=1000):
        """Run simulation with animation"""
        self.setup_visualization()
        
        # Animation function
        def update_frame(frame):
            # Run multiple updates per frame for faster simulation
            for _ in range(5):  # Multiple updates per frame
                self.update()
            self.visualize()
            return self.ax,
        
        # Create animation with faster updates
        ani = FuncAnimation(self.fig, update_frame, frames=num_steps, 
                            interval=40, blit=False, repeat=False)
        
        plt.show()
    
    def run_simulation_fast(self, num_steps=1000):
        """Run simulation in fast mode without complex animation for better performance"""
        self.setup_visualization()
        
        for step in range(num_steps):
            # Multiple updates per visualization step
            for _ in range(5):  # More updates for fast mode
                self.update()
            
            # Visualize less frequently to speed up simulation
            if step % 2 == 0:  # Reduced from 3 to 2
                self.visualize()
                plt.pause(0.01)  # Very short pause
            
            # Check if all drones have failed
            if all(drone.status == DroneStatus.FAILED for drone in self.drones):
                print("All drones have failed. Ending simulation.")
                break
                
        plt.show()
    
    def toggle_waypoint_navigation(self, event):
        """Mettre en pause ou reprendre la navigation par waypoints"""
        self.waypoint_navigation_active = not self.waypoint_navigation_active
        status = "activée" if self.waypoint_navigation_active else "désactivée"
        self.toggle_waypoint_button.label.set_text(
            "Pause Navigation" if self.waypoint_navigation_active else "Reprendre Navigation"
        )
        print(f"Navigation par waypoints {status}")
        
        # Si la navigation est désactivée, les drones ne perçoivent plus le waypoint
        if not self.waypoint_navigation_active:
            for drone in self.drones:
                drone.waypoint_reached = True  # Considérer le waypoint comme atteint pour arrêter le mouvement
        else:
            # Réinitialiser la perception des waypoints
            for drone in self.drones:
                drone.waypoint_reached = False
                drone.perceived_target = None
        
        # Rafraîchir l'affichage du bouton
        self.toggle_waypoint_button_ax.figure.canvas.draw_idle()