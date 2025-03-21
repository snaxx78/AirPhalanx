import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D
import random
import math
from enum import Enum
import time
import numpy as np
from matplotlib.widgets import RadioButtons, Button

class DroneStatus(Enum):
    ACTIVE = 1
    FAILING = 2
    FAILED = 3

class FormationType(Enum):
    V = "V Formation"
    LINE = "Line Formation"
    CIRCLE = "Circle Formation"
    SQUARE = "Square Formation"
    CUBE = "Cube Formation"
    SPHERE = "Sphere Formation"
    SHOAL = "Shoal Formation"

class Drone:
    def __init__(self, drone_id, position, velocity):
        self.id = drone_id
        self.position = np.array(position, dtype=float)
        self.velocity = np.array(velocity, dtype=float)
        self.acceleration = np.zeros(3)
        self.status = DroneStatus.ACTIVE
        
        # Communication and perception parameters
        self.communication_range = 100.0
        self.perception_range = 80.0
        
        # Network and neighbor management
        self.neighbors = []
        self.last_known_positions = {}
        
        # Formation information
        self.formation_position = None
        self.previous_formation_position = None
        self.changing_formation = False
        self.formation_transition_progress = 0.0
        self.transition_duration = 60
        self.transition_counter = 0
        
        # State and appearance
        self.color = 'blue'
        self.failure_probability = 0.0001
        self.failing_countdown = 20

    def distance_to(self, other):
        """Calculate distance between this drone and another drone safely"""
        try:
            # Check for NaN or inf values first
            if (np.isnan(self.position).any() or np.isnan(other.position).any() or
                np.isinf(self.position).any() or np.isinf(other.position).any()):
                # Return a large value to discourage interaction with invalid positions
                return float('inf')
                
            return np.linalg.norm(self.position - other.position)
        except Exception as e:
            print(f"Error calculating distance between drones: {e}")
            return float('inf')  # Safe fallback
    
    def can_communicate_with(self, other_drone):
        """Check if the drone can communicate with another drone"""
        return (self.distance_to(other_drone) <= self.communication_range and 
                other_drone.status != DroneStatus.FAILED)
    
    def discover_neighbors(self, swarm):
        """Discover neighboring drones within communication range"""
        self.neighbors = []
        
        for drone in swarm:
            if drone.id != self.id and self.can_communicate_with(drone):
                self.neighbors.append(drone.id)
                self.last_known_positions[drone.id] = drone.position.copy()
    
    def normalize(self, v):
        """Normalize a vector safely with thorough error checking"""
        # Check for NaN values first
        if np.isnan(v).any():
            print(f"Warning: NaN detected in vector: {v}")
            return np.zeros_like(v)
            
        # Check for inf values
        if np.isinf(v).any():
            print(f"Warning: Inf detected in vector: {v}")
            return np.zeros_like(v)
            
        # Calculate norm safely
        try:
            norm = np.linalg.norm(v)
            if norm > 1e-10:  # Use a small threshold instead of exactly zero
                return v / norm
            else:
                # Return zero vector if input is too small
                return np.zeros_like(v)
        except:
            print(f"Error normalizing vector: {v}")
            return np.zeros_like(v)
    
    def calculate_formation_position(self, swarm, formation_center, formation_type, formation_params):
        """Calculate the position the drone should occupy in the formation"""
        if self.status == DroneStatus.FAILED:
            return
            
        # Find active drones
        active_drones = [d for d in swarm if d.status != DroneStatus.FAILED]
        if not active_drones:
            return
        
        # Get drone index among active drones
        active_ids = sorted([d.id for d in active_drones])
        if self.id not in active_ids:
            return
            
        idx = active_ids.index(self.id)
        num_drones = len(active_ids)
        
        # Global spacing factor and min height
        spacing = formation_params.get('spacing', 20.0)
        min_height = formation_params.get('min_height', 40.0)
        
        # Calculate offsets based on formation type
        if formation_type == FormationType.V:
            # V formation
            angle = math.pi / 4
            side = 1 if idx % 2 == 0 else -1
            row = (idx // 2) + 1 if idx > 0 else 0  # Leader at the front
            
            offset_x = -row * spacing * math.cos(angle)
            offset_y = side * row * spacing * math.sin(angle)
            offset_z = -row * 3.0  # Slight descent for each row
            
        elif formation_type == FormationType.LINE:
            # Line formation
            offset_x = -idx * spacing
            offset_y = 0
            offset_z = 0
            
        elif formation_type == FormationType.CIRCLE:
            # Circle formation
            radius = spacing * 1.5
            angle = 2 * math.pi * idx / max(1, num_drones)
            offset_x = radius * math.cos(angle)
            offset_y = radius * math.sin(angle)
            offset_z = 0
            
        elif formation_type == FormationType.SQUARE:
            # Square formation
            side_length = math.ceil(math.sqrt(num_drones))
            row = idx // side_length
            col = idx % side_length
            
            offset_x = (col - side_length/2) * spacing
            offset_y = (row - side_length/2) * spacing
            offset_z = 0
            
        elif formation_type == FormationType.CUBE:
            # 3D Cube formation
            cube_side = math.ceil(num_drones**(1/3))
            depth = idx // (cube_side * cube_side)
            remainder = idx % (cube_side * cube_side)
            row = remainder // cube_side
            col = remainder % cube_side
            
            offset_x = (col - cube_side/2) * spacing
            offset_y = (row - cube_side/2) * spacing
            offset_z = (depth - cube_side/2) * spacing
            
        elif formation_type == FormationType.SPHERE:
            # Spherical formation using Fibonacci sphere algorithm for uniform distribution
            golden_ratio = (1 + 5**0.5) / 2
            i = idx + 1  # Start from 1
            theta = 2 * math.pi * i / golden_ratio
            phi = math.acos(1 - 2 * i / (num_drones + 1))
            
            radius = spacing * 1.5
            offset_x = radius * math.sin(phi) * math.cos(theta)
            offset_y = radius * math.sin(phi) * math.sin(theta)
            offset_z = radius * math.cos(phi)
            
        elif formation_type == FormationType.SHOAL:
            # Shoal (fish-like) formation with controlled randomness
            random.seed(self.id)  # Use drone ID as seed for consistent positions
            
            # Base position with controlled randomness
            max_radius = spacing * 2
            radius = random.random() * max_radius
            theta = random.uniform(0, 2 * math.pi)
            phi = random.uniform(-math.pi/4, math.pi/4)  # More limited in vertical
            
            offset_x = radius * math.cos(theta) * math.cos(phi)
            offset_y = radius * math.sin(theta) * math.cos(phi)
            offset_z = radius * math.sin(phi) * 0.5
            
            # Add time-based variations for natural movement
            time_factor = time.time() % 10  # 10-second cycle
            wave_x = math.sin(time_factor + self.id * 0.7) * 5.0
            wave_y = math.cos(time_factor + self.id * 1.3) * 5.0
            wave_z = math.sin(time_factor + self.id * 2.1) * 2.0
            
            offset_x += wave_x
            offset_y += wave_y
            offset_z += wave_z
            
            random.seed()  # Reset random seed
        else:
            # Default to V formation if unknown
            angle = math.pi / 4
            side = 1 if idx % 2 == 0 else -1
            row = (idx // 2) + 1 if idx > 0 else 0
            
            offset_x = -row * spacing * math.cos(angle)
            offset_y = side * row * spacing * math.sin(angle)
            offset_z = -row * 3.0

        # For static swarm, just use a simple offset system
        global_offset = np.array([offset_x, offset_y, offset_z])
        
        # Final formation position 
        new_formation_pos = formation_center + global_offset
        
        # Ensure minimum height
        if new_formation_pos[2] < min_height:
            new_formation_pos[2] = min_height
        
        # Implement smooth transitions for formation position
        if self.formation_position is None:
            # First calculation of formation position
            self.formation_position = new_formation_pos
            self.previous_formation_position = new_formation_pos.copy()
        else:
            # If in transition or if we need to start a new transition
            formation_distance = np.linalg.norm(new_formation_pos - self.formation_position)
            
            # If the new position is significantly different, start a transition
            if formation_distance > 5.0 and not self.changing_formation:
                self.changing_formation = True
                self.previous_formation_position = self.formation_position.copy()
                self.transition_counter = 0
                
            # Handle progressive transition if active
            if self.changing_formation:
                self.transition_counter += 1
                # Calculate transition progress (0.0 to 1.0)
                self.formation_transition_progress = min(1.0, self.transition_counter / self.transition_duration)
                
                # Interpolation between old and new position
                t = self.formation_transition_progress
                # Use attenuation function for more natural transition
                t_smooth = t * t * (3 - 2 * t)  # Hermite curve smoothing
                
                self.formation_position = (1 - t_smooth) * self.previous_formation_position + t_smooth * new_formation_pos
                
                # End the transition once complete
                if self.formation_transition_progress >= 1.0:
                    self.changing_formation = False
            else:
                # Small continuous updates for minor adjustments without complete transition
                self.formation_position = 0.9 * self.formation_position + 0.1 * new_formation_pos

    def calculate_steering_forces(self, swarm, formation_center, formation_type, formation_params):
        """Calculate steering forces with improved physics and error handling"""
        # Handle failed drones
        if self.status == DroneStatus.FAILED:
            return np.array([0, 0, -0.5])
            
        if self.status == DroneStatus.FAILING:
            return np.array([
                random.uniform(-0.5, 0.5),
                random.uniform(-0.5, 0.5),
                random.uniform(-0.5, 0.1)
            ])
        
        # Force weights (local copy to avoid modifying the original)
        weights = {
            'separation': formation_params.get('weights', {}).get('separation', 1.5),
            'alignment': formation_params.get('weights', {}).get('alignment', 0.8),
            'cohesion': formation_params.get('weights', {}).get('cohesion', 0.7),
            'formation': formation_params.get('weights', {}).get('formation', 2.0),
            'target': formation_params.get('weights', {}).get('target', 1.0),  # Reduced for static
            'altitude': formation_params.get('weights', {}).get('altitude', 1.2)
        }
        
        # Update neighbors list
        self.discover_neighbors(swarm)
        
        # Get neighboring drones
        neighbors = [d for d in swarm if d.id in self.neighbors]
        
        # Calculate vector to formation center
        try:
            # Check for NaN in positions
            if np.isnan(self.position).any() or np.isnan(formation_center).any():
                print(f"Warning: NaN in position or center. Position: {self.position}, Center: {formation_center}")
                # Reset position if it's NaN (prevent complete failure)
                if np.isnan(self.position).any():
                    self.position = np.array([0.0, 0.0, 50.0])
                
            vector_to_center = formation_center - self.position
            dist_to_center = np.linalg.norm(vector_to_center)
            
            # If isolated, move directly toward center
            if not neighbors:
                center_direction = self.normalize(vector_to_center)
                return center_direction * weights['target'] * 1.5
        except Exception as e:
            print(f"Error in center calculation for drone {self.id}: {e}")
            # Return a safe vector pointing upward in case of error
            return np.array([0.0, 0.0, 0.5])
        
        # Initialize forces
        separation_force = np.zeros(3)
        alignment_force = np.zeros(3)
        cohesion_force = np.zeros(3)
        formation_force = np.zeros(3)
        center_force = np.zeros(3)
        altitude_force = np.zeros(3)
        
        # 1. Separation force - with error checking
        separation_count = 0
        for neighbor in neighbors:
            try:
                distance = self.distance_to(neighbor)
                if distance < 15.0 and distance > 0:  # Avoid division by zero
                    repulsion = self.position - neighbor.position
                    # Safer calculation with capped division
                    repulsion = repulsion / max(distance * distance, 1e-6)
                    separation_force += repulsion
                    separation_count += 1
            except Exception as e:
                print(f"Error in separation calculation for drone {self.id}: {e}")
        
        if separation_count > 0:
            separation_force /= separation_count
            separation_force = self.normalize(separation_force)
            
        # 2. Alignment force - with error checking
        alignment_count = 0
        for neighbor in neighbors:
            try:
                if neighbor.status == DroneStatus.ACTIVE and not np.isnan(neighbor.velocity).any():
                    alignment_force += neighbor.velocity
                    alignment_count += 1
            except Exception as e:
                print(f"Error in alignment calculation for drone {self.id}: {e}")
        
        if alignment_count > 0:
            alignment_force /= alignment_count
            alignment_force = self.normalize(alignment_force)
        
        # 3. Cohesion force - with error checking
        center_of_mass = np.zeros(3)
        cohesion_count = 0
        for neighbor in neighbors:
            try:
                if neighbor.status == DroneStatus.ACTIVE and not np.isnan(neighbor.position).any():
                    center_of_mass += neighbor.position
                    cohesion_count += 1
            except Exception as e:
                print(f"Error in cohesion calculation for drone {self.id}: {e}")
        
        if cohesion_count > 0:
            center_of_mass /= cohesion_count
            cohesion_force = self.normalize(center_of_mass - self.position)
        
        # 4. Formation force - with error checking and smooth transitions
        formation_weight = weights['formation']  # Initialize with default
        if self.formation_position is not None:
            try:
                # Check for NaNs
                if not np.isnan(self.formation_position).any():
                    formation_force = self.normalize(self.formation_position - self.position)
                    
                    # Adjust formation force during transitions
                    distance_to_formation = np.linalg.norm(self.position - self.formation_position)
                    
                    # In transition, slightly reduce force to avoid too abrupt movements
                    if self.changing_formation:
                        # Adjust force based on transition progress
                        transition_factor = 0.6 + 0.4 * self.formation_transition_progress
                        formation_weight = weights['formation'] * transition_factor
                    else:
                        # Calculate adaptive force based on distance
                        formation_weight_scale = min(1.0, distance_to_formation / 20.0)
                        formation_weight = weights['formation'] * (1.0 + formation_weight_scale)
                        
                    formation_weight = min(formation_weight, 10.0)  # Cap the maximum weight
                else:
                    print(f"Warning: NaN in formation_position for drone {self.id}")
            except Exception as e:
                print(f"Error in formation force calculation for drone {self.id}: {e}")
        
        # 5. Center force (replaced target force for static swarm)
        center_force = self.normalize(vector_to_center)
        
        # 6. Altitude maintenance force - with error checking
        min_altitude = formation_params.get('min_height', 40.0)
        if self.position[2] < min_altitude:
            altitude_deficit = min_altitude - self.position[2]
            altitude_force = np.array([0, 0, 1.0]) * min(altitude_deficit / 10.0, 1.0)  # Cap the force
        
        # Combined steering force with weighted components
        try:
            # Adjust weights during transitions to favor formation
            if self.changing_formation:
                # Progressively reduce center influence during transition
                center_adjustment = 0.6 + 0.4 * (1 - self.formation_transition_progress)
                # Reduce alignment to prioritize positioning
                alignment_adjustment = 0.6
                # Increase separation to avoid collisions during transitions
                separation_adjustment = 1.3
                
                steering_force = (
                    separation_force * weights['separation'] * separation_adjustment +
                    alignment_force * weights['alignment'] * alignment_adjustment +
                    cohesion_force * weights['cohesion'] +
                    formation_force * formation_weight +
                    center_force * (weights['target'] * center_adjustment) +
                    altitude_force * weights['altitude']
                )
            else:
                steering_force = (
                    separation_force * weights['separation'] +
                    alignment_force * weights['alignment'] +
                    cohesion_force * weights['cohesion'] +
                    formation_force * formation_weight +
                    center_force * weights['target'] +
                    altitude_force * weights['altitude']
                )
            
            # Final check for NaN/inf
            if np.isnan(steering_force).any() or np.isinf(steering_force).any():
                print(f"Warning: Invalid steering force calculated for drone {self.id}: {steering_force}")
                # Return a safe default force
                return np.array([0.0, 0.0, 0.5])
                
            return steering_force
                
        except Exception as e:
            print(f"Error in final steering force calculation for drone {self.id}: {e}")
            # Return a safe default force
            return np.array([0.0, 0.0, 0.5])
    
    def update_status(self):
        """Update drone status (failure simulation)"""
        if self.status == DroneStatus.ACTIVE:
            if random.random() < self.failure_probability:
                self.status = DroneStatus.FAILING
                self.color = 'orange'
                print(f"Drone {self.id} is starting to fail!")
                
        elif self.status == DroneStatus.FAILING:
            self.failing_countdown -= 1
            if self.failing_countdown <= 0:
                self.status = DroneStatus.FAILED
                self.color = 'black'
                print(f"Drone {self.id} has completely failed!")
    
    def update(self, swarm, formation_center, formation_type, formation_params):
        """Update drone state based on environment with robust error handling"""
        try:
            # Update status
            self.update_status()
            
            # Check if position contains NaN and reset if needed
            if np.isnan(self.position).any():
                print(f"Drone {self.id} has NaN position, resetting position")
                self.position = np.array([
                    random.uniform(0, 20),
                    random.uniform(0, 20),
                    random.uniform(50, 60)
                ])
                self.velocity = np.array([0.0, 0.0, 0.0])
                self.acceleration = np.array([0.0, 0.0, 0.0])
            
            # Calculate formation position
            self.calculate_formation_position(swarm, formation_center, formation_type, formation_params)
            
            # Calculate steering forces
            steering_force = self.calculate_steering_forces(swarm, formation_center, formation_type, formation_params)
            
            # Apply acceleration with robust error handling
            self.acceleration = steering_force
            
            # Check for NaN or inf in acceleration
            if np.isnan(self.acceleration).any() or np.isinf(self.acceleration).any():
                print(f"Drone {self.id} has invalid acceleration, resetting it")
                self.acceleration = np.array([0.0, 0.0, 0.1])  # Small upward acceleration as a fallback
            
            # Apply acceleration limits safely
            max_acceleration = 0.35
            acc_magnitude = np.linalg.norm(self.acceleration)
            
            if acc_magnitude > max_acceleration and acc_magnitude > 1e-6:
                self.acceleration = self.acceleration * (max_acceleration / acc_magnitude)
            elif acc_magnitude <= 1e-6:
                # For near-zero acceleration, provide a tiny upward drift
                self.acceleration = np.array([0.0, 0.0, 0.01])
            
            # Adjust damping based on transition state
            damping = 0.9  # Default damping
            
            # Increase damping during transitions for smoother movements
            if self.changing_formation:
                # Adjust damping based on transition progress
                transition_smoothing = 0.85 - 0.05 * (1 - self.formation_transition_progress)
                damping = min(0.9, transition_smoothing)
            else:
                # If near formation position, increase damping for stability
                if self.formation_position is not None:
                    distance_to_formation = np.linalg.norm(self.position - self.formation_position)
                    if distance_to_formation < 5.0:
                        damping = 0.85  # Stronger damping when close to target position
            
            self.velocity = self.velocity * damping + self.acceleration
            
            # Check for NaN or inf in velocity
            if np.isnan(self.velocity).any() or np.isinf(self.velocity).any():
                print(f"Drone {self.id} has invalid velocity, resetting it")
                self.velocity = np.array([0.0, 0.0, 0.1])  # Small upward velocity as a fallback
            
            # Limit velocity safely - with adjustment for transitions
            if self.changing_formation:
                # Limit speed during transitions for more control
                transition_speed_factor = 0.8 + 0.2 * self.formation_transition_progress
                max_speed = 1.7 * transition_speed_factor if self.status == DroneStatus.ACTIVE else 0.7
            else:
                max_speed = 2.0 if self.status == DroneStatus.ACTIVE else 0.8
                
            speed = np.linalg.norm(self.velocity)
            
            if speed > max_speed and speed > 1e-6:
                self.velocity = self.velocity * (max_speed / speed)
            
            # Update position
            old_position = self.position.copy()
            self.position += self.velocity
            
            # Check for unrealistic position changes (teleporting)
            position_change = np.linalg.norm(self.position - old_position)
            # Reduce movement limit during transitions
            max_movement = 6.0 if self.changing_formation else 8.0
            if position_change > max_movement:
                print(f"Drone {self.id} moved too far in one step ({position_change}), limiting movement")
                # Limit the movement to a reasonable distance
                direction = self.normalize(self.position - old_position)
                self.position = old_position + direction * max_movement
            
            # Keep drones within a reasonable boundary box
            bounds = {
                'min_x': -100, 'max_x': 400,
                'min_y': -100, 'max_y': 300,
                'min_z': 10, 'max_z': 200
            }
            
            # Apply soft boundary constraints
            for i, (axis, min_val, max_val) in enumerate([
                ('x', bounds['min_x'], bounds['max_x']),
                ('y', bounds['min_y'], bounds['max_y']),
                ('z', bounds['min_z'], bounds['max_z'])
            ]):
                if self.position[i] < min_val:
                    self.position[i] = min_val
                    self.velocity[i] *= -0.5  # Bounce with energy loss
                elif self.position[i] > max_val:
                    self.position[i] = max_val
                    self.velocity[i] *= -0.5  # Bounce with energy loss
                    
        except Exception as e:
            print(f"Error updating drone {self.id}: {e}")
            # Reset drone to a stable state
            if self.status != DroneStatus.FAILED:
                self.position = np.array([
                    random.uniform(0, 20),
                    random.uniform(0, 20),
                    random.uniform(50, 60)
                ])
                self.velocity = np.array([0.0, 0.0, 0.0])
                self.acceleration = np.array([0.0, 0.0, 0.1])

class StaticDroneSwarm:
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
        self.start_time = time.time()
        self.simulation_stats = {
            'mean_distance': [],
            'formation_quality': []
        }
        
        # Static formation center
        self.formation_center = np.array([150, 150, 70])
        
        # Current formation type
        self.current_formation = FormationType.V
        self.previous_formation = None
        self.formation_change_time = time.time()
        
        # Formation parameters
        self.formation_params = {
            'spacing': 20.0,
            'min_height': 40.0,
            'weights': {
                'separation': 1.2,
                'alignment': 1.5,
                'cohesion': 1.0,
                'formation': 10.0,
                'target': 2.0,  # Reduced for static swarm
                'altitude': 1.2
            }
        }
        
        # Initialize swarm
        self.initialize_swarm()
    
    def initialize_swarm(self):
        """Initialize swarm of drones with starting positions"""
        for i in range(self.num_drones):
            position = [
                random.uniform(130, 170),  # Positions closer to formation center
                random.uniform(130, 170),
                random.uniform(60, 80)
            ]
            
            velocity = [
                random.uniform(-0.1, 0.1),
                random.uniform(-0.1, 0.1),
                random.uniform(-0.05, 0.05)
            ]
            
            drone = Drone(drone_id=i, position=position, velocity=velocity)
            self.drones.append(drone)
    
    def update(self):
        """Update global simulation state"""
        # Update each drone
        for drone in self.drones:
            drone.update(self.drones, self.formation_center, self.current_formation, self.formation_params)
    
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
        
        self.configure_axes()
    
    def configure_axes(self):
        """Configure plot axes"""
        self.ax.set_xlim([50, 250])
        self.ax.set_ylim([50, 250])
        self.ax.set_zlim([0, 150])
        self.ax.set_xlabel('X')
        self.ax.set_ylabel('Y')
        self.ax.set_zlabel('Z')
        self.ax.set_title('Static Drone Swarm Simulation - Select Formation')
    
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
        self.start_time = time.time()
        self.simulation_stats = {
            'mean_distance': [],
            'formation_quality': []
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
            position = [150, 150, 70]  # Default position near formation center
        
        velocity = [
            random.uniform(-0.1, 0.1),
            random.uniform(-0.1, 0.1),
            random.uniform(-0.05, 0.05)
        ]
        
        # Create and add the new drone
        new_drone = Drone(drone_id=new_id, position=position, velocity=velocity)
        self.drones.append(new_drone)
        print(f"Added new drone with ID {new_id}")
    
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
    
    def visualize(self):
        """Display current simulation state with robust error handling"""
        try:
            self.ax.clear()
            self.configure_axes()
            
            # Collect statistics
            self.collect_statistics()
            
            # Display drones
            for drone in self.drones:
                try:
                    # Skip if position contains NaN
                    if np.isnan(drone.position).any() or np.isinf(drone.position).any():
                        continue
                        
                    # Drone position marker
                    self.ax.scatter(
                        drone.position[0],
                        drone.position[1],
                        drone.position[2],
                        color=drone.color,
                        s=20,
                        marker='o' if drone.status == DroneStatus.ACTIVE else 'x',
                        alpha=0.8
                    )
                    
                    # Velocity vector
                    if drone.status != DroneStatus.FAILED:
                        velocity_norm = np.linalg.norm(drone.velocity)
                        if velocity_norm > 0.01:  # Only show velocity for meaningful movements
                            # Check for NaN in velocity
                            if not np.isnan(drone.velocity).any() and not np.isinf(drone.velocity).any():
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
                    
                    # Formation position indicator
                    if (drone.formation_position is not None and 
                        drone.status == DroneStatus.ACTIVE and 
                        not np.isnan(drone.formation_position).any() and 
                        not np.isinf(drone.formation_position).any()):
                        
                        # Line connecting drone to formation position
                        self.ax.plot(
                            [drone.position[0], drone.formation_position[0]],
                            [drone.position[1], drone.formation_position[1]],
                            [drone.position[2], drone.formation_position[2]],
                            color='lightblue',
                            linestyle=':',
                            alpha=0.3
                        )
                        
                        # Small marker for formation position
                        self.ax.scatter(
                            drone.formation_position[0],
                            drone.formation_position[1],
                            drone.formation_position[2],
                            color='lightblue',
                            s=20,
                            marker='.',
                            alpha=0.5
                        )
                except Exception as e:
                    print(f"Error displaying drone {drone.id}: {e}")
                    continue
                
            # Display formation center
            self.ax.scatter(
                self.formation_center[0],
                self.formation_center[1],
                self.formation_center[2],
                color='green',
                s=150,
                marker='*',
                alpha=0.8
            )
            
            # Display simulation information
            elapsed_time = time.time() - self.start_time
            
            # Count drones by status
            active_count = sum(1 for d in self.drones if d.status == DroneStatus.ACTIVE)
            failing_count = sum(1 for d in self.drones if d.status == DroneStatus.FAILING)
            failed_count = sum(1 for d in self.drones if d.status == DroneStatus.FAILED)
            
            # Count drones with NaN positions (unstable)
            unstable_count = sum(1 for d in self.drones if 
                                np.isnan(d.position).any() or np.isinf(d.position).any())
            
            # Display information
            info_text = [
                f"Time: {elapsed_time:.1f}s",
                f"Current Formation: {self.current_formation.value}",
                f"Active Drones: {active_count}",
                f"Failing Drones: {failing_count}",
                f"Failed Drones: {failed_count}"
            ]
            
            # Add unstable drone count if any
            if unstable_count > 0:
                info_text.append(f"Unstable Drones: {unstable_count}")
            
            # Add statistics if available
            if self.simulation_stats['formation_quality']:
                try:
                    quality = self.simulation_stats['formation_quality'][-1]
                    quality_status = "Excellent" if quality > 0.9 else "Good" if quality > 0.7 else "Fair" if quality > 0.5 else "Poor"
                    info_text.append(f"Formation Quality: {quality:.2f} ({quality_status})")
                except:
                    pass
                
            if self.simulation_stats['mean_distance']:
                try:
                    info_text.append(f"Avg Distance: {self.simulation_stats['mean_distance'][-1]:.1f}")
                except:
                    pass
                
            # Add color legend
            info_text.extend([
                "",
                "Blue: Active Drone",
                "Orange: Failing Drone",
                "Black: Failed Drone",
                "Green: Formation Center"
            ])
            
            # Display each information line
            for i, text in enumerate(info_text):
                y_pos = 0.95 - i * 0.03
                self.ax.text2D(0.70, y_pos, text, transform=self.ax.transAxes)
                
        except Exception as e:
            print(f"Error in visualization: {e}")
            # Try to recover the visualization for next frame
    
    def run_simulation(self, num_steps=1000):
        """Run simulation with animation"""
        self.setup_visualization()
        
        # Animation function
        def update_frame(frame):
            # Run multiple updates per frame for faster simulation
            for _ in range(5):  # Increased from 3 to 5 updates per frame
                self.update()
            self.visualize()
            return self.ax,
        
        # Create animation with faster updates
        ani = FuncAnimation(self.fig, update_frame, frames=num_steps, 
                            interval=40, blit=False, repeat=False)  # Reduced interval from 50 to 40
        
        plt.show()

# Main execution block
if __name__ == "__main__":
    # Parse command line arguments if desired
    import argparse
    
    parser = argparse.ArgumentParser(description='Interactive Drone Swarm Simulation')
    parser.add_argument('--drones', type=int, default=5, help='Number of drones in the swarm')
    parser.add_argument('--fast', action='store_true', help='Run in fast mode for better performance')
    parser.add_argument('--steps', type=int, default=1000, help='Maximum simulation steps')
    
    args = parser.parse_args()
    
    # Create the simulation
    print(f"Initializing drone swarm with {args.drones} drones...")
    print("Use the radio buttons to select different formations in real-time.")
    print("Press the 'Reset Simulation' button to restart with new drones.")
    print("Press the 'Add Drone' button to add a new drone to the swarm.")
    
    simulation = StaticDroneSwarm(num_drones=args.drones)
    
    # Run the simulation
    if args.fast:
        print("Running in fast mode...")
        simulation.run_simulation_fast(num_steps=args.steps)
    else:
        print("Running in standard mode...")
        simulation.run_simulation(num_steps=args.steps)
    
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