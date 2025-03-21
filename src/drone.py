import numpy as np
import random
import math
import time
from config import DroneStatus, FormationType

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
        
        # Waypoint perception - each drone has its own perception of the target
        self.perceived_target = None
        self.waypoint_reached = False
        self.target_threshold = 15.0  # Distance threshold to consider waypoint reached
        
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
                
                # Share waypoint information with neighbors
                if drone.perceived_target is not None and self.perceived_target is None:
                    self.perceived_target = drone.perceived_target.copy()
                elif drone.perceived_target is not None and self.perceived_target is not None:
                    # Take average of targets to account for slight perception differences
                    self.perceived_target = 0.5 * self.perceived_target + 0.5 * drone.perceived_target
    
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
    
    def perceive_target(self, current_waypoint):
        """Update the drone's perception of the current target waypoint"""
        # Si la navigation est désactivée, ne pas percevoir de cible
        if current_waypoint is None:
            self.perceived_target = None
            self.waypoint_reached = True  # Considérer comme "atteint" pour ne pas bouger
            return
            
        # Add small perceptual noise to mimic decentralized decision making
        if current_waypoint is not None:
            # If first perception or major change, reset perception
            if self.perceived_target is None or np.linalg.norm(current_waypoint - self.perceived_target) > 50.0:
                self.perceived_target = current_waypoint + np.array([
                    random.uniform(-5.0, 5.0),
                    random.uniform(-5.0, 5.0),
                    random.uniform(-2.0, 2.0)
                ])
            else:
                # Small update to current perception (gradual convergence to real waypoint)
                self.perceived_target = 0.95 * self.perceived_target + 0.05 * current_waypoint
                # Add small noise to simulate imperfect communication
                self.perceived_target += np.array([
                    random.uniform(-0.5, 0.5),
                    random.uniform(-0.5, 0.5),
                    random.uniform(-0.2, 0.2)
                ])
                
            # Update waypoint reached status
            dist_to_target = np.linalg.norm(self.position - current_waypoint)
            self.waypoint_reached = dist_to_target < self.target_threshold
    
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

        # Create offset vector
        offset = np.array([offset_x, offset_y, offset_z])
        
        # If moving toward a waypoint, orient the formation accordingly
        if self.perceived_target is not None:
            # Calculate direction toward target
            swarm_center = np.mean([d.position for d in active_drones], axis=0)
            direction_to_target = self.perceived_target - swarm_center
            
            # Only orient if there's significant movement needed
            if np.linalg.norm(direction_to_target) > 5.0:
                # Create coordinate system for formation
                forward = self.normalize(direction_to_target)
                up = np.array([0, 0, 1])
                right = self.normalize(np.cross(forward, up))
                if np.linalg.norm(right) < 0.01:
                    right = np.array([1, 0, 0])
                up = self.normalize(np.cross(right, forward))
                
                # Transform offset to global coordinates
                transform = np.column_stack([right, up, forward])
                global_offset = transform.dot(offset)
            else:
                global_offset = offset
        else:
            global_offset = offset

        # Calculate formation position based on formation center
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

    def calculate_steering_forces(self, swarm, formation_params):
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
            'target': formation_params.get('weights', {}).get('target', 2.5),
            'altitude': formation_params.get('weights', {}).get('altitude', 1.2)
        }
        
        # Update neighbors list
        self.discover_neighbors(swarm)
        
        # Get neighboring drones
        neighbors = [d for d in swarm if d.id in self.neighbors]
        
        # Initialize forces
        separation_force = np.zeros(3)
        alignment_force = np.zeros(3)
        cohesion_force = np.zeros(3)
        formation_force = np.zeros(3)
        target_force = np.zeros(3)
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
                    
                    # Calculate formation quality for this drone (0.0 to 1.0)
                    distance_to_formation = np.linalg.norm(self.position - self.formation_position)
                    drone_formation_quality = max(0, 1 - (distance_to_formation / 30.0))
                    
                    # Adaptive formation weight: apply stronger correction when formation is poor
                    if drone_formation_quality < 0.5:  # Poor formation
                        formation_weight = weights['formation'] * 2.5  # Significant correction
                    elif drone_formation_quality < 0.7:  # Fair formation
                        formation_weight = weights['formation'] * 2.0  # Strong correction
                    elif drone_formation_quality < 0.9:  # Good formation
                        formation_weight = weights['formation'] * 1.5  # Moderate correction
                    else:  # Excellent formation
                        formation_weight = weights['formation']  # Standard weight
                    
                    # In transition, slightly reduce force to avoid too abrupt movements
                    if self.changing_formation:
                        # Adjust force based on transition progress
                        transition_factor = 0.6 + 0.4 * self.formation_transition_progress
                        formation_weight *= transition_factor
                    
                    formation_weight = min(formation_weight, 12.0)  # Increase max cap for stronger corrections
                else:
                    print(f"Warning: NaN in formation_position for drone {self.id}")
            except Exception as e:
                print(f"Error in formation force calculation for drone {self.id}: {e}")
        
        # 5. Target force - with error checking
        if self.perceived_target is not None and not self.waypoint_reached:
            try:
                vector_to_target = self.perceived_target - self.position
                distance_to_target = np.linalg.norm(vector_to_target)
                
                if distance_to_target > 0.1:  # Prevent division by zero
                    target_direction = vector_to_target / distance_to_target
                    target_force = target_direction
                    
                    # Determine if formation is good enough to move toward target
                    # Get overall formation quality from neighbors
                    formation_qualities = [max(0, 1 - (np.linalg.norm(n.position - n.formation_position) / 30.0)) 
                                          for n in neighbors 
                                          if n.formation_position is not None]
                    
                    # Add own quality
                    if self.formation_position is not None:
                        formation_qualities.append(drone_formation_quality)
                    
                    # Calculate average formation quality
                    avg_formation_quality = sum(formation_qualities) / max(1, len(formation_qualities))
                    
                    # Dynamically adjust target weight based on formation quality
                    quality_factor = min(avg_formation_quality * 1.5, 1.0)  # Cap at 1.0
                    
                    # Adjust target weight based on distance and formation quality
                    target_weight = weights['target'] * quality_factor
                    
                    if distance_to_target > 100.0:
                        target_weight *= 1.5  # Reduced from 2.0 to prioritize formation
                    elif distance_to_target > 50.0:
                        target_weight *= 1.2  # Reduced from 1.5
                    
                    # If formation is poor, significantly reduce target weight to fix formation first
                    if avg_formation_quality < 0.4:
                        target_weight *= 0.3  # Severely limit movement to prioritize formation
                    elif avg_formation_quality < 0.6:
                        target_weight *= 0.6  # Moderately limit movement
                    
                    # Reduce target force as we get closer
                    if distance_to_target < 30.0:
                        target_factor = distance_to_target / 30.0
                        target_weight *= target_factor
            except Exception as e:
                print(f"Error in target force calculation: {e}")
        
        # 6. Altitude maintenance force - with error checking
        min_altitude = formation_params.get('min_height', 40.0)
        if self.position[2] < min_altitude:
            altitude_deficit = min_altitude - self.position[2]
            altitude_force = np.array([0, 0, 1.0]) * min(altitude_deficit / 10.0, 1.0)  # Cap the force
        
        # Combined steering force with weighted components
        try:
            # Adjust weights during transitions to favor formation
            if self.changing_formation:
                # Progressively reduce target influence during transition
                target_adjustment = 0.6 + 0.4 * (1 - self.formation_transition_progress)
                # Reduce alignment to prioritize positioning
                alignment_adjustment = 0.6
                # Increase separation to avoid collisions during transitions
                separation_adjustment = 1.3
                
                steering_force = (
                    separation_force * weights['separation'] * separation_adjustment +
                    alignment_force * weights['alignment'] * alignment_adjustment +
                    cohesion_force * weights['cohesion'] +
                    formation_force * formation_weight +
                    target_force * (weights['target'] * target_adjustment) +
                    altitude_force * weights['altitude']
                )
            else:
                steering_force = (
                    separation_force * weights['separation'] +
                    alignment_force * weights['alignment'] +
                    cohesion_force * weights['cohesion'] +
                    formation_force * formation_weight +
                    target_force * weights['target'] +
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
    
    def update(self, swarm, current_waypoint, formation_type, formation_params):
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
            
            # Update target perception
            self.perceive_target(current_waypoint)
            
            # Calculate formation center based on swarm position and current waypoint
            active_drones = [d for d in swarm if d.status != DroneStatus.FAILED]
            if not active_drones:
                return
                
            swarm_center = np.mean([d.position for d in active_drones], axis=0)
            
            # If we have a waypoint, bias formation center toward it
            formation_center = swarm_center
            if self.perceived_target is not None and not self.waypoint_reached:
                vector_to_target = self.perceived_target - swarm_center
                distance_to_target = np.linalg.norm(vector_to_target)
                
                # Calculate a leading point ahead of the swarm
                if distance_to_target > 5.0:
                    direction_to_target = vector_to_target / distance_to_target
                    ahead_distance = min(30.0, distance_to_target * 0.5)  # Don't go too far ahead
                    formation_center = swarm_center + direction_to_target * ahead_distance
            
            # Calculate formation position
            self.calculate_formation_position(swarm, formation_center, formation_type, formation_params)
            
            # Calculate steering forces
            steering_force = self.calculate_steering_forces(swarm, formation_params)
            
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