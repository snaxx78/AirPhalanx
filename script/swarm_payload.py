import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D
import random
import math
from enum import Enum
import time
import numpy as np
from matplotlib.widgets import RadioButtons, Button
from itertools import combinations

class DroneStatus(Enum):
    ACTIVE = 1
    FAILING = 2
    FAILED = 3

class CableStatus(Enum):
    NORMAL = 1
    TANGLED = 2
    BROKEN = 3

class FormationType(Enum):
    V = "V Formation"
    LINE = "Line Formation"
    CIRCLE = "Circle Formation"
    SQUARE = "Square Formation"
    CUBE = "Cube Formation"
    SPHERE = "Sphere Formation"
    SHOAL = "Shoal Formation"

class Payload:
    def __init__(self, position, mass=10.0):
        self.position = np.array(position, dtype=float)
        self.velocity = np.zeros(3)
        self.acceleration = np.zeros(3)
        self.mass = mass  # Mass of the payload in kg
        self.attachments = []  # List of drones attached to the payload
        
        # Physical properties
        self.drag_coefficient = 0.1
        self.gravity = np.array([0, 0, -0.2])  # Gravity vector (reduced for simulation)
        
        # Physical dimensions
        self.width = 20.0
        self.height = 5.0
        self.depth = 20.0
        
        # Cable crossing detection
        self.cable_crossings = []  # List to store detected cable crossings
        
        # Visualization
        self.color = 'red'
        self.size = 50  # Display size
        
        # Stability metrics
        self.stability_history = []  # History of stability values
        self.position_history = []  # History of positions for tracking movement
        self.max_history_length = 50  # Maximum length of history to maintain
        
        # Weight distribution
        self.weight_distribution = []  # How evenly the weight is distributed
        
    def attach_drone(self, drone_id, attachment_point, cable_length=30.0, cable_status=CableStatus.NORMAL):
        """Attach a drone to the payload with a fixed-length cable"""
        # attachment_point is relative to payload center
        self.attachments.append({
            'drone_id': drone_id,
            'attachment_point': np.array(attachment_point),
            'cable_length': cable_length,  # Fixed cable length with no slack
            'force': np.zeros(3),
            'tension': 0.0,
            'load_percentage': 0.0,  # Percentage of total load this drone carries
            'cable_status': cable_status
        })
    
    def detach_drone(self, drone_id):
        """Detach a drone from the payload"""
        self.attachments = [a for a in self.attachments if a['drone_id'] != drone_id]
    
    def update_attachment(self, drone_id, drone_position, drone_lift_capacity):
        """Update the force applied by a drone on the payload"""
        for attachment in self.attachments:
            if attachment['drone_id'] == drone_id:
                # Skip if cable is broken
                if attachment['cable_status'] == CableStatus.BROKEN:
                    attachment['force'] = np.zeros(3)
                    attachment['tension'] = 0.0
                    return
                
                # Calculate position where cable connects to payload
                attachment_world_pos = self.position + attachment['attachment_point']
                
                # Vector from attachment point to drone
                cable_vector = drone_position - attachment_world_pos
                distance = np.linalg.norm(cable_vector)
                
                # Fixed-length cable physics:
                # 1. If drone is too far, pull payload up toward drone
                # 2. If drone is too close, no force (slack cable)
                
                # Normalize cable direction
                if distance > 0:  # Avoid division by zero
                    cable_direction = cable_vector / distance
                else:
                    cable_direction = np.array([0, 0, 1])  # Default upward
                
                # Calculate tension based on drone position vs cable length
                if distance >= attachment['cable_length']:
                    # Cable is taut - force proportional to how much it's stretched
                    # With fixed cables we add a strong constraint force
                    tension = min(drone_lift_capacity, 5.0)  # Limited by drone capacity
                    attachment['force'] = cable_direction * tension
                    attachment['tension'] = tension
                else:
                    # Cable is slack - no force
                    attachment['force'] = np.zeros(3)
                    attachment['tension'] = 0.0
                
                return
                
    def calculate_center_of_mass(self, drones):
        """Calculate center of mass of attached active drones"""
        total_mass = 0
        weighted_position = np.zeros(3)
        
        # Only consider active, attached drones
        for attachment in self.attachments:
            if attachment['cable_status'] != CableStatus.BROKEN:
                drone = next((d for d in drones if d.id == attachment['drone_id'] and 
                            d.status != DroneStatus.FAILED), None)
                
                if drone:
                    weighted_position += drone.position * drone.mass
                    total_mass += drone.mass
        
        if total_mass > 0:
            return weighted_position / total_mass
        else:
            return self.position.copy()  # Default to current position
    
    def detect_cable_crossings(self, drones):
        """Detect if any cables are crossing/tangled"""
        self.cable_crossings = []
        
        # Only check active attachments with valid drones
        active_attachments = []
        for attachment in self.attachments:
            drone_id = attachment['drone_id']
            drone = next((d for d in drones if d.id == drone_id), None)
            
            if drone and drone.status != DroneStatus.FAILED and drone.attached_to_payload:
                attachment_world_pos = self.position + attachment['attachment_point']
                active_attachments.append({
                    'attachment': attachment,
                    'drone': drone,
                    'drone_pos': drone.position,
                    'attachment_pos': attachment_world_pos
                })
        
        # Check all pairs of cables for crossings
        for i, a1 in enumerate(active_attachments):
            for a2 in active_attachments[i+1:]:
                # Skip if either cable is already broken
                if (a1['attachment']['cable_status'] == CableStatus.BROKEN or 
                    a2['attachment']['cable_status'] == CableStatus.BROKEN):
                    continue
                
                # Check for cable crossing using line segment intersection
                if self.check_cable_intersection(
                    a1['drone_pos'], a1['attachment_pos'],
                    a2['drone_pos'], a2['attachment_pos']):
                    
                    # Record the crossing
                    self.cable_crossings.append((a1['attachment']['drone_id'], a2['attachment']['drone_id']))
                    
                    # Mark cables as tangled (will be broken if they stay tangled)
                    a1['attachment']['cable_status'] = CableStatus.TANGLED
                    a2['attachment']['cable_status'] = CableStatus.TANGLED
    
    def check_cable_intersection(self, p1, p2, p3, p4):
        """
        Check if two line segments (p1-p2) and (p3-p4) intersect in 3D space.
        Using a simplified approach - checking if the minimum distance between
        the line segments is less than a threshold.
        """
        # Define a threshold for considering lines as crossing
        threshold = 3.0  # Increased for larger payload
        
        # Calculate minimum distance between two line segments
        def closest_pt_to_line(a, b, p):
            # Find closest point on line segment a-b to point p
            ab = b - a
            ab_squared = np.dot(ab, ab)
            
            if ab_squared < 1e-10:  # Line segment is a point
                return a
            
            ap = p - a
            t = np.dot(ap, ab) / ab_squared
            t = max(0, min(1, t))  # Clamp to [0,1] for line segment
            
            return a + t * ab
        
        # Compute closest points on each line segment to the other
        closest_on_1_to_3 = closest_pt_to_line(p1, p2, p3)
        closest_on_1_to_4 = closest_pt_to_line(p1, p2, p4)
        closest_on_2_to_3 = closest_pt_to_line(p3, p4, p1)
        closest_on_2_to_4 = closest_pt_to_line(p3, p4, p2)
        
        # Calculate minimum distances
        min_dist_1 = min(
            np.linalg.norm(closest_on_1_to_3 - p3),
            np.linalg.norm(closest_on_1_to_4 - p4)
        )
        
        min_dist_2 = min(
            np.linalg.norm(closest_on_2_to_3 - p1),
            np.linalg.norm(closest_on_2_to_4 - p2)
        )
        
        # Return True if distance is below threshold
        return min(min_dist_1, min_dist_2) < threshold
    
    def break_tangled_cables(self):
        """Break any cables that have been tangled for too long"""
        tangled_count = 0
        
        for attachment in self.attachments:
            if attachment['cable_status'] == CableStatus.TANGLED:
                tangled_count += 1
                # 10% chance per update to break a tangled cable
                if random.random() < 0.1:
                    attachment['cable_status'] = CableStatus.BROKEN
                    print(f"Cable to drone {attachment['drone_id']} BROKE due to tangling!")
        
        return tangled_count
    
    def calculate_weight_distribution(self):
        """Calculate how evenly the payload weight is distributed"""
        # Skip if no attachments
        active_attachments = [a for a in self.attachments 
                            if a['cable_status'] != CableStatus.BROKEN and a['tension'] > 0]
        
        if not active_attachments:
            self.weight_distribution = []
            return 0.0
        
        # Sum all tensions
        total_tension = sum(a['tension'] for a in active_attachments)
        if total_tension < 1e-6:
            self.weight_distribution = []
            return 0.0
        
        # Calculate percentage of load for each attachment
        for attachment in active_attachments:
            percentage = attachment['tension'] / total_tension
            attachment['load_percentage'] = percentage
        
        # Store the distribution for visualization
        self.weight_distribution = [a['load_percentage'] for a in active_attachments]
        
        # Calculate distribution evenness (1.0 is perfectly even, lower values are less even)
        ideal_percentage = 1.0 / len(active_attachments)
        variance = sum((a['load_percentage'] - ideal_percentage) ** 2 for a in active_attachments)
        evenness = 1.0 / (1.0 + variance * 10)  # Scale to 0-1 range
        
        return evenness
    
    def update(self, drones):
        """Update payload physics - now driven directly by drone positions"""
        # Detect cable crossings before calculating forces
        self.detect_cable_crossings(drones)
        
        # Break tangled cables with some probability
        tangled_count = self.break_tangled_cables()
        
        # Reset acceleration
        self.acceleration = np.zeros(3)
        
        # Add gravity
        self.acceleration += self.gravity
        
        # Calculate forces from attached drones
        total_force = np.zeros(3)
        active_attachments = 0
        
        # First, update tensions and forces for all attachments
        for attachment in self.attachments:
            drone_id = attachment['drone_id']
            # Find the drone by ID
            drone = next((d for d in drones if d.id == drone_id), None)
            
            # Skip broken cables
            if attachment['cable_status'] == CableStatus.BROKEN:
                continue
            
            if drone and drone.status != DroneStatus.FAILED:
                # Update the force from this drone
                self.update_attachment(drone_id, drone.position, drone.max_lift_force)
                
                if np.linalg.norm(attachment['force']) > 0:
                    active_attachments += 1
                    total_force += attachment['force']
        
        # If there are active carriers, calculate target position based on center of mass
        if active_attachments > 0:
            # Calculate ideal position - center of mass of carrying drones, shifted down by cable length
            ideal_position = self.calculate_center_of_mass(drones)
            
            # Adjust vertical position based on average cable length
            avg_cable_length = sum(a['cable_length'] for a in self.attachments 
                                 if a['cable_status'] != CableStatus.BROKEN) / max(1, active_attachments)
            
            # Shift down by cable length - payload hangs below drones
            ideal_position[2] -= avg_cable_length
            
            # Calculate new forces to move toward ideal position and counter gravity
            gravity_compensation = -self.gravity * self.mass  # Force to exactly counter gravity
            
            # Position correction force (spring-like)
            correction_factor = 0.1  # Strength of correction
            position_correction = (ideal_position - self.position) * correction_factor
            
            # Direct forces-based model - use direct forces from drones, plus gravity compensation
            self.acceleration = (total_force + gravity_compensation) / self.mass
            
            # Add damping based on current velocity to prevent oscillation
            damping_factor = 0.95
            self.velocity *= damping_factor
            
            # Update velocity and position
            self.velocity += self.acceleration
            self.position += self.velocity
        else:
            # No active carriers - payload falls under gravity
            self.acceleration = self.gravity
            self.velocity += self.acceleration
            self.position += self.velocity
        
        # Ensure minimum height (don't go below ground)
        if self.position[2] < self.height / 2:
            self.position[2] = self.height / 2
            self.velocity[2] = max(0, self.velocity[2])  # Bounce if hitting ground
        
        # Calculate weight distribution
        distribution_evenness = self.calculate_weight_distribution()
        
        # Calculate stability metric (less movement = more stable)
        current_stability = 1.0 / (1.0 + np.linalg.norm(self.velocity))
        
        # Update history
        self.stability_history.append(current_stability)
        self.position_history.append(self.position.copy())
        
        # Trim history if too long
        if len(self.stability_history) > self.max_history_length:
            self.stability_history = self.stability_history[-self.max_history_length:]
            self.position_history = self.position_history[-self.max_history_length:] #Bounce if hitting ground
    
    def get_current_stability(self):
        """Get the current stability value (0 to 1)"""
        if self.stability_history:
            return self.stability_history[-1]
        return 0
    
    def get_average_stability(self):
        """Get the average stability over recent history"""
        if self.stability_history:
            return sum(self.stability_history) / len(self.stability_history)
        return 0
    
    def get_position_variance(self):
        """Calculate position variance over recent history to gauge stability"""
        if len(self.position_history) < 2:
            return 0
            
        positions = np.array(self.position_history)
        return np.mean(np.var(positions, axis=0))
    
    def get_weight_distribution_evenness(self):
        """Get how evenly the weight is distributed (1 is perfect)"""
        active_attachments = [a for a in self.attachments 
                           if a['cable_status'] != CableStatus.BROKEN and a['tension'] > 0]
        
        if not active_attachments:
            return 0.0
        
        ideal_percentage = 1.0 / len(active_attachments)
        variance = sum((a['load_percentage'] - ideal_percentage) ** 2 for a in active_attachments)
        return 1.0 / (1.0 + variance * 10)  # Scale to 0-1 range

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
        
        # Payload attachment information
        self.attached_to_payload = False
        self.attachment_point = None
        self.cable_vector = None
        self.cable_length = 0.0
        self.cable_tension = 0.0
        self.cable_status = CableStatus.NORMAL
        
        # Physical properties
        self.mass = 1.0  # Drone mass in kg
        self.max_lift_force = 2.0  # Maximum lift force in kg for payload (varies by drone size)
        
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

    def calculate_payload_force(self, payload):
        """Calculate the force needed to support the payload"""
        payload_force = np.zeros(3)
        
        if not self.attached_to_payload or self.status == DroneStatus.FAILED:
            return payload_force
        
        # Find my attachment info in the payload
        attachment = next((a for a in payload.attachments if a['drone_id'] == self.id), None)
        if not attachment:
            return payload_force
            
        # Update cable status from payload
        self.cable_status = attachment['cable_status']
        
        # If cable is broken, no force is applied
        if self.cable_status == CableStatus.BROKEN:
            self.cable_tension = 0.0
            return payload_force
            
        # Calculate the world position of my attachment point
        attachment_world_pos = payload.position + attachment['attachment_point']
        
        # Vector from drone to attachment point (cable direction)
        cable_vector = attachment_world_pos - self.position
        distance = np.linalg.norm(cable_vector)
        
        # Store cable vector and length for visualization
        self.cable_vector = cable_vector
        self.cable_length = distance
        
        # If cable is taut, apply force
        if distance >= attachment['cable_length']:
            # Normalize cable direction
            cable_direction = cable_vector / max(distance, 1e-6)
            
            # Calculate the tension in the cable
            stretch = distance - attachment['cable_length']
            
            # Spring force (Hooke's law)
            spring_constant = 0.5  # Spring stiffness
            tension = stretch * spring_constant
            
            # Limit the maximum force based on drone capability
            tension = min(tension, self.max_lift_force)
            
            # Force is applied in the direction from drone to attachment point
            payload_force = cable_direction * tension
            
            # Store tension for visualization
            self.cable_tension = tension
        else:
            # Cable is slack
            self.cable_tension = 0.0
        
        return payload_force
        
    def calculate_steering_forces(self, swarm, formation_center, payload, formation_type, formation_params):
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
            'target': formation_params.get('weights', {}).get('target', 1.0),
            'altitude': formation_params.get('weights', {}).get('altitude', 1.2),
            'payload': formation_params.get('weights', {}).get('payload', 3.0),
            'cable_avoidance': formation_params.get('weights', {}).get('cable_avoidance', 2.0)  # New weight
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
        payload_force = np.zeros(3)
        cable_avoidance_force = np.zeros(3)  # New force for avoiding other cables
        
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
        
        # 7. Payload force - force to support the payload
        if self.attached_to_payload and payload:
            try:
                # Don't calculate payload forces if cable is broken
                if self.cable_status != CableStatus.BROKEN:
                    payload_support_force = self.calculate_payload_force(payload)
                    payload_force = self.normalize(payload_support_force)
                    
                    # If payload is below, increase upward force
                    if payload.position[2] < self.position[2]:
                        vertical_adjustment = np.array([0, 0, 1.0]) * (self.position[2] - payload.position[2]) * 0.05
                        payload_force += vertical_adjustment
                    
                    # Normalize again
                    payload_force = self.normalize(payload_force)
                    
                    # If this drone is carrying too much weight, increase its importance
                    attachment = next((a for a in payload.attachments if a['drone_id'] == self.id), None)
                    if attachment and attachment['load_percentage'] > (1.0 / len(payload.attachments)) * 1.5:
                        # This drone is carrying >50% more than its fair share
                        weights['payload'] *= 1.5  # Increase payload weight to signal other drones to help
            except Exception as e:
                print(f"Error in payload force calculation for drone {self.id}: {e}")
        
        # 8. Cable avoidance force - avoid crossing other cables
        if self.attached_to_payload and payload and self.cable_status != CableStatus.BROKEN:
            try:
                avoidance_vector = np.zeros(3)
                avoidance_count = 0
                
                # Get my attachment point
                my_attachment = next((a for a in payload.attachments 
                                    if a['drone_id'] == self.id and a['cable_status'] != CableStatus.BROKEN), None)
                
                if my_attachment:
                    my_attachment_pos = payload.position + my_attachment['attachment_point']
                    
                    # Check all other active cables
                    for other_attachment in payload.attachments:
                        if (other_attachment['drone_id'] != self.id and 
                            other_attachment['cable_status'] != CableStatus.BROKEN):
                            
                            # Find the other drone
                            other_drone = next((d for d in swarm 
                                              if d.id == other_attachment['drone_id'] and 
                                              d.status != DroneStatus.FAILED), None)
                            
                            if other_drone:
                                other_attachment_pos = payload.position + other_attachment['attachment_point']
                                
                                # Check if cables might be getting close
                                # Calculate minimum distance between cable line segments
                                min_dist = self.minimum_distance_between_lines(
                                    self.position, my_attachment_pos,
                                    other_drone.position, other_attachment_pos
                                )
                                
                                # If cables are close, create avoidance force
                                if min_dist < 5.0:  # Threshold for "too close"
                                    # Calculate midpoint of my cable
                                    my_midpoint = (self.position + my_attachment_pos) / 2
                                    # Calculate midpoint of other cable
                                    other_midpoint = (other_drone.position + other_attachment_pos) / 2
                                    
                                    # Direction to move away
                                    away_vector = my_midpoint - other_midpoint
                                    away_vector = self.normalize(away_vector)
                                    
                                    # Stronger avoidance for closer cables
                                    avoidance_strength = 1.0 / max(min_dist, 0.1)
                                    avoidance_vector += away_vector * avoidance_strength
                                    avoidance_count += 1
                    
                    if avoidance_count > 0:
                        cable_avoidance_force = self.normalize(avoidance_vector)
            except Exception as e:
                print(f"Error in cable avoidance calculation for drone {self.id}: {e}")
        
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
                    altitude_force * weights['altitude'] +
                    payload_force * weights['payload'] +
                    cable_avoidance_force * weights['cable_avoidance']
                )
            else:
                steering_force = (
                    separation_force * weights['separation'] +
                    alignment_force * weights['alignment'] +
                    cohesion_force * weights['cohesion'] +
                    formation_force * formation_weight +
                    center_force * weights['target'] +
                    altitude_force * weights['altitude'] +
                    payload_force * weights['payload'] +
                    cable_avoidance_force * weights['cable_avoidance']
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
    
    def minimum_distance_between_lines(self, p1, p2, p3, p4):
        """Calculate the minimum distance between two line segments in 3D space"""
        def closest_point_on_line(a, b, p):
            # Find closest point on line segment a-b to point p
            ab = b - a
            ab_squared = np.dot(ab, ab)
            
            if ab_squared < 1e-10:  # Line segment is a point
                return a
            
            ap = p - a
            t = np.dot(ap, ab) / ab_squared
            t = max(0, min(1, t))  # Clamp to [0,1] for line segment
            
            return a + t * ab
        
        # Compute closest points on each line segment to the other
        p13 = closest_point_on_line(p1, p2, p3)
        p14 = closest_point_on_line(p1, p2, p4)
        p23 = closest_point_on_line(p3, p4, p1)
        p24 = closest_point_on_line(p3, p4, p2)
        
        # Calculate distances
        d13 = np.linalg.norm(p13 - p3)
        d14 = np.linalg.norm(p14 - p4)
        d23 = np.linalg.norm(p23 - p1)
        d24 = np.linalg.norm(p24 - p2)
        
        # Return minimum distance
        return min(d13, d14, d23, d24)
    
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
    
    def update(self, swarm, formation_center, payload, formation_type, formation_params):
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
            
            # If failed, just apply gravity and minimal physics
            if self.status == DroneStatus.FAILED:
                # Detach from payload if needed
                if self.attached_to_payload and payload:
                    payload.detach_drone(self.id)
                    self.attached_to_payload = False
                
                # Apply gravity
                self.acceleration = np.array([0, 0, -0.2])
                self.velocity += self.acceleration
                self.position += self.velocity
                
                # Stop if on ground
                if self.position[2] <= 0:
                    self.position[2] = 0
                    self.velocity = np.zeros(3)
                    self.acceleration = np.zeros(3)
                return
            
            # Check if cable is broken and update status
            if self.attached_to_payload and payload:
                attachment = next((a for a in payload.attachments if a['drone_id'] == self.id), None)
                if attachment:
                    self.cable_status = attachment['cable_status']
            
            # Calculate steering forces
            steering_force = self.calculate_steering_forces(swarm, formation_center, payload, formation_type, formation_params)
            
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
            'formation_quality': [],
            'payload_stability': [],
            'weight_distribution': []
        }
        
        # Static formation center
        self.formation_center = np.array([150, 150, 70])
        
        # Payload setup
        self.payload = Payload(position=[150, 150, 50], mass=10.0)
        
        # Current formation type
        self.current_formation = FormationType.CIRCLE
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
                'formation': 8.0,      # Increased for better formation
                'target': 1.5,         # Reduced for static swarm
                'altitude': 1.2,
                'payload': 3.0,        # Weight for payload forces
                'cable_avoidance': 2.0 # Weight for cable avoidance
            }
        }
        
        # Initialize swarm
        self.initialize_swarm()
    
    def initialize_swarm(self):
        """Initialize swarm of drones with starting positions and attach them to payload with fixed-length cables"""
        # Create drones
        for i in range(self.num_drones):
            position = [
                random.uniform(130, 170),  # Positions closer to formation center
                random.uniform(130, 170),
                random.uniform(70, 90)     # Start higher above payload
            ]
            
            velocity = [
                random.uniform(-0.1, 0.1),
                random.uniform(-0.1, 0.1),
                random.uniform(-0.05, 0.05)
            ]
            
            # Vary lift capacity slightly between drones
            lift_capacity = 2.0 + random.uniform(-0.3, 0.3)
            
            drone = Drone(drone_id=i, position=position, velocity=velocity)
            drone.max_lift_force = lift_capacity  # Set drone's lift capacity
            
            self.drones.append(drone)
        
        # Choose carrier drones (first 4 drones will be carriers)
        carrier_count = min(4, self.num_drones)
        
        # Make payload larger
        self.payload.size = 100  # Visual size for drawing
        
        # Define physical dimensions of the payload (rectangular box)
        payload_width = 20
        payload_height = 5
        payload_depth = 20
        
        # Fixed cable length (same for all carrier drones)
        fixed_cable_length = 30.0
        
        # Attach carrier drones to corners of the payload
        for i in range(carrier_count):
            # Calculate corner position based on index
            # Front-left, front-right, back-right, back-left
            corner_x = payload_width/2 * (1 if i in [1, 2] else -1)
            corner_y = payload_depth/2 * (1 if i in [2, 3] else -1)
            corner_z = payload_height/2  # Top of payload
            
            attachment_point = np.array([corner_x, corner_y, corner_z])
            
            # Register attachment with fixed length cable
            self.payload.attach_drone(
                drone_id=i, 
                attachment_point=attachment_point,
                cable_length=fixed_cable_length,  # Fixed cable length
                cable_status=CableStatus.NORMAL
            )
            
            # Update drone's attachment info
            self.drones[i].attached_to_payload = True
            self.drones[i].attachment_point = attachment_point.copy()
            
            # Position drone initially at proper distance above attachment point
            world_attachment = self.payload.position + attachment_point
            
            # Position directly above attachment point at fixed cable length
            initial_position = world_attachment.copy()
            initial_position[2] += fixed_cable_length  # Position directly above
            
            # Set drone position
            self.drones[i].position = initial_position
    
    def update(self):
        """Update global simulation state"""
        # Update each drone
        for drone in self.drones:
            drone.update(self.drones, self.formation_center, self.payload, self.current_formation, self.formation_params)
        
        # Update payload physics after all drones have updated
        self.payload.update(self.drones)
    
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
        self.ax.set_title('Static Drone Swarm with Payload - Select Formation')
    
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
            'formation_quality': [],
            'payload_stability': [],
            'weight_distribution': []
        }
        
        # Reset payload
        self.payload = Payload(position=[150, 150, 50], mass=10.0)
        self.payload.width = 20.0
        self.payload.height = 5.0
        self.payload.depth = 20.0
        
        # Initialize swarm again
        self.initialize_swarm()
        print("Simulation reset with new drones and payload.")
    
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
        new_drone.max_lift_force = 2.0 + random.uniform(-0.3, 0.3)  # Vary lift capacity
        self.drones.append(new_drone)
        
        # Decide if the new drone should be a carrier
        # Only attach if we've lost carriers and there are fewer than 4 carriers
        active_carriers = sum(1 for d in self.drones 
                             if d.attached_to_payload and d.status != DroneStatus.FAILED)
        
        if active_carriers < 4:
            # Determine which corner to attach to (based on which corners are free)
            attached_corners = []
            for attachment in self.payload.attachments:
                if attachment['cable_status'] != CableStatus.BROKEN:
                    attached_corners.append(tuple(attachment['attachment_point']))
            
            # Define the 4 corners of the payload
            hw = self.payload.width / 2
            hh = self.payload.height / 2
            hd = self.payload.depth / 2
            
            corners = [
                np.array([-hw, -hd, hh]),  # Front left
                np.array([hw, -hd, hh]),   # Front right
                np.array([hw, hd, hh]),    # Back right
                np.array([-hw, hd, hh])    # Back left
            ]
            
            # Find a free corner
            for corner in corners:
                if tuple(corner) not in attached_corners:
                    # Attach drone to this corner
                    fixed_cable_length = 30.0  # Fixed cable length
                    
                    # Register attachment with fixed length cable
                    self.payload.attach_drone(
                        drone_id=new_id, 
                        attachment_point=corner,
                        cable_length=fixed_cable_length,
                        cable_status=CableStatus.NORMAL
                    )
                    
                    # Update drone's attachment info
                    new_drone.attached_to_payload = True
                    new_drone.attachment_point = corner.copy()
                    
                    # Position drone initially at proper distance above attachment point
                    world_attachment = self.payload.position + corner
                    
                    # Position directly above attachment point at fixed cable length
                    initial_position = world_attachment.copy()
                    initial_position[2] += fixed_cable_length  # Position directly above
                    
                    # Set drone position
                    new_drone.position = initial_position
                    
                    print(f"Added new drone with ID {new_id} and attached to payload at corner")
                    return
            
            # If all corners are taken, just add as non-carrier
            print(f"Added new drone with ID {new_id} (not attached to payload - all corners occupied)")
        else:
            print(f"Added new drone with ID {new_id} (not attached to payload)")
    
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
            
            # Collect payload stability statistics
            if self.payload:
                stability = self.payload.get_current_stability()
                self.simulation_stats['payload_stability'].append(stability)
                
                # Weight distribution evenness
                evenness = self.payload.get_weight_distribution_evenness()
                self.simulation_stats['weight_distribution'].append(evenness)
                        
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
                
                # Display payload first
                if self.payload:
                    # Main payload body
                    self.ax.scatter(
                        self.payload.position[0],
                        self.payload.position[1],
                        self.payload.position[2],
                        color=self.payload.color,
                        s=self.payload.size,
                        marker='s',  # Square for payload
                        alpha=0.8
                    )
                    
                    # Draw payload's recent path (shows stability/instability)
                    if len(self.payload.position_history) > 2:
                        history = np.array(self.payload.position_history[-20:])  # Last 20 positions
                        self.ax.plot(
                            history[:, 0],
                            history[:, 1],
                            history[:, 2],
                            'r--',  # Red dashed line
                            linewidth=1,
                            alpha=0.3
                        )
                
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
                        
                        # Draw cable to payload if attached
                        if drone.attached_to_payload and self.payload and drone.status != DroneStatus.FAILED:
                            # Find the attachment point
                            attachment = next((a for a in self.payload.attachments if a['drone_id'] == drone.id), None)
                            if attachment:
                                # Calculate world position of attachment point
                                attachment_world_pos = self.payload.position + attachment['attachment_point']
                                
                                # Draw cable with color based on tension
                                tension = attachment['tension']
                                # Color from green (low tension) to yellow to red (high tension)
                                if tension <= 0.5:
                                    cable_color = 'green'
                                elif tension <= 1.5:
                                    cable_color = 'yellow'
                                else:
                                    cable_color = 'red'
                                    
                                # Draw cable
                                self.ax.plot(
                                    [drone.position[0], attachment_world_pos[0]],
                                    [drone.position[1], attachment_world_pos[1]],
                                    [drone.position[2], attachment_world_pos[2]],
                                    color=cable_color,
                                    linestyle='-',
                                    linewidth=1,
                                    alpha=0.7
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
                
                # Count drones attached to payload
                attached_count = sum(1 for d in self.drones if d.attached_to_payload and d.status != DroneStatus.FAILED)
                
                # Display information
                info_text = [
                    f"Time: {elapsed_time:.1f}s",
                    f"Current Formation: {self.current_formation.value}",
                    f"Active Drones: {active_count}",
                    f"Failing Drones: {failing_count}",
                    f"Failed Drones: {failed_count}",
                    f"Attached to Payload: {attached_count}"
                ]
                
                # Add payload information
                if self.payload:
                    info_text.append(f"Payload Position: ({self.payload.position[0]:.1f}, {self.payload.position[1]:.1f}, {self.payload.position[2]:.1f})")
                
                # Add statistics if available
                if self.simulation_stats['formation_quality']:
                    try:
                        quality = self.simulation_stats['formation_quality'][-1]
                        quality_status = "Excellent" if quality > 0.9 else "Good" if quality > 0.7 else "Fair" if quality > 0.5 else "Poor"
                        info_text.append(f"Formation Quality: {quality:.2f} ({quality_status})")
                    except:
                        pass
                    
                if self.simulation_stats['payload_stability']:
                    try:
                        stability = self.simulation_stats['payload_stability'][-1]
                        stability_status = "High" if stability > 0.9 else "Good" if stability > 0.7 else "Moderate" if stability > 0.5 else "Poor"
                        info_text.append(f"Payload Stability: {stability:.2f} ({stability_status})")
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
                    "Red: Payload",
                    "Green: Formation Center"
                ])
                
                # Display each information line
                for i, text in enumerate(info_text):
                    y_pos = 0.95 - i * 0.03
                    self.ax.text2D(0.70, y_pos, text, transform=self.ax.transAxes)
                    
            except Exception as e:
                print(f"Error in visualization: {e}")
                # Try to recover the visualization for next frameimport matplotlib.pyplot as plt
        
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

if __name__ == "__main__":
    # Parse command line arguments if desired
    import argparse
    
    parser = argparse.ArgumentParser(description='Static Drone Swarm with Payload Simulation')
    parser.add_argument('--drones', type=int, default=4, help='Number of drones in the swarm')
    parser.add_argument('--fast', action='store_true', help='Run in fast mode for better performance')
    parser.add_argument('--steps', type=int, default=1000, help='Maximum simulation steps')
    parser.add_argument('--payload-mass', type=float, default=10.0, help='Mass of the payload in kg')
    parser.add_argument('--cable-length', type=float, default=30.0, help='Fixed cable length in units')
    parser.add_argument('--failure-rate', type=float, default=0.0001, help='Drone failure probability per step')
    
    args = parser.parse_args()
    
    # Create the simulation
    print(f"Initializing drone swarm with {args.drones} drones")
    print(f"Payload mass: {args.payload_mass}kg, Cable length: {args.cable_length} units")
    print("Use the radio buttons to select different formations in real-time.")
    print("Press the 'Reset Simulation' button to restart with new drones.")
    print("Press the 'Add Drone' button to add a new drone to the swarm.")
    
    simulation = StaticDroneSwarm(num_drones=args.drones)
    
    # Configure payload and cables
    simulation.payload.mass = args.payload_mass
    
    # Update cable lengths
    for attachment in simulation.payload.attachments:
        attachment['cable_length'] = args.cable_length
    
    # Set failure rate
    for drone in simulation.drones:
        drone.failure_probability = args.failure_rate
    
    # Run the simulation
    if args.fast:
        print("Running in fast mode...")
        simulation.run_simulation_fast(num_steps=args.steps)
    else:
        print("Running in standard mode...")
        simulation.run_simulation(num_steps=args.steps)