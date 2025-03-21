import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import time
from config import DroneStatus

def visualize_swarm(swarm):
    """Display current simulation state with robust error handling"""
    try:
        swarm.ax.clear()
        swarm.configure_axes()
        
        # Collect statistics
        swarm.collect_statistics()
        
        # Display drones
        for drone in swarm.drones:
            try:
                # Skip if position contains NaN
                if np.isnan(drone.position).any() or np.isinf(drone.position).any():
                    continue
                    
                # Drone position marker
                swarm.ax.scatter(
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
                            swarm.ax.quiver(
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
                    
                    # Calculate formation error for this drone
                    error = np.linalg.norm(drone.position - drone.formation_position)
                    # Color gradient from red (poor) to green (good)
                    error_normalized = min(1.0, error / 30.0)  # Cap at 1.0
                    formation_color = (
                        error_normalized,  # Red: more for higher error
                        1 - error_normalized,  # Green: more for lower error
                        0.5  # Fixed blue component
                    )
                    
                    # Line connecting drone to formation position
                    swarm.ax.plot(
                        [drone.position[0], drone.formation_position[0]],
                        [drone.position[1], drone.formation_position[1]],
                        [drone.position[2], drone.formation_position[2]],
                        color=formation_color,
                        linestyle=':',
                        alpha=0.3
                    )
                    
                    # Small marker for formation position
                    swarm.ax.scatter(
                        drone.formation_position[0],
                        drone.formation_position[1],
                        drone.formation_position[2],
                        color=formation_color,
                        s=20,
                        marker='.',
                        alpha=0.5
                    )
                
                # Display target perception for each drone (showing decentralization)
                if (drone.perceived_target is not None and 
                    drone.status == DroneStatus.ACTIVE and
                    not np.isnan(drone.perceived_target).any()):
                    
                    # Small marker for perceived target
                    swarm.ax.scatter(
                        drone.perceived_target[0],
                        drone.perceived_target[1],
                        drone.perceived_target[2],
                        color='pink' if drone.waypoint_reached else 'yellow',
                        s=5,
                        marker='.',
                        alpha=0.3
                    )
            except Exception as e:
                print(f"Error displaying drone {drone.id}: {e}")
                continue
            
        # Display mission waypoints
        visualize_waypoints(swarm)
        
        # Display simulation information
        display_simulation_info(swarm)
            
    except Exception as e:
        print(f"Error in visualization: {e}")
        # Try to recover the visualization for next frame

def visualize_waypoints(swarm):
    """Visualize waypoints and their connections"""
    for i, waypoint in enumerate(swarm.waypoints):
        # Couleur différente si navigation en pause
        if not swarm.waypoint_navigation_active:
            color = 'red' if i == swarm.current_waypoint_index else 'gray'
        else:
            color = 'green' if i == swarm.current_waypoint_index else 'gray'
        
        swarm.ax.scatter(
            waypoint[0],
            waypoint[1],
            waypoint[2],
            color=color,
            s=150,
            marker='*',
            alpha=0.8
        )
        
        # Connect waypoints with lines
        if i < len(swarm.waypoints) - 1:
            next_waypoint = swarm.waypoints[i+1]
            swarm.ax.plot(
                [waypoint[0], next_waypoint[0]],
                [waypoint[1], next_waypoint[1]],
                [waypoint[2], next_waypoint[2]],
                color='gray',
                linestyle='--',
                alpha=0.5
            )
    
    # Draw waypoint reach radius
    if swarm.waypoint_navigation_active:
        current_waypoint = swarm.waypoints[swarm.current_waypoint_index]
        u = np.linspace(0, 2 * np.pi, 20)
        v = np.linspace(0, np.pi, 20)
        radius = swarm.waypoint_radius
        x = current_waypoint[0] + radius * np.outer(np.cos(u), np.sin(v))
        y = current_waypoint[1] + radius * np.outer(np.sin(u), np.sin(v))
        z = current_waypoint[2] + radius * np.outer(np.ones(np.size(u)), np.cos(v))
        swarm.ax.plot_surface(x, y, z, color='green', alpha=0.1)

def display_simulation_info(swarm):
    """Display information about the current simulation state"""
    try:
        # Display simulation information
        elapsed_time = time.time() - swarm.start_time
        
        # Count drones by status
        active_count = sum(1 for d in swarm.drones if d.status == DroneStatus.ACTIVE)
        failing_count = sum(1 for d in swarm.drones if d.status == DroneStatus.FAILING)
        failed_count = sum(1 for d in swarm.drones if d.status == DroneStatus.FAILED)
        
        # Count drones with NaN positions (unstable)
        unstable_count = sum(1 for d in swarm.drones if 
                            np.isnan(d.position).any() or np.isinf(d.position).any())
        
        # Count drones that perceive waypoint as reached
        reached_count = sum(1 for d in swarm.drones if 
                            d.status == DroneStatus.ACTIVE and d.waypoint_reached)
        
        # Get formation quality
        formation_quality = 0.0
        if swarm.simulation_stats['formation_quality']:
            formation_quality = swarm.simulation_stats['formation_quality'][-1]
        quality_status = "Excellent" if formation_quality > 0.9 else "Good" if formation_quality > 0.7 else "Fair" if formation_quality > 0.5 else "Poor"
        
        # Display information
        info_text = [
            f"Time: {elapsed_time:.1f}s",
            f"Current Formation: {swarm.current_formation.value}",
            f"Navigation: {'ACTIVE' if swarm.waypoint_navigation_active else 'PAUSED'}",
            f"Waypoint: {swarm.current_waypoint_index+1}/{len(swarm.waypoints)}",
            f"Active Drones: {active_count}",
            f"Failing Drones: {failing_count}",
            f"Failed Drones: {failed_count}",
            f"Waypoint Reached: {reached_count}/{active_count} drones",
            f"Formation Quality: {formation_quality:.2f} ({quality_status})",
            f"Stabilization Time: {swarm.formation_stabilization_time}/{swarm.stabilization_threshold}"
        ]
        
        # Add unstable drone count if any
        if unstable_count > 0:
            info_text.append(f"Unstable Drones: {unstable_count}")
        
        # Add weight information
        info_text.append(f"Formation Weight: {swarm.formation_params['weights']['formation']:.1f}")
        info_text.append(f"Target Weight: {swarm.formation_params['weights']['target']:.1f}")
            
        if swarm.simulation_stats['mean_distance']:
            try:
                info_text.append(f"Avg Distance: {swarm.simulation_stats['mean_distance'][-1]:.1f}")
            except:
                pass
        
        # Add mission progress
        info_text.append(f"Mission Progress: {swarm.simulation_stats['mission_progress']:.1f}%")
            
        # Add color legend
        info_text.extend([
            "",
            "Blue: Active Drone",
            "Orange: Failing Drone",
            "Black: Failed Drone",
            "Green → Red: Good → Poor Formation",
            f"{'Green' if swarm.waypoint_navigation_active else 'Red'}: Current Waypoint",
            "Gray: Future Waypoints",
            "Yellow: Perceived Target"
        ])
        
        # Display each information line
        for i, text in enumerate(info_text):
            y_pos = 0.95 - i * 0.03
            swarm.ax.text2D(1.10, y_pos, text, transform=swarm.ax.transAxes)
                
    except Exception as e:
        print(f"Error in display_simulation_info: {e}")