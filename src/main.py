#!/usr/bin/env python3
"""
Decentralized Drone Swarm Simulation
-----------------------------------
Main entry point for the drone swarm simulation.
Run with --help for command line options.
"""

import argparse
from swarm import DecentralizedSwarm
from visualization import visualize_swarm

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Decentralized Drone Swarm Simulation')
    parser.add_argument('--drones', type=int, default=5, help='Number of drones in the swarm')
    parser.add_argument('--fast', action='store_true', help='Run in fast mode for better performance')
    parser.add_argument('--steps', type=int, default=1000, help='Maximum simulation steps')
    
    return parser.parse_args()

def main():
    """Main function"""
    # Parse command line arguments
    args = parse_arguments()
    
    # Print welcome message
    print(f"Initializing decentralized drone swarm with {args.drones} drones...")
    print("Use the radio buttons to select different formations in real-time.")
    print("Press the 'Reset Simulation' button to restart with new drones.")
    print("Press the 'Add Drone' button to add a new drone to the swarm.")
    print("Press the 'Add Waypoint' button to add a new waypoint to the mission.")
    print("Press the 'Pause/Resume Navigation' button to toggle waypoint navigation.")
    
    # Create and run the simulation
    simulation = DecentralizedSwarm(num_drones=args.drones)
    
    # Patch the visualize method to use our visualization function
    simulation.visualize = lambda: visualize_swarm(simulation)
    
    # Run the simulation
    if args.fast:
        print("Running in fast mode...")
        simulation.run_simulation_fast(num_steps=args.steps)
    else:
        print("Running in standard mode...")
        simulation.run_simulation(num_steps=args.steps)

if __name__ == "__main__":
    main()