#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import argparse
from src.visualization.gpu_accelerated import GPUAcceleratedSwarmSimulation

def main():
    # Analyser les arguments de ligne de commande
    parser = argparse.ArgumentParser(description='AirPhalanx - Simulation GPU d\'Essaim de Drones Décentralisée')
    parser.add_argument('--drones', type=int, default=9, help='Nombre de drones dans l\'essaim')
    parser.add_argument('--waypoints', type=float, nargs='+', 
                     default=[
                         0, 0, 60,           # Coin bas-gauche
                         150, 100, 80,       # Coin bas-droite
                         200, 200, 20,       # Coin haut-droite
                         0, 200, 100,        # Coin haut-gauche
                         0, 0, 60            # Retour au point de départ
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

if __name__ == "__main__":
    main()