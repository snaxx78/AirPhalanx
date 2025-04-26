#!/usr/bin/env python3
"""
Simulation Décentralisée d'un Essaim de Drones
----------------------------------------------
Point d'entrée principal pour la simulation d'essaim de drones.
Lancer avec --help pour afficher les options de ligne de commande.
"""

import argparse
from swarm import DecentralizedSwarm
from visualization import visualize_swarm

def parse_arguments():
    """Analyser les arguments de la ligne de commande"""
    parser = argparse.ArgumentParser(description='Simulation Décentralisée d\'un Essaim de Drones')
    parser.add_argument('--drones', type=int, default=5, help='Nombre de drones dans l\'essaim')
    parser.add_argument('--fast', action='store_true', help='Exécuter en mode rapide pour de meilleures performances')
    parser.add_argument('--steps', type=int, default=1000, help='Nombre maximum d\'étapes de simulation')

    
    return parser.parse_args()

def main():
    """Fonction principale"""
    # Analyser les arguments de la ligne de commande
    args = parse_arguments()
    
    # Afficher le message de bienvenue
    print(f"Initialisation de l'essaim de drones décentralisé avec {args.drones} drones...")
    print("Utilisez les boutons radio pour sélectionner différentes formations en temps réel.")
    print("Appuyez sur le bouton 'Réinitialiser la simulation' pour redémarrer avec de nouveaux drones.")
    print("Appuyez sur le bouton 'Ajouter un drone' pour ajouter un nouveau drone à l'essaim.")
    print("Appuyez sur le bouton 'Ajouter un waypoint' pour ajouter un nouveau point de passage à la mission.")
    print("Appuyez sur le bouton 'Pause/Reprendre la navigation' pour activer ou désactiver la navigation par waypoints.")
    
    # Créer et lancer la simulation
    simulation = DecentralizedSwarm(num_drones=args.drones)
    
    # Modifier la méthode de visualisation pour utiliser notre fonction personnalisée
    simulation.visualize = lambda: visualize_swarm(simulation)
    
    # Exécuter la simulation
    if args.fast:
        print("Exécution en mode rapide...")
        simulation.run_simulation_fast(num_steps=args.steps)
    else:
        print("Exécution en mode standard...")
        simulation.run_simulation(num_steps=args.steps)

if __name__ == "__main__":
    main()
