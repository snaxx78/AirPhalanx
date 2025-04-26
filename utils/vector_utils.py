import numpy as np

def normalize_vector(v):
    # Vérifier les NaN
    if np.isnan(v).any():
        return np.zeros_like(v)
        
    # Calculer la norme en toute sécurité
    norm = np.linalg.norm(v)
    if norm > 1e-10:
        return v / norm
    else:
        return np.zeros_like(v)