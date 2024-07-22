import numpy as np


# Normalise points in the unit sphere
def unit_sphere_normalisation(post_processed_points):
    center = np.mean(post_processed_points, axis=0)
    centered_points = post_processed_points - center
    radius = np.max(np.linalg.norm(centered_points, axis=1))
    normalised_points = centered_points / radius
    return normalised_points
