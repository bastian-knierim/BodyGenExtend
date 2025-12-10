import numpy as np


def rand_val(np_random, range):
    sign = np_random.choice([-1, 1])
    return sign * np_random.randint(range[0], range[1])

def rand_cord_pair(np_random):    
    coord1 = np.array([rand_val(np_random), rand_val(np_random)])
    coord2 = np.array([rand_val(np_random), rand_val(np_random)])
    if np.linalg.norm(coord1) <= np.linalg.norm(coord2):
        return coord1, coord2
    else:
        return coord2, coord1

def rand_coord(np_random, range):    
    coord = np.array([rand_val(np_random, range), rand_val(np_random, range)])
    return coord

def random_unit_quat(np_random):
    u1, u2, u3 = np_random.rand(3)
    q = np.array([
        np.sqrt(1 - u1) * np.sin(2 * np.pi * u2),
        np.sqrt(1 - u1) * np.cos(2 * np.pi * u2),
        np.sqrt(u1) * np.sin(2 * np.pi * u3),
        np.sqrt(u1) * np.cos(2 * np.pi * u3)
    ])
    return q / np.linalg.norm(q)