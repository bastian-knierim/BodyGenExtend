import numpy as np


def rand_val():
    sign = np.random.choice([-1, 1])
    return sign * np.random.randint(1, 41)

def rand_cord_pair():
    coord1 = np.array([rand_val(), rand_val()])
    while True:
        coord2 = np.array([rand_val(), rand_val()])
        if np.any(np.sign(coord1) != np.sign(coord2)):
            return coord1, coord2
