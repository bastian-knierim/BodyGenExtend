import numpy as np
import quaternion
import matplotlib.pyplot as plt

def add_rotation(q, np_random):
    if not isinstance(q, np.quaternion):
        q = array2quat(q)
    axes = ['x', 'y', 'z']
    axis = np_random.choice(axes)
    sign = np_random.choice([1, -1])
    half = sign * np.pi / 4
    if axis == 'x':
        q_rot = np.quaternion(np.cos(half), np.sin(half), 0, 0)
    elif axis == 'y':
        q_rot = np.quaternion(np.cos(half), 0, np.sin(half), 0)
    elif axis == 'z':
        q_rot = np.quaternion(np.cos(half), 0, 0, np.sin(half))

    q_new = q * q_rot
    q_new_unit = unit_quat(q_new)
    return q_new_unit

def unit_quat(q):
    if not isinstance(q, np.quaternion):
        q = array2quat(q)
    return q / np.linalg.norm([q.x, q.y, q.z, q.w])

def quat_dist(q1, q2):
    q1 = unit_quat(q1)
    q2 = unit_quat(q2)
    q1_conj = np.conjugate(q1)
    return q2 * q1_conj

def ang_dist(q1,q2):
    quat_dis = quat_dist(q1, q2)
    # theta = 2 arccos(q1*q2')
    return 2 * np.arccos(np.clip(np.abs(quat_dis.w),-1.0, 1.0))

def array2quat(q):
    return np.quaternion(*q)

def quat2array(q):
    return np.array([q.w, q.x, q.y, q.z])


if __name__ == '__main__':
    q = array2quat([1.0, 0.0, 0.0, 0.0])
    # Basisachsen
    axes = np.eye(3)  # x, y, z
    axes_prior = quaternion.rotate_vectors(q, axes)
    new_q = add_rotation(q, np.random)
    axes_rot = 0.5 * quaternion.rotate_vectors(new_q, axes)  # gedrehte Achsen
    print(f'print old rotation: {q} | new rotation: {new_q}')
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Originale Koordinatenachsen
    ax.quiver(0, 0, 0, *axes_prior[0], color='r', alpha=0.4)
    ax.quiver(0, 0, 0, *axes_prior[1], color='g', alpha=0.4)
    ax.quiver(0, 0, 0, *axes_prior[2], color='b', alpha=0.4)

    # Gedrehte Koordinatenachsen
    ax.quiver(0, 0, 0, *axes_rot[0], color='r', linewidth=2)
    ax.quiver(0, 0, 0, *axes_rot[1], color='g', linewidth=2)
    ax.quiver(0, 0, 0, *axes_rot[2], color='b', linewidth=2)

    ax.set_xlim([-1, 1])
    ax.set_ylim([-1, 1])
    ax.set_zlim([-1, 1])
    ax.set_box_aspect([1, 1, 1])
    ax.set_title("Original (transparent) vs. Rotated (bold)")
    plt.show()