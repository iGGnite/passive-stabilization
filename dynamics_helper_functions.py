import numpy as np

def quat_multiply(q, p):
    output_q = np.zeros(4)
    output_q[0] = q[0]*p[0] - np.dot(q[1:], p[1:])
    output_q[1:] = q[0]*p[1:] + p[0]*q[1:] + np.cross(q[1:], p[1:])
    return output_q

def eul_to_quat(euler_angles: np.ndarray) -> np.ndarray:
    if euler_angles.size > 3:
        phi, theta, psi = euler_angles[:,0], euler_angles[:,1], euler_angles[:,2]
    else:
        phi, theta, psi = euler_angles[0], euler_angles[1], euler_angles[2]
    q0 = np.cos(phi/2)*np.cos(theta/2)*np.cos(psi/2) + np.sin(phi/2)*np.sin(theta/2)*np.sin(psi/2)
    q1 = np.sin(phi/2)*np.cos(theta/2)*np.cos(psi/2) - np.cos(phi/2)*np.sin(theta/2)*np.sin(psi/2)
    q2 = np.cos(phi/2)*np.sin(theta/2)*np.cos(psi/2) + np.sin(phi/2)*np.cos(theta/2)*np.sin(psi/2)
    q3 = np.cos(phi/2)*np.cos(theta/2)*np.sin(psi/2) - np.sin(phi/2)*np.sin(theta/2)*np.cos(psi/2)
    return np.array([q0, q1, q2, q3])

def q_dot(q, omega):
    omega_vec = np.zeros(4)
    omega_vec[1:] = omega
    return 0.5 * quat_multiply(q, omega_vec)

def quat_to_eul(quaternion):
    if quaternion.size > 4:
        q0, q1, q2, q3 = quaternion[:,0], quaternion[:,1], quaternion[:,2], quaternion[:,3]
    else:
        q0, q1, q2, q3 = quaternion[0], quaternion[1], quaternion[2], quaternion[3]
    phi = np.arctan2(2*(q0*q1 + q2*q3), (1-2*q1**2 - 2*q2**2))
    theta = np.arcsin(2*(q0*q2 - q1*q3))
    psi = np.arctan2(2*(q0*q3 + q1*q2), (1-2*q2**2 - 2*q3**2))
    return np.array([phi, theta, psi]).T

def quat_to_CTM(quaternion):
    if quaternion.size > 4:
        q0, q1, q2, q3 = quaternion[:,0], quaternion[:,1], quaternion[:,2], quaternion[:,3]
    else:
        q0, q1, q2, q3 = quaternion[0], quaternion[1], quaternion[2], quaternion[3]
    C = np.array([q0**2+q1**2-q2**2-q3**2, 2*(q1*q2+q0*q3), 2*(q1*q3-q2*q0),
                  2*(q1*q2-q0*q3), q0**2-q1**2+q2**2-q3**2, 2*(q2*q3+q0*q1),
                  2*(q1*q3+q2*q0), 2*(q2*q3-q1*q0), q0**2-q1**2-q2**2+q3**2,
                  ]).reshape(-1,3,3).squeeze()
    print(C.shape)
    return C