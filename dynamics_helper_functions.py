import numpy as np

def quat_multiply(p, q):
    output_q = np.zeros(4)
    output_q = np.array([p[0]*q[0]-p[1]*q[1]-p[2]*q[2]-p[3]*q[3],
                p[0]*q[1]+p[1]*q[0]+p[2]*q[3]-p[3]*q[2],
                p[0]*q[2]-p[1]*q[3]+p[2]*q[0]+p[3]*q[1],
                p[0]*q[3]+p[1]*q[2]-p[2]*q[1]+p[3]*q[0],
    ])
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

def q_update(q, omega_ib_b,dt):
    theta = omega_ib_b*dt
    n_theta = np.linalg.norm(theta)  # If the rotation rate is too large, the small angle approximation does not hold
    # print(f"n_theta = {n_theta} rad/timestep")
    dq = np.zeros(4)
    if n_theta > 5E-3:
        dq[0] = np.cos(n_theta/2)
        dq[1:4] = theta.T/n_theta * np.sin(n_theta/2)
        q_out = quat_multiply(q, dq)
        return q_out
    else:
        dq[0] = 1
        dq[1:4] = 0.5*theta
        q_out = quat_multiply(q,dq)
        return q_out / np.linalg.norm(q_out)



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
    # print(quaternion.shape)
    if quaternion.ndim == 2: # Batch input: shape (N, 4)
        q0, q1, q2, q3 = quaternion[:, 0], quaternion[:, 1], quaternion[:, 2], quaternion[:, 3]
        C = np.array([[q0 ** 2 + q1 ** 2 - q2 ** 2 - q3 ** 2, 2 * (q1 * q2 - q0 * q3), 2 * (q1 * q3 + q2 * q0)],
                      [2 * (q1 * q2 + q0 * q3), q0 ** 2 - q1 ** 2 + q2 ** 2 - q3 ** 2, 2 * (q2 * q3 - q0 * q1)],
                      [2 * (q1 * q3 - q2 * q0), 2 * (q2 * q3 + q1 * q0), q0 ** 2 - q1 ** 2 - q2 ** 2 + q3 ** 2],
                      ])
        C = np.rollaxis(C,2,0)
        return C
    else: # Single quaternion: shape (4,)
        q0, q1, q2, q3 = quaternion[0], quaternion[1], quaternion[2], quaternion[3]
        C = np.array([[q0**2+q1**2-q2**2-q3**2, 2*(q1*q2-q0*q3),         2*(q1*q3+q2*q0)],
                      [2*(q1*q2+q0*q3),         q0**2-q1**2+q2**2-q3**2, 2*(q2*q3-q0*q1)],
                      [2*(q1*q3-q2*q0),         2*(q2*q3+q1*q0),         q0**2-q1**2-q2**2+q3**2],
                      ])
        return C

def skew_symmetric(v):
    return np.array([[0, -v[2], v[1]],
                     [v[2], 0, -v[0]],
                     [-v[1], v[0], 0]])