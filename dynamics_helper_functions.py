"""
Equations related to converting between attitude representations and propagation
"""

import numpy as np
# WGS 84 defined coordinate system
a = 6378137
f_inverse = 298.257223563
e_squared = 6.69437999014e-3
GM = 3.986004418e14


def quat_multiply(p, q):
    output_q = np.zeros(4)
    output_q = np.array([p[0]*q[0]-p[1]*q[1]-p[2]*q[2]-p[3]*q[3],
                p[0]*q[1]+p[1]*q[0]+p[2]*q[3]-p[3]*q[2],
                p[0]*q[2]-p[1]*q[3]+p[2]*q[0]+p[3]*q[1],
                p[0]*q[3]+p[1]*q[2]-p[2]*q[1]+p[3]*q[0],
    ])
    return output_q

def quat_to_eul(quaternion):
    if quaternion.size > 4:
        print('quaternion set')
        q0, q1, q2, q3 = quaternion[:,0], quaternion[:,1], quaternion[:,2], quaternion[:,3]
    else:
        q0, q1, q2, q3 = quaternion[0], quaternion[1], quaternion[2], quaternion[3]
    phi = np.arctan2(2*(q0*q1 + q2*q3), (1-2*q1**2 - 2*q2**2))
    theta = np.arcsin(2*(q0*q2 - q1*q3))
    psi = np.arctan2(2*(q0*q3 + q1*q2), (1-2*q2**2 - 2*q3**2))
    return np.array([phi, theta, psi]).T

def eul_to_quat(euler_angles: np.ndarray) -> np.ndarray:
    if euler_angles.size > 3:
        phi, theta, psi = euler_angles[:,0], euler_angles[:,1], euler_angles[:,2]
    else:
        phi, theta, psi = euler_angles[0], euler_angles[1], euler_angles[2]
    q0 = np.cos(phi/2)*np.cos(theta/2)*np.cos(psi/2) + np.sin(phi/2)*np.sin(theta/2)*np.sin(psi/2)
    q1 = np.sin(phi/2)*np.cos(theta/2)*np.cos(psi/2) - np.cos(phi/2)*np.sin(theta/2)*np.sin(psi/2)
    q2 = np.cos(phi/2)*np.sin(theta/2)*np.cos(psi/2) + np.sin(phi/2)*np.cos(theta/2)*np.sin(psi/2)
    q3 = np.cos(phi/2)*np.cos(theta/2)*np.sin(psi/2) - np.sin(phi/2)*np.sin(theta/2)*np.cos(psi/2)
    q = np.array([q0, q1, q2, q3])
    q *= np.sign(q[0])
    return q

def q_update(q, omega_ib_b,dt):
    theta = omega_ib_b*dt
    # n_theta = np.linalg.norm(theta)  # If the rotation rate is too large, the small angle approximation does not hold
    dq = np.zeros(4)
    # if n_theta > 5E-3:
    #     dq[0] = np.cos(n_theta/2)
    #     dq[1:4] = theta.T/n_theta * np.sin(n_theta/2)
    #     q_out = quat_multiply(q, dq)
    #     return q_out
    # else:
    dq[0] = 1
    dq[1:4] = 0.5*theta
    q_out = quat_multiply(q,dq)
    return q_out / np.linalg.norm(q_out)

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


def CTM_to_quat(C):
    """
    :param C:
    :return:
    """
    r11, r12, r13 = C[0,:]
    r21, r22, r23 = C[1,:]
    r31, r32, r33 = C[2,:]
    q0 = 0.5 * np.sqrt(1 + r11 + r22 + r33)
    q1 = 0.5 * np.sqrt(1 + r11 - r22 - r33) * np.sign(r32 - r23)
    q2 = 0.5 * np.sqrt(1 - r11 + r22 - r33) * np.sign(r13 - r31)
    q3 = 0.5 * np.sqrt(1 - r11 - r22 + r33) * np.sign(r21 - r12)
    q = np.array([q0, q1, q2, q3])
    q *= np.sign(q[0])
    return q


def skew_symmetric(v):
    return np.array([[0, -v[2], v[1]],
                     [v[2], 0, -v[0]],
                     [-v[1], v[0], 0]])

def gravitation_acceleration_ECI(state):
    r = state[:3]
    return -GM / np.linalg.norm(r)**3*r


# def CTM_to_quat(C):
#     """
#
#     :param C:
#     :return:
#     """
#
#     r11, r12, r13 = C[0,:]
#     # r12 = C[0,1]
#     # r13 = C[0,2]
#     r21, r22, r23 = C[1,:]
#     # r22 = C[1,1]
#     # r23 = C[1,2]
#     r31, r32, r33 = C[2,:]
#     # r32 = C[2,1]
#     # r33 = C[2,2]
#     q0 = (
#         0.5*np.sqrt(1 + r11 + r22 + r33) if r11 + r22 + r33 > 0 else
#         0.5*np.sqrt(((r32 - r23)**2 + (r13 - r31)**2 + (r21 - r12)**2)/(3 - r11 - r22 - r33))
#     )
#     q1 = (
#         0.5 * np.sqrt(1 + r11 - r22 - r33) if r11 - r22 - r33 > 0 else
#         0.5 * np.sqrt(((r32 - r23)**2 + (r12 + r21)**2 + (r31 + r13)**2)/(3 - r11 + r22 + r33))
#     )
#     q2 = (
#         0.5 * np.sqrt(1 - r11 + r22 - r33) if -r11 + r22 - r33 > 0 else
#         0.5 * np.sqrt(((r13 - r31)**2 + (r12 + r21)**2 + (r23 + r32)**2)/(3 + r11 - r22 + r33))
#     )
#     q3 = (
#         0.5 * np.sqrt(1 - r11 - r22 + r33) if -r11 - r22 + r33 > 0 else
#         0.5 * np.sqrt(((r21 - r12)**2 + (r21 + r12)**2 + (r32 + r23)**2)/(3 + r11 + r22 - r33))
#     )
#     return np.array([q0, q1, q2, q3])