import numpy as np
from sympy.algebras import quaternion

from dynamics_helper_functions import *

eul = np.deg2rad(np.array([
    [0,0,0],
    [15,15,15],
    [0,15,15],
    [30,15,0],
    ])
)

quats = np.array([eul_to_quat(att) for att in eul])
# print(quats)
q0, q1, q2, q3 = quats[:,0], quats[:,1], quats[:,2], quats[:,3]
# print((q0 ** 2 + q1 ** 2 - q2 ** 2 - q3 ** 2).shape)
C = np.rollaxis(np.array([[q0 ** 2 + q1 ** 2 - q2 ** 2 - q3 ** 2, 2 * (q1 * q2 - q0 * q3), 2 * (q1 * q3 + q2 * q0)],
              [2 * (q1 * q2 + q0 * q3), q0 ** 2 - q1 ** 2 + q2 ** 2 - q3 ** 2, 2 * (q2 * q3 - q0 * q1)],
              [2 * (q1 * q3 - q2 * q0), 2 * (q2 * q3 + q1 * q0), q0 ** 2 - q1 ** 2 - q2 ** 2 + q3 ** 2],
              ]),2,0)
print(C.shape)
print(C[3])
print(quat_to_CTM(quats[3]))