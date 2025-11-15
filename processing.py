import matplotlib.pyplot as plt
from dynamics_helper_functions import *
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
import numpy as np
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

def plot_angular_rate(state):
    time = state[:,0]
    angular_rate = state[:,1:4]
    plt.plot(time, angular_rate[:,0])
    plt.plot(time, angular_rate[:,1])
    plt.plot(time, angular_rate[:,2])
    plt.plot(time, np.linalg.norm(angular_rate,axis=1))
    plt.legend(['omega_ib_b,x','omega_ib_b,y','omega_ib_b,z','|omega_ib_b|'])
    plt.xlabel('Time (s)')
    # plt.show()


def plot_attitude(state, quaternion = False):
    plt.figure()
    time = state[:,0]
    quaternions = state[:,4:8]
    if quaternion:
        plt.plot(time, quaternions[:,0])
        plt.plot(time, quaternions[:,1])
        plt.plot(time, quaternions[:,2])
        plt.plot(time, quaternions[:,3])
        plt.legend(['q0','q1','q2','q3'])
        plt.ylabel('Quaternion (-)')
    else:
        euler_angles = np.rad2deg(quat_to_eul(quaternions))
        print(euler_angles.shape)
        plt.plot(time, euler_angles[:,0])
        plt.plot(time, euler_angles[:,1])
        plt.plot(time, euler_angles[:,2])
        plt.legend(['Roll','Pitch','Yaw'])
        plt.ylabel('Euler Angle (deg)')
    plt.xlabel('Time (s)')
    # plt.show()

def plot_angular_momentum(inertia, state: np.ndarray):
    time = state[:,0]
    omega_ib_b = state[:,1:4]
    print(omega_ib_b.shape)
    H = (inertia @ omega_ib_b.T).T
    print(H.shape)
    plt.plot(time, H[:, 0])
    plt.plot(time, H[:, 1])
    plt.plot(time, H[:, 2])
    plt.plot(time, np.linalg.norm(H, axis=1))
    plt.xlabel('Time (s)')
    plt.ylabel('Angular momentum (kgm^2/s)')



def animate_rotations(sat, state, time=None, ):
    time = state[:, 0] if time is None else time
    steps_per_second = int(1/(time[1] - time[0]))

    rotations = quat_to_CTM(state[:, 4:8])  # shape: (N, 3, 3)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlim([-1, .5])
    ax.set_ylim([-.75, .75])
    ax.set_zlim([-.75, .75])
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    sample = int(steps_per_second)
    def update(frame):
        frame = sample*frame
        ax.cla()  # Clears the whole axes safely
        ax.set_xlim([-1, .5])
        ax.set_ylim([-.75, .75])
        ax.set_zlim([-.75, .75])
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        R = rotations[frame]
        for panel in sat.panel_vertices:  # Draw the panels which make up the cubesat
            vertices = [(R @ (panel - sat.com).T).T]
            panel_collection = Poly3DCollection(vertices, facecolors='skyblue', edgecolors='k', linewidths=.5,
                                                alpha=0.3)
            ax.add_collection3d(panel_collection)
        # for vec, color in zip(R.T, ['r', 'g', 'b']):
        #     ax.quiver(0, 0, 0, *vec/2, color=color)
        ax.set_title(f"Time: {time[frame]:.2f}s")

    ani = FuncAnimation(fig, update, frames=int(len(time)/sample), interval=50)
    plt.show()