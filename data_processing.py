import matplotlib.pyplot as plt
from dynamics_helper_functions import *
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
import numpy as np

def plot_angular_rate(state):
    time = state[:,0]
    angular_rate = state[:,1:4]
    plt.plot(time, angular_rate[:,0])
    plt.plot(time, angular_rate[:,1])
    plt.plot(time, angular_rate[:,2])
    plt.legend(['p','q','r'])
    # plt.show()


def plot_attitude(state, quaternion = False):
    time = state[:,0]
    quaternions = state[:,4:8]
    if quaternion:
        plt.plot(time, quaternions[:,0])
        plt.plot(time, quaternions[:,1])
        plt.plot(time, quaternions[:,2])
        plt.plot(time, quaternions[:,3])
    else:
        euler_angles = np.rad2deg(quat_to_eul(quaternions))
        print(euler_angles.shape)
        plt.plot(time, euler_angles[:,0])
        plt.plot(time, euler_angles[:,1])
        plt.plot(time, euler_angles[:,2])
        plt.legend(['Roll','Pitch','Yaw'])
    # plt.show()




def animate_rotations(state, time=None):
    time = state[:, 0] if time is None else time
    rotations = quat_to_CTM(state[:, 4:8])  # shape: (N, 3, 3)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlim([-1, 1])
    ax.set_ylim([-1, 1])
    ax.set_zlim([-1, 1])
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    def update(frame):
        ax.cla()  # Clears the whole axes safely
        ax.set_xlim([-1, 1])
        ax.set_ylim([-1, 1])
        ax.set_zlim([-1, 1])
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        R = rotations[frame]
        for vec, color in zip(R.T, ['r', 'g', 'b']):
            ax.quiver(0, 0, 0, *vec, color=color)
        ax.set_title(f"Time: {time[frame]:.2f}s")

    ani = FuncAnimation(fig, update, frames=len(time), interval=5)
    plt.show()