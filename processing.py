import matplotlib.pyplot as plt
from dynamics_helper_functions import *
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
import numpy as np
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

def plot_angular_rate(state):
    time = state[:,0]
    angular_rate = state[:,5:8]
    plt.title("Angular Rate body wrt inertial frame, expressed in body frame")
    plt.plot(time, angular_rate[:,0])
    plt.plot(time, angular_rate[:,1])
    plt.plot(time, angular_rate[:,2],'--')
    plt.plot(time, np.linalg.norm(angular_rate,axis=1))
    plt.legend(['omega_ib_b,x','omega_ib_b,y','omega_ib_b,z','|omega_ib_b|'])
    plt.xlabel('Time (s)')
    plt.ylabel('Angular rate (rad/s)')
    plt.show()


def plot_attitude(state, quaternion = False):
    plt.figure()
    time = state[:,0]
    quaternions = state[:,1:5]
    if quaternion:
        plt.plot(time, quaternions[:,0])
        plt.plot(time, quaternions[:,1])
        plt.plot(time, quaternions[:,2])
        plt.plot(time, quaternions[:,3])
        plt.legend(['q0','q1','q2','q3'])
        plt.ylabel('Quaternion (-)')
    else:
        euler_angles = np.rad2deg(quat_to_eul(quaternions))
        # print(euler_angles.shape)
        plt.title('Euler angles')
        plt.plot(time, euler_angles[:,0])
        plt.plot(time, euler_angles[:,1])
        plt.plot(time, euler_angles[:,2], '--')
        plt.legend(['Roll','Pitch','Yaw'])
        plt.ylabel('Euler Angle (deg)')
    plt.xlabel('Time (s)')
    # plt.show()

def plot_angular_momentum(inertia, state: np.ndarray):
    time = state[:,0]
    omega_ib_b = state[:,1:4]
    H = (inertia @ omega_ib_b.T).T
    plt.title("Instantaneous angular momentum based on body rates")
    plt.plot(time, H[:, 0], label='Angular momentum x axis')
    plt.plot(time, H[:, 1], label='Angular momentum y axis')
    plt.plot(time, H[:, 2], label='Angular momentum z axis')
    plt.plot(time, np.linalg.norm(H, axis=1), label='Total angular momentum')
    plt.xlabel('Time (s)')
    plt.ylabel('Angular momentum (kgm^2/s)')
    plt.legend()
    plt.show()



def animate_rotations(sat, state, time=None, spf: int = 10):
    time = state[:, 0] if time is None else time
    steps_per_second = int(spf/(time[1] - time[0]))
    rotations = quat_to_CTM(state[:, 1:5])  # shape: (N, 3, 3)
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
        ax.quiver(1, 0, 0, -.5, 0, 0,color='black',label='Incoming flow')
        ax.set_title(f"Time: {time[frame]:.2f}s")
        ax.legend()

    ani = FuncAnimation(fig, update, frames=int(len(time)/sample), interval=50)
    plt.show()


def plot_torque(state: np.ndarray):
    time = state[:,0]
    fig, ax = plt.subplots(4, sharex=True)
    fig.suptitle('Instantaneous Torque')
    ax[0].plot(time, state[:,8], color='tab:blue',   label='Torque body x axis')
    ax[1].plot(time, state[:,9], color='tab:orange', label='Torque body y axis')
    ax[2].plot(time, state[:,10], color='tab:green', label='Torque body z axis')
    ax[3].plot(time, np.linalg.norm(state[:,8:11], axis=1),
               color='tab:red', label='Total torque')
    for a in ax:
        a.set_ylabel('Torque (Nm)')
    fig.supxlabel('Time (s)')
    fig.legend()
    plt.show()


def plot_impacts(impacts):
    time = impacts[:,0]
    gen, imp = impacts[:,1], impacts[:,2]

    fig, ax1 = plt.subplots()
    ax1.plot(time, gen, label='Generated')
    ax1.plot(time, imp, label='Impacted')
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Particles (-)')
    ax1.legend(loc='upper left')

    ax2 = ax1.twinx()
    ax2.plot(time, 100*imp/gen, 'g--', label='% Impacted')
    ax2.set_ylim(0, 100)
    ax2.set_ylabel('Percentage of generated impacted (%)')
    ax2.legend(loc='upper right')
    plt.show()
