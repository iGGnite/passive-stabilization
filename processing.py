import matplotlib.pyplot as plt
from dynamics_helper_functions import *
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
import numpy as np
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from spacecraft_body import CubeSat
from simulation import PassiveStabilization
from transformation_functions import a

# Create Earth
u = np.linspace(0, 2 * np.pi, 30)
v = np.linspace(0, np.pi, 30)
x_earth = a * np.outer(np.cos(u), np.sin(v))
y_earth = a * np.outer(np.sin(u), np.sin(v))
z_earth = a * np.outer(np.ones(np.size(u)), np.cos(v))


def set_axes_equal(ax):
    """
    Set 3D plot axes to have equal scale so that spheres appear as spheres,
    cubes as cubes, etc. This makes one unit along x, y or z look the same.
    """
    limits = np.array([
        ax.get_xlim3d(),
        ax.get_ylim3d(),
        ax.get_zlim3d(),
    ])
    center = np.mean(limits, axis=1)
    radius = 0.5 * np.max(limits[:, 1] - limits[:, 0])
    ax.set_xlim3d([center[0] - radius, center[0] + radius])
    ax.set_ylim3d([center[1] - radius, center[1] + radius])
    ax.set_zlim3d([center[2] - radius, center[2] + radius])


def plot_angular_rate(state):
    time = state[:,0]
    angular_rate = state[:,5:8]
    plt.title("Angular Rate body wrt inertial frame, expressed in body frame")
    plt.plot(time, angular_rate[:,0], label='omega_ib_b,x')
    plt.plot(time, angular_rate[:,1], label='omega_ib_b,y')
    plt.plot(time, angular_rate[:,2],'--', label='omega_ib_b,z')
    plt.plot(time, np.linalg.norm(angular_rate,axis=1),label='|omega_ib_b|')
    plt.legend()
    plt.xlabel('Time (s)')
    plt.ylabel('Angular rate (rad/s)')


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
        print(quaternions.shape)
        euler_angles = np.rad2deg(quat_to_eul(quaternions))
        print(euler_angles.shape)
        plt.title('Euler angles')
        plt.plot(time, euler_angles[:,0])
        plt.plot(time, euler_angles[:,1], '-.')
        plt.plot(time, euler_angles[:,2], '--')
        plt.legend(['Roll','Pitch','Yaw'])
        plt.ylabel('Euler Angle (deg)')
    plt.xlabel('Time (s)')

def plot_angular_momentum(inertia, state: np.ndarray):
    plt.figure()
    time = state[:,0]
    omega_ib_b = state[:,5:8]
    H = (inertia @ omega_ib_b.T).T
    plt.title("Instantaneous angular momentum based on body rates")
    plt.plot(time, H[:, 0], label='Angular momentum x axis')
    plt.plot(time, H[:, 1], label='Angular momentum y axis')
    plt.plot(time, H[:, 2], label='Angular momentum z axis')
    plt.plot(time, np.linalg.norm(H, axis=1), label='Total angular momentum')
    plt.xlabel('Time (s)')
    plt.ylabel('Angular momentum (kgm^2/s)')
    plt.legend()



def animate_rotations(sat, state: np.ndarray, spf: int = 10):
    """Animate orientation as a function of time

    :param sat: Satellite object to be rendered
        :type sat: CubeSat
    :param state: State of the CubeSat object
        :type state: np.ndarray
    :param spf: Number of seconds jump between each frame, standard 10s
        :type spf: int
    """
    time = state[:, 0]
    steps_per_second = int(spf/(time[1] - time[0]))
    rotations = quat_to_CTM(state[:, 1:5])  # shape: (N, 3, 3)
    incoming_flow = -state[:, 11:14]
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
        ax.set_box_aspect([1, 1, 1])
        set_axes_equal(ax)
        R = rotations[frame]
        for panel in sat.panel_vertices:  # Draw the panels which make up the cubesat
            vertices = [(R @ (panel - sat.com).T).T]
            panel_collection = Poly3DCollection(vertices, facecolors='skyblue', edgecolors='k', linewidths=.5,
                                                alpha=0.3)
            ax.add_collection3d(panel_collection)
        particle_vec = incoming_flow[frame]
        ax.quiver(*(-particle_vec)/np.linalg.norm(particle_vec),
                  *particle_vec/np.linalg.norm(particle_vec)/2,
                  color='black',label='Incoming flow')
        ax.set_title(f"Time: {time[frame]:.2f}s")
        ax.legend()

    ani = FuncAnimation(fig, update, frames=int(len(time)/sample), interval=50)
    plt.show()


def plot_torque(impact_data: np.ndarray):
    time = impact_data[:,0]
    fig, ax = plt.subplots(4, sharex=True)
    fig.suptitle('Instantaneous Torque')
    ax[0].plot(time, impact_data[:,1], color='tab:blue',   label='Torque body x axis')
    ax[1].plot(time, impact_data[:,2], color='tab:orange', label='Torque body y axis')
    ax[2].plot(time, impact_data[:,3], color='tab:green', label='Torque body z axis')
    ax[3].plot(time, np.linalg.norm(impact_data[:,1:4], axis=1),
               color='tab:red', label='Total torque')
    for a in ax:
        a.set_ylabel('Torque (Nm)')
    fig.supxlabel('Time (s)')
    fig.legend()


def plot_kinematics(impact_data: np.ndarray):
    time = impact_data[:,0]
    fig, ax = plt.subplots(3, sharex=True)
    fig.suptitle('Instantaneous Kinematics')
    ax[0].plot(time, impact_data[:,4], color='tab:blue',   label='Kinematic  x axis')
    ax[1].plot(time, impact_data[:,5], color='tab:orange', label='Kinematic dev y axis')
    ax[2].plot(time, impact_data[:,6], color='tab:green', label='Kinematic dev z axis')
    for a in ax:
        a.set_ylabel('Torque-like (Nm)')
    fig.supxlabel('Time (s)')
    fig.legend()

def plot_impacts(impact_data):
    time = impact_data[:,0]
    gen, imp = impact_data[:,7], impact_data[:,8]

    fig, ax1 = plt.subplots()
    fig.suptitle('Generated particles and number of impacts')
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


def plot_orbit(state: np.ndarray):
    time = state[:,0]
    pos = state[:,8:11]
    max_pos = pos.max()
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d', computed_zorder=False)

    ax.plot_surface(x_earth, y_earth, z_earth, zorder=0)
    ax.plot(pos[:,0], pos[:,1], pos[:,2], color='tab:red',zorder=1) # Always plot the orbit over the earth

    ax.set_xlim([-max_pos, max_pos])
    ax.set_ylim([-max_pos, max_pos])
    ax.set_zlim([-max_pos, max_pos])
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_box_aspect([1, 1, 1])
    set_axes_equal(ax)



def animate_orbit(state: np.ndarray, spf: int = 10):
    """
    Animate the orbit (and the inertial attitude) as a function of time
    :param state:
        :type state: np.ndarray
    :param spf: Number of seconds jump between each frame, standard 10s
        :type spf: int
    :return:
    """
    time = state[:, 0]
    pos = state[:,8:11]
    max_pos = pos.max()
    steps_per_second = int(spf/(time[1] - time[0]))
    rotations = quat_to_CTM(state[:, 1:5])  # shape: (N, 3, 3), inertial to body frame
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d',computed_zorder=False)
    ax.set_xlim([-max_pos, max_pos])
    ax.set_ylim([-max_pos, max_pos])
    ax.set_zlim([-max_pos, max_pos])
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    sample = int(steps_per_second)
    def update(frame):
        frame = sample*frame
        ax.cla()  # Clears the whole axes safely

        ax.plot_surface(x_earth, y_earth, z_earth, zorder=0)
        ax.plot(pos[:frame, 0], pos[:frame, 1], pos[:frame, 2], color='tab:orange', zorder=1)  # Always plot the orbit over the earth
        ax.scatter(pos[frame,0], pos[frame,1], pos[frame,2], color='tab:red', zorder=2)
        ax.set_xlim([-max_pos, max_pos])
        ax.set_ylim([-max_pos, max_pos])
        ax.set_zlim([-max_pos, max_pos])
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_box_aspect([1, 1, 1])
        set_axes_equal(ax)
        vec_size = 1500000
        R = rotations[frame]
        ax.quiver(*pos[frame,:], *(R[:,0] * vec_size), color='red', arrow_length_ratio=0.2, linewidth=2)
        ax.quiver(*pos[frame,:], *(R[:,1] * vec_size), color='green', arrow_length_ratio=0.2, linewidth=2)
        ax.quiver(*pos[frame,:], *(R[:,2] * vec_size), color='blue', arrow_length_ratio=0.2, linewidth=2)
        ax.plot([], [], [], color="red", label="Body x")
        ax.plot([], [], [], color="green", label="Body y")
        ax.plot([], [], [], color="blue", label="Body z")
        ax.set_title(f"Time: {time[frame]:.2f}s")
        ax.legend()

    ani = FuncAnimation(fig, update, frames=int(len(time)/sample), interval=50)
    plt.show()



###### VISUALISE CUBESAT ######
def visualise(CubeSat,
              show_center_of_mass: bool = True,
              show_velocity_vector: bool = False,
              show_panel_vectors: bool = False,
              show_shadow_axis_system: bool = False,
              show_particle_vectors: bool = False,
              highlight_nodes: bool = False,
              impacts: np.ndarray = None,
              particle_vectors: list[np.ndarray] = None,
              p_at_impact_vectors: np.ndarray = None,
              points_in_projection: np.ndarray = None,
              projection_borders: bool = False, ):
    """
    Function to create 3D plot of CubeSat. Very useful for debugging purposes.

    3D render CubeSat in body frame, with various options to show. Can show geometric, mass, or construction features;
    or particle generation location, impact location, momentum transfer vector, etcetera.
    :param show_center_of_mass: Visualise location of center of mass in spacecraft body.
    :param show_velocity_vector: Visualise direction of incoming particles.
    :param show_panel_vectors: Visualise normal and forward vectors of each panel.
    :param show_shadow_axis_system: Render the axis system created in the plane normal to the velocity vector, with which particles are generated.
    :param show_particle_vectors: Render direction of particles. Renders identical vectors if none are provided with 'particle_vectors'.
    :param highlight_nodes: Highlight the vertices making up the CubeSat body.
    :param impacts: Visualise impact locations by providing 3D coordinates of impacts.
    :param particle_vectors: Individual direction vectors for each particle. Useful for particles from different directions.
    :param p_at_impact_vectors: Render vectors of momentum transfer to spacecraft.
    :param points_in_projection: Render points generated in velocity normal plane.
    :param projection_borders: Render plane in which particles are generated.
    """
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    for panel in CubeSat.panel_vertices:  # Draw the panels which make up the cubesat
        vertices = [panel]
        panel_collection = Poly3DCollection(vertices, facecolors='skyblue', edgecolors='k', linewidths=.5,
                                            alpha=0.3)
        ax.add_collection3d(panel_collection)
        if highlight_nodes:
            ax.scatter(panel[:, 0], panel[:, 1], panel[:, 2], color='r')

    if show_center_of_mass:
        ax.scatter(CubeSat.com[0], CubeSat.com[1], CubeSat.com[2], color='orange')

    if show_velocity_vector:
        vel_vec = CubeSat.particle_velocity_vector_b / np.linalg.norm(CubeSat.particle_velocity_vector_b) / 2
        ax.quiver(0 - vel_vec[0], 0 - vel_vec[1], 0 - vel_vec[2], vel_vec[0], vel_vec[1], vel_vec[2], color='black')

    if show_panel_vectors:
        for panel, nodes in zip(CubeSat.panels, CubeSat.panel_vertices):
            center = nodes.mean(axis=0)
            n_vec = panel.body_normal_vector / np.linalg.norm(panel.body_normal_vector) * 0.3
            f_vec = panel.body_forward_vector / np.linalg.norm(panel.body_forward_vector) * 0.3
            ax.quiver(center[0], center[1], center[2],
                      n_vec[0], n_vec[1], n_vec[2],
                      color='b', arrow_length_ratio=0.2, linewidth=2)
            ax.quiver(center[0], center[1], center[2],
                      f_vec[0], f_vec[1], f_vec[2],
                      color='g', arrow_length_ratio=0.2, linewidth=2)
        legend_elements = [Line2D([0], [0], color='b', lw=2, label='Panel normal vector'),
                           Line2D([0], [0], color='g', lw=2, label='Panel forward vector')]
        ax.legend(handles=legend_elements, loc="best")

    if show_shadow_axis_system:  # Visualise the axis system basis generated for the 2D CubeSat shadow
        x_vec, y_vec, z_vec, origin = CubeSat.shadow_projection_axis_system
        axis_length = 0.5  # scale for visibility
        ax.quiver(*origin, *(x_vec * axis_length), color='darkred', arrow_length_ratio=0.2, linewidth=2)
        ax.quiver(*origin, *(y_vec * axis_length), color='darkgreen', arrow_length_ratio=0.2, linewidth=2)
        ax.quiver(*origin, *(z_vec * axis_length), color='darkblue', arrow_length_ratio=0.2, linewidth=2)

    if particle_vectors is not None and impacts is not None:
        if len(particle_vectors) != len(impacts):
            raise ValueError("Number of specified impacts and particle vectors must match")
    if impacts is not None:  # Plot the impact locations of particles with an arrow indicating the particle direction
        vec_x_dir, vec_y_dir, vec_z_dir = CubeSat.particle_velocity_vector_b / np.linalg.norm(
            CubeSat.particle_velocity_vector_b) / 5
        for idx in range(impacts.shape[0]):
            impact_coords = impacts[idx, :]  #TODO: VECTORIZE
            ax.scatter(impact_coords[0], impact_coords[1], impact_coords[2], marker='x', color='r')
            if particle_vectors is not None and show_particle_vectors is True:  # If the particles have a specified direction, overwrite
                particle_vector = particle_vectors[idx]
                vec_x_dir, vec_y_dir, vec_z_dir = particle_vector / np.linalg.norm(particle_vector)
                ax.quiver(impact_coords[0] - vec_x_dir, impact_coords[1] - vec_y_dir, impact_coords[2] - vec_z_dir,
                          vec_x_dir, vec_y_dir, vec_z_dir)
            if p_at_impact_vectors is not None:
                p_vector = p_at_impact_vectors[idx]
                vec_x_dir, vec_y_dir, vec_z_dir = p_vector * 1e4
                ax.quiver(impact_coords[0], impact_coords[1], impact_coords[2],
                          vec_x_dir, vec_y_dir, vec_z_dir)
    if points_in_projection is not None:
        ax.scatter(points_in_projection[:, 0], points_in_projection[:, 1], points_in_projection[:, 2], marker='o',
                   color='purple')

    if projection_borders:
        vertices = np.array([[-CubeSat.max_dist_from_geom_center, -CubeSat.max_dist_from_long_axis],
                             [-CubeSat.max_dist_from_geom_center, CubeSat.max_dist_from_long_axis],
                             [CubeSat.max_dist_from_geom_center, CubeSat.max_dist_from_long_axis],
                             [CubeSat.max_dist_from_geom_center, -CubeSat.max_dist_from_long_axis],
                             ])
        print(vertices[:, 0])
        vertices = ((vertices[:, 0] * (CubeSat.shadow_projection_axis_system[0])[:, None] +
                     vertices[:, 1] * (CubeSat.shadow_projection_axis_system[1])[:, None]).T +
                    CubeSat.shadow_projection_axis_system[3])
        print(vertices)
        ax.add_collection3d(
            Poly3DCollection([vertices], facecolors='cyan', linewidths=1, edgecolors='r', alpha=0.5))
    ax.set_xlabel('Length (x)')
    ax.set_ylabel('Width (y)')
    ax.set_zlabel('Height (z)')
    ax.set_xlim(-(2 * CubeSat.x_len + .2), 0.5)
    ax.set_ylim(-(CubeSat.y_width + .5), (CubeSat.y_width + .5))
    ax.set_zlim(-(CubeSat.z_width + .5), (CubeSat.z_width + .5))
    plt.title("CubeSat configuration")
    ax.set_box_aspect([1, 1, 1])
    set_axes_equal(ax)
    plt.show()

def show_attitude(sim: PassiveStabilization):
    aero_axes = np.eye(3)
    body_axes = sim.R_inertial_to_body @ np.eye(3)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.quiver(0, 0, 0, body_axes[:, 0], body_axes[:, 1], body_axes[:, 2], color=['red', 'green', 'blue'])
    ax.quiver(0, 0, 0, aero_axes[:, 0], aero_axes[:, 1], aero_axes[:, 2],
              color=['orange', 'darkgreen', 'lightblue'])
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    plt.show()

def visualise_3d(sim: PassiveStabilization,
                 show_velocity_vector: bool = False,
                 impacts: np.ndarray = None,
                 p_at_impacts: np.ndarray = None,
                 points_in_projection: np.ndarray = None, ):
    """Provide 3D visualisation of CubeSat body, with options for additional phenomena like impacts and momentum
    exchange
    :param: sim: PassiveStabilization object which is to be visualized
        :type sim: PassiveStabilization object
    :param show_velocity_vector: Whether to show the velocity vector
        :type show_velocity_vector: bool
    :param impacts: Numpy array containing location of particle impacts
        :type impacts: np.ndarray
    :param p_at_impacts: Numpy array containing vectors representing momentum transfer at particle impacts
        :type p_at_impacts: np.ndarray
    :param points_in_projection: Numpy array containing points generated on projection plane
        :type points_in_projection: np.ndarray
    :return:
    """
    visualise(sim.sat,show_velocity_vector=False, impacts=None, show_shadow_axis_system=False,
                       p_at_impact_vectors=p_at_impacts, points_in_projection=points_in_projection)
    plt.show()