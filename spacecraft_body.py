import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from shapely.geometry import Polygon

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



class Panel:
    def __init__(self, length: float, width: float, normal_vector: np.array, forward_vector: np.array, body: bool = True):
        A_inv = np.matrix(
            [[1, 0, 0],
             [0, 1, 0],
             [0, 0, 1]]
        )
        self.body = body
        self.length = length
        self.width = width
        self.n = normal_vector
        self.f = forward_vector
        self.body_normal_vector = np.array
        self.body_forward_vector = np.array
        self.body_nodes = np.array
        ### Panel origin in centre of panel ###
        self.local_nodes = np.array((
            [-self.length/2, self.width/2,  0], # rear left
            [-self.length/2, -self.width/2, 0], # rear right
            [ self.length/2, -self.width/2, 0], # front right
            [ self.length/2, self.width/2,  0], # front left
        ))
        n_cross_f = np.cross(self.n, self.f)
        B = np.matrix(np.vstack((self.f, n_cross_f, self.n))).T
        self.initial_forward_vector = np.array([1, 0, 0])
        self.initial_normal_vector = np.array([0, 0, 1])
        self.body_normal_vector = B @ self.initial_normal_vector
        self.body_forward_vector = B @ self.initial_forward_vector
        if body:  # Rotate about centre of panel (NOTE: potential flaw if body panels not perpendicular)
            self.local_nodes = (B @ self.local_nodes.T).T
        if not body:  # Rotate around hinge point
            self.local_nodes = self.local_nodes + np.array([-self.length/2, 0, 0]).T
            self.local_nodes = (B @ self.local_nodes.T).T

    def define_body_nodes(self, position_vector: np.array):
        self.body_nodes = np.array(self.local_nodes + position_vector)
        return

    def project_nodes_along_velocity_vector(self, projection_plane: list):

        return















class Satellite:
    def __init__(self, x_len: float, y_width: float, z_width: float, panel_length: float, panel_width: float,
                 panel_angles: np.array, inertia: np.array):
        self.x_len = x_len
        self.y_width = y_width
        self.z_width = z_width
        self.panel_length = panel_length
        self.panel_width = panel_width
        self.panel_angles = panel_angles.astype(float)
        self.normal_plane_velocity_vector = list
        self._velocity_vector_i = np.array([-1, 0, 0])
        self._C_ib = np.eye(3)


        for idx, panel_angle in enumerate(panel_angles):
            while panel_angle < 0 or panel_angle > 90:
                ValueError("Panel angle must be between 0 and 90 degrees, but is " + str(panel_angle) + " degrees.")
                panel_angle = float(input("Provide a new panel angle in degrees:"))
            self.panel_angles[idx] = np.deg2rad(panel_angle)
        self.com = np.array([-.5, 0, 0])
        self.inertia = inertia

        ######## CALCULATE PROJECTION PLANE WITH INCOMING PARTICLE VELOCITY VECTOR ########
        self.velocity_vector_b = self.C_ib @ self.velocity_vector_i
        self.calc_velocity_vector_normal_plane_basis()

        ######## CREATION OF BODY PANEL COORDINATES ########
        self.body_panel_centres = [
            np.array([0, 0, 0]), np.array([-self.x_len, 0, 0]),  # front, rear
            np.array([-self.x_len/2, self.y_width/2, 0]), np.array([-self.x_len/2, -self.y_width/2, 0]), # left, right
            np.array([-self.x_len/2, 0, self.z_width/2]), np.array([-self.x_len/2, 0, -self.z_width/2]), # top, bottom
        ]
        self.body_forward_vectors = [
            np.array([0, 0, 1]), np.array([0, 0, 1]), # point up
            np.array([1, 0, 0]), np.array([1, 0, 0]), # point forward
            np.array([1, 0, 0]), np.array([1, 0, 0]), # point forward
        ]
        self.body_normal_vectors = [
            np.array([1, 0, 0]), np.array([-1, 0,  0]),  # point forward, backward
            np.array([0, 1, 0]), np.array([0,  -1, 0]),  # point left, right
            np.array([0, 0, 1]), np.array([0,  0,  -1]),  # point up, down
        ]
        ######## CREATION OF REAR PANEL COORDINATES, WITH ABILITY TO DETERMINE ANGLE PER PANEL ########
        #TODO: There is likely room to make this code more succinct
        self.panel_hinges = [
            np.array([-x_len, 0, z_width / 2]),  # up
            np.array([-x_len, 0, -z_width / 2]),  # down
            np.array([-x_len, y_width / 2, 0]),  # left
            np.array([-x_len, -y_width / 2, 0]),  # right
        ]
        self.rear_forward_vectors = [
            np.array([np.cos(self.panel_angles[0]), 0, -np.sin(self.panel_angles[0])]), # upper rear panel
            np.array([np.cos(self.panel_angles[1]), 0, np.sin(self.panel_angles[1])]),  # lower rear panel
            np.array([np.cos(self.panel_angles[2]), -np.sin(self.panel_angles[2]), 0]), # left rear panel
            np.array([np.cos(self.panel_angles[3]), np.sin(self.panel_angles[3]), 0]),  # right rear panels
        ]
        self.rear_normal_vectors = [
            np.array([np.sin(self.panel_angles[0]), 0, np.cos(self.panel_angles[0])]), # upper rear panel
            np.array([np.sin(self.panel_angles[1]), 0, -np.cos(self.panel_angles[1])]),  # lower rear panel
            np.array([np.sin(self.panel_angles[2]), np.cos(self.panel_angles[2]), 0]), # left rear panel
            np.array([np.sin(self.panel_angles[3]), -np.cos(self.panel_angles[3]), 0]),  # right rear panels
        ]
        self.panels = []
        self.panel_nodes = []
        for idx, panel_center in enumerate(self.body_panel_centres):
            if idx < 2:  # front and back
                panel = Panel(self.y_width, self.z_width, self.body_normal_vectors[idx], self.body_forward_vectors[idx])
            elif 1 < idx < 4:  # sides
                panel = Panel(self.x_len, self.y_width, self.body_normal_vectors[idx], self.body_forward_vectors[idx])
            else:  # top and bottom
                panel = Panel(self.x_len, self.z_width, self.body_normal_vectors[idx], self.body_forward_vectors[idx])
            panel.define_body_nodes(panel_center)
            self.panels.append(panel)
            self.panel_nodes.append(panel.body_nodes)
        for idx, hinge_location in enumerate(self.panel_hinges):
            panel = Panel(self.panel_length, self.panel_width, self.rear_normal_vectors[idx],
                          self.rear_forward_vectors[idx], body=False)
            panel.define_body_nodes(hinge_location)
            self.panels.append(panel)
            self.panel_nodes.append(panel.body_nodes)


    def calc_velocity_vector_normal_plane_basis(self):
        velocity_unit_vector_b = self.velocity_vector_b / np.linalg.norm(self.velocity_vector_b)
        dummy_vector = np.eye(3)[np.argmin(np.abs(velocity_unit_vector_b))]
        v1 = np.cross(velocity_unit_vector_b, dummy_vector)
        v1 /= np.linalg.norm(v1)
        v2 = np.cross(velocity_unit_vector_b, v1)
        self.normal_plane_velocity_vector = [v1, v2, np.array([0, 0])]
        return


    def print_nodes(self):
        for nodes in self.panel_nodes:
            print(nodes)
        return

    def new_com(self):
        """
        Insert some calculator here to find the new center of mass
        """
        self.com = np.array([0, 0, 0])
        return

    ###### AUTOMATICALLY RE-CALCULATE THE NORMAL PLANE WHEN THE VELOCITY VECTOR CHANGES #######
    @property
    def velocity_vector_i(self):
        return self._velocity_vector_i

    @velocity_vector_i.setter
    def velocity_vector_i(self, new_velocity_vector_i):
        self._velocity_vector_i = new_velocity_vector_i
        self.velocity_vector_b = self.C_ib @ new_velocity_vector_i
        self.calc_velocity_vector_normal_plane_basis()

    @property
    def C_ib(self):
        return self._C_ib

    @C_ib.setter
    def C_ib(self, new_C_ib):
        self._C_ib = new_C_ib
        self.velocity_vector_b = new_C_ib @ self.velocity_vector_i







    def visualise(self, show_vectors: bool = True):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        for panel in self.panel_nodes:
            verts = [panel]
            panel_collection = Poly3DCollection(
                verts, facecolors='skyblue', edgecolors='k',
                linewidths=1, alpha=0.8
            )
            ax.add_collection3d(panel_collection)
            ax.scatter(panel[:, 0], panel[:, 1], panel[:, 2], color='r')

        if show_vectors:
            for panel, nodes in zip(self.panels, self.panel_nodes):
                center = nodes.mean(axis=0)
                n_vec = panel.n / np.linalg.norm(panel.n) * 0.3
                f_vec = panel.f / np.linalg.norm(panel.f) * 0.3
                ax.quiver(center[0], center[1], center[2],
                          n_vec[0], n_vec[1], n_vec[2],
                          color='b', arrow_length_ratio=0.2, linewidth=2)
                ax.quiver(center[0], center[1], center[2],
                          f_vec[0], f_vec[1], f_vec[2],
                          color='g', arrow_length_ratio=0.2, linewidth=2)

            legend_elements = [ Line2D([0], [0], color='b', lw=2, label='Normal vector'),
                                Line2D([0], [0], color='g', lw=2, label='Forward vector')]
            ax.legend(handles=legend_elements, loc="best")

        ax.set_xlabel('Length (x)')
        ax.set_ylabel('Width (y)')
        ax.set_zlabel('Height (z)')
        plt.title("Satellite configuration")
        ax.set_box_aspect([1, 1, 1])
        set_axes_equal(ax)
        plt.show()




