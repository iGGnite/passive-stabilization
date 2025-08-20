import numpy as np
import matplotlib.pyplot as plt
import mapbox_earcut as earcut
from matplotlib.lines import Line2D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from shapely.geometry import Polygon
from shapely.set_operations import unary_union


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
        self.body = body
        self.length = length
        self.width = width
        self.n = normal_vector
        self.f = forward_vector
        self.body_normal_vector = np.array
        self.body_forward_vector = np.array
        self.body_nodes = np.array
        self.polygon = Polygon
        ### Panel origin in centre of panel ###
        self.local_nodes = np.array((
            [-self.length/2, self.width/2,  0], # rear left
            [-self.length/2, -self.width/2, 0], # rear right
            [ self.length/2, -self.width/2, 0], # front right
            [ self.length/2, self.width/2,  0], # front left
        ))
        n_cross_f = np.cross(self.n, self.f)
        B = np.array(np.vstack((self.f, n_cross_f, self.n))).T
        self.initial_forward_vector = np.array([1, 0, 0])
        self.initial_normal_vector = np.array([0, 0, 1])
        self.body_normal_vector = B @ self.initial_normal_vector
        self.body_forward_vector = B @ self.initial_forward_vector
        self.panel_center = np.ndarray
        if body:  # Rotate about centre of panel (NOTE: potential flaw if body panels not perpendicular)
            self.local_nodes = (B @ self.local_nodes.T).T
        if not body:  # Rotate around hinge point
            self.local_nodes = self.local_nodes + np.array([-self.length/2, 0, 0]).T
            self.local_nodes = (B @ self.local_nodes.T).T
        # self.projection_plane = list
    def define_body_nodes(self, position_vector: np.ndarray):
        self.body_nodes = np.array(self.local_nodes + position_vector)
        self.panel_center = np.mean(self.body_nodes, axis=0)
        return

    def projected_polygon(self, projection_plane: list)->Polygon:
        x = np.dot((self.body_nodes - projection_plane[-1]), projection_plane[0])
        y = np.dot((self.body_nodes - projection_plane[-1]), projection_plane[1])
        projected_coords = np.vstack((x,y)).T
        self.polygon = Polygon(projected_coords)
        # print(f"area: {self.polygon.area}")
        return self.polygon



class Satellite:
    def __init__(self, x_len: float, y_width: float, z_width: float, panel_length: float, panel_width: float,
                 rear_panel_angles: np.ndarray, inertia: np.ndarray):
        self.x_len = x_len
        self.y_width = y_width
        self.z_width = z_width
        self.panel_length = panel_length
        self.panel_width = panel_width
        self._panel_angles = rear_panel_angles.astype(float)
        self._velocity_vector_i = np.array([-1, 0, 0])  # Velocity vector of the incoming airflow in the inertial frame
        self._C_ib = np.eye(3)  # Velocity vector of the incoming airflow in the body frame
        self.panel_polygons = [None] * 10
        self.shadow: Polygon = None # type: ignore
        self.shadow_triangle_coords = None
        self.shadow_triangle_areas: np.ndarray = np.array
        self.rear_forward_vectors: list[np.ndarray | None] = [None] * 4 # type: ignore
        self.rear_normal_vectors: list[np.ndarray | None] = [None] * 4  # type: ignore
        self.panels: list[Panel | None] = [None] * 10
        self.panel_nodes = [np.ndarray] * 10

        #TODO: may be unnecessarily expensive to run this
        for idx, panel_angle in enumerate(rear_panel_angles):
            while panel_angle < 0 or panel_angle > 90:
                ValueError("Panel angle must be between 0 and 90 degrees, but is " + str(panel_angle) + " degrees.")
                panel_angle = float(input("Provide a new panel angle in degrees:"))
            self._panel_angles[idx] = np.deg2rad(panel_angle)
        self.com = np.array([-.5, 0, 0])
        self.inertia = inertia

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
        self.panel_hinges = [
            np.array([-x_len, 0, z_width / 2]),  # up
            np.array([-x_len, 0, -z_width / 2]),  # down
            np.array([-x_len, y_width / 2, 0]),  # left
            np.array([-x_len, -y_width / 2, 0]),  # right
        ]
        #TODO: There is likely room to make this code more succinct
        for idx, panel_center in enumerate(self.body_panel_centres):
            if idx < 2:  # front and back
                panel = Panel(self.y_width, self.z_width, self.body_normal_vectors[idx], self.body_forward_vectors[idx])
            elif 1 < idx < 4:  # sides
                panel = Panel(self.x_len, self.y_width, self.body_normal_vectors[idx], self.body_forward_vectors[idx])
            else:  # top and bottom
                panel = Panel(self.x_len, self.z_width, self.body_normal_vectors[idx], self.body_forward_vectors[idx])
            panel.define_body_nodes(panel_center)
            self.panels[idx] = panel
            self.panel_nodes[idx] = panel.body_nodes  # type: ignore

        ######## CREATION OF REAR PANEL COORDINATES, WITH ABILITY TO DETERMINE ANGLE PER PANEL ########
        self.create_rear_panels()

        ######## CALCULATE PROJECTION PLANE WITH INCOMING PARTICLE VELOCITY VECTOR ########
        self.velocity_vector_b = self.C_ib @ self.velocity_vector_i
        self.shadow_projection_axis_system = [np.array([0, 1, 0]), np.array([0, 0, 1]), np.array([0, 0, 0])]
        self.project_panels()

    def create_rear_panels(self):
        self.rear_forward_vectors = [
            np.array([np.cos(self._panel_angles[0]), 0, -np.sin(self._panel_angles[0])]), # upper rear panel
            np.array([np.cos(self._panel_angles[1]), 0, np.sin(self._panel_angles[1])]),  # lower rear panel
            np.array([np.cos(self._panel_angles[2]), -np.sin(self._panel_angles[2]), 0]), # left rear panel
            np.array([np.cos(self._panel_angles[3]), np.sin(self._panel_angles[3]), 0]),  # right rear panel
        ]
        self.rear_normal_vectors = [
            np.array([np.sin(self._panel_angles[0]), 0, np.cos(self._panel_angles[0])]), # upper rear panel
            np.array([np.sin(self._panel_angles[1]), 0, -np.cos(self._panel_angles[1])]),  # lower rear panel
            np.array([np.sin(self._panel_angles[2]), np.cos(self._panel_angles[2]), 0]), # left rear panel
            np.array([np.sin(self._panel_angles[3]), -np.cos(self._panel_angles[3]), 0]),  # right rear panel
        ]
        for idx, hinge_location in enumerate(self.panel_hinges):
            panel = Panel(self.panel_length, self.panel_width, self.rear_normal_vectors[idx],
                          self.rear_forward_vectors[idx], body=False)
            panel.define_body_nodes(hinge_location)
            self.panels[6+idx] = panel
            self.panel_nodes[6+idx] = panel.body_nodes  # type: ignore
        return


    def project_panels(self):
        self.shadow_triangle_areas = []
        self.panel_polygons = []
        velocity_unit_vector_b = self.velocity_vector_b / np.linalg.norm(self.velocity_vector_b)
        dummy_vector = np.eye(3)[np.argmin(np.abs(velocity_unit_vector_b))]  # help construct projection plane
        origin = np.array([0, 0, 0])
        x = np.cross(velocity_unit_vector_b, dummy_vector)
        x /= np.linalg.norm(x)
        y = np.cross(velocity_unit_vector_b,x)  # Crucial order to have a right-handed coordinate system!
        y /= np.linalg.norm(y)
        self.shadow_projection_axis_system = [x, y, velocity_unit_vector_b, origin] # x, y, z + origin
        for panel in self.panels:
            self.panel_polygons.append(panel.projected_polygon(self.shadow_projection_axis_system))
        # Because the shadow area will matter to determine number of particles encountered
        self.shadow = unary_union(self.panel_polygons)
        vertices = np.array(self.shadow.exterior.coords[:-1], dtype=np.float32)  # type: ignore
        triangulated = earcut.triangulate_float32(vertices, np.array([len(vertices)], dtype=np.uint32) )
        self.shadow_triangle_coords = vertices[triangulated.reshape(-1, 3)]
        self.shadow_triangle_areas = np.zeros(self.shadow_triangle_coords.shape[0])
        for idx, tri in enumerate(self.shadow_triangle_coords):
            self.shadow_triangle_areas[idx] = 0.5*np.abs(np.cross(tri[0,:]-tri[1,:],tri[0,:]-tri[2,:]))
        return

    def get_random_point(self):
        total_area = np.sum(self.shadow_triangle_areas)
        probs = self.shadow_triangle_areas / total_area
        # Choose a random triangle
        tri_idx = np.random.choice(len(self.shadow_triangle_coords), p=probs)
        tri = self.shadow_triangle_coords[tri_idx]
        # Uniform sample inside triangle using barycentric coordinates
        r1 = np.random.rand()
        r2 = np.random.rand()
        sr1 = np.sqrt(r1)
        u = 1.0 - sr1
        v = sr1 * (1.0 - r2)
        w = sr1 * r2
        point_2d = u * tri[0] + v * tri[1] + w * tri[2]  # random 2D point on the projected plane
        # print(point_2d)
        # Map 2D plane coordinates back to 3D using the plane basis
        x, y, z, origin = self.shadow_projection_axis_system
        # normal_plane_vector = -self.
        point_3d = origin + point_2d[0] * x + point_2d[1] * y
        # print(point_3d)
        return point_2d, point_3d

    def get_distance_to_planes(self, particle_location: np.ndarray, particle_velocity_vector: np.ndarray):
        """https://en.wikipedia.org/wiki/Line%E2%80%93plane_intersection"""
        l = (particle_velocity_vector / np.linalg.norm(particle_velocity_vector)).flatten()
        l0 = np.squeeze(particle_location)
        distance = 10 * [None]
        impacts = []
        #TODO: Note that we should automatically exclude all panels which does not have a component of the outward normal pointing to the particle
        #TODO: Adjust code to exclude panels to which we are exactly parallel
        for idx, panel in enumerate(self.panels):
            p0 = np.squeeze(panel.panel_center)
            n = panel.body_normal_vector.reshape(-1)
            d = np.dot((p0-l0),n)/np.dot(l,n)
            p = l0 + l*d  # point of intersection with the infinite plane
            array1 = panel.body_nodes
            array2 = np.vstack((panel.body_nodes[1:,:], panel.body_nodes[0,:]))
            # if idx == 0:
            edges = array2 - array1
            # print(edges)
            u = p - array1
            # print(u)
            cross = np.cross(u, array1)
            # print(cross)
            dir = np.dot(cross, n)
            if np.all(dir > 0):
                impacts.append(p)
            distance[idx] = d
            #TODO: Investigate results of within-panel-hit
        # print(distance)
        return impacts

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
    #TODO: Under simulation the update may happen twice per timestep, wasting computational resources
    @property
    def velocity_vector_i(self):
        return self._velocity_vector_i

    @velocity_vector_i.setter
    def velocity_vector_i(self, new_velocity_vector_i):
        self._velocity_vector_i = new_velocity_vector_i
        self.velocity_vector_b = self.C_ib @ new_velocity_vector_i
        self.project_panels()

    @property
    def C_ib(self):
        return self._C_ib

    @C_ib.setter
    def C_ib(self, new_C_ib):
        self._C_ib = new_C_ib
        self.velocity_vector_b = new_C_ib @ self.velocity_vector_i
        self.project_panels()

    ###### Make panels moveable during the simulation if wanted ######
    @property
    def panel_angles(self):
        return self._panel_angles

    @panel_angles.setter
    def panel_angles(self, new_panel_angles):
        self._panel_angles = np.deg2rad(new_panel_angles)
        self.create_rear_panels()
        self.project_panels()


    ###### VISUALISE CUBESAT ######
    def visualise(self, show_vectors: bool = False,
                  particle_vectors: np.ndarray = None, impacts: np.ndarray = None):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        for panel in self.panel_nodes:
            verts = [panel]
            panel_collection = Poly3DCollection(
                verts, facecolors='skyblue', edgecolors='k',
                linewidths=1, alpha=0.3
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
        if particle_vectors is not None:
            for vector in particle_vectors:
                x, y, z = self.velocity_vector_b/np.linalg.norm(self.velocity_vector_b)
                ax.quiver(vector[0]-x, vector[1]-y, vector[2]-z, x, y, z, )
        if impacts is not None:
            for point in impacts:
                ax.scatter(point[0], point[1], point[2],marker='x')

        if self.shadow_projection_axis_system is not None:
            x_vec, y_vec, z_vec, origin = self.shadow_projection_axis_system
            axis_length = 0.5  # scale for visibility

            ax.quiver(*origin, *(x_vec * axis_length), color='darkred', arrow_length_ratio=0.2, linewidth=2)
            ax.quiver(*origin, *(y_vec * axis_length), color='darkgreen', arrow_length_ratio=0.2, linewidth=2)
            ax.quiver(*origin, *(z_vec * axis_length), color='darkblue', arrow_length_ratio=0.2, linewidth=2)

        ax.set_xlabel('Length (x)')
        ax.set_ylabel('Width (y)')
        ax.set_zlabel('Height (z)')
        plt.title("Satellite configuration")
        ax.set_box_aspect([1, 1, 1])
        set_axes_equal(ax)
        # plt.show()
