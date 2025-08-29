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
        self.panel_area = self.length * self.width
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

    def __init__(self, x_len: float, y_width: float, z_width: float, rear_panel_angles: np.ndarray,
                 panel_length: float = None, panel_width: float = None, inertia: np.ndarray = np.eye(3)):
        self.x_len = x_len
        self.y_width = y_width
        self.z_width = z_width
        self.panel_length = panel_length
        self.panel_width = panel_width
        if self.panel_length is None:
            self.panel_length = self.x_len
        if self.panel_width is None:
            self.panel_width = max(self.y_width, self.z_width)
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
        self.shaded_area: float = 0
        self.com: np.ndarray = np.zeros(3)

        #TODO: may be unnecessarily expensive to run this
        for idx, panel_angle in enumerate(rear_panel_angles):
            while panel_angle < 0 or panel_angle > 90:
                ValueError("Panel angle must be between 0 and 90 degrees, but is " + str(panel_angle) + " degrees.")
                panel_angle = float(input("Provide a new panel angle in degrees:"))
            self._panel_angles[idx] = np.deg2rad(panel_angle)
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
        centres = np.zeros((10,3))
        total_surface_area = 0.0
        for idx, panel in enumerate(self.panels):
            centres[idx,:] = panel.panel_area*panel.panel_center
            total_surface_area += panel.panel_area
        self.com = np.sum(centres, axis=0)/total_surface_area
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
        self.shaded_area = np.sum(self.shadow_triangle_areas)
        return

    def generate_impacting_particle(self, particle_velocity_vector: np.ndarray = None,
                                    n_particles: int = 1, method: str = "elastic") -> list[tuple]:
        """
        Generate a single particle to impact the spacecraft body. As of 27/08/2025 not yet vectorised so that I can get
        a working model fast, will need to be called multiple times.
        :param particle_velocity_vector:
            :type particle_velocity_vector: np.ndarray
        :param n_particles: (Currently not working) number of particles to generate at once
            :type n_particles: int
        :param method: Type of momentum exchange between particle and panel
        - "elastic" represents an elastic collision, where no kinetic energy is lost
        :return: A list of tuples (for later vectorisation).
        - Entry 0 specifies index of impacted panel.
        - Entry 1 specifies coordinates of impact location.
        - Entry 2 specifies distance from satellite 2d projection onto plane normal to velocity vector.
            :rtype: list[tuple]
        """
        impacts = []
        probs = self.shadow_triangle_areas / self.shaded_area
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

        ######## Map 2D plane coordinates back to 3D using the plane basis ########
        x, y, z, origin = self.shadow_projection_axis_system
        point_3d = origin + point_2d[0] * x + point_2d[1] * y

        ## FIND IMPACTOR POINT ON SPACECRAFT BODY ##
        """https://en.wikipedia.org/wiki/Line%E2%80%93plane_intersection"""
        if particle_velocity_vector is None:
            particle_velocity_vector = self.velocity_vector_b
        l = (particle_velocity_vector / np.linalg.norm(particle_velocity_vector)).flatten()
        l0 = np.squeeze(point_3d)
        impact_candidates = []

        # We exclude panels which do not even face the particle
        for idx, panel in enumerate(self.panels):
            dot = np.dot(panel.body_normal_vector, self.velocity_vector_b)
            if dot < 0:  # if negative, panel faces the 'wind' and is eligible for evaluation
                p0 = np.squeeze(panel.panel_center)
                n = panel.body_normal_vector.reshape(-1)
                d = np.dot((p0 - l0), n) / np.dot(l, n)
                p = l0 + l * d  # point of intersection with the infinite plane
                array1 = panel.body_nodes
                array2 = np.vstack((panel.body_nodes[1:, :], panel.body_nodes[0, :]))
                edges = array2 - array1
                u = p - array1
                cross = np.cross(edges, u)
                direction = np.dot(cross, n)
                if np.all(direction > 0):  # Check if point lies in plane
                    impact_candidates.append((idx, p, d))
        # print(f"impact_candidates: {impact_candidates}")
        impact = None
        for point in impact_candidates:
            if impact is None:
                impact = point
            elif point[2] < impact[2]:  # If the impact occurs earlier along the direction of travel select it
                impact = point
        # if method == "elastic":
        #     impacted_panel = self.panels[impact[0]]
        #     impacted_panel_normal = impacted_panel.body_normal_vector
        #     particle_momentum = particle_velocity_vector *
        #     momentum_normal_to_panel =
        #     print(impacted_panel_normal)
        # elif method == "inelastic":
        #     ...
        impacts.append(impact)
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
    def visualise(self,
                  show_velocity_vector: bool = False,
                  show_panel_vectors: bool = False,
                  show_shadow_axis_system: bool = False,
                  highlight_nodes: bool = False,
                  impacts: np.ndarray = None,
                  particle_vectors: list[np.ndarray] = None):
        """
        :param show_velocity_vector:
        :param show_panel_vectors:
        :param highlight_nodes:
        :param impacts:
        :param particle_vectors:
        :param show_shadow_axis_system:
        :return:
        """
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        for panel in self.panel_nodes: # Draw the panels which make up the cubesat
            vertices = [panel]
            panel_collection = Poly3DCollection(vertices, facecolors='skyblue', edgecolors='k', linewidths=.5, alpha=0.3)
            ax.add_collection3d(panel_collection)
            if highlight_nodes:
               ax.scatter(panel[:, 0], panel[:, 1], panel[:, 2], color='r')

        if show_velocity_vector:
            vel_vec = self.velocity_vector_b / np.linalg.norm(self.velocity_vector_b)
            ax.quiver(0-vel_vec[0], 0-vel_vec[1], 0-vel_vec[2], vel_vec[0], vel_vec[1], vel_vec[2])

        if show_panel_vectors:
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
            legend_elements = [ Line2D([0], [0], color='b', lw=2, label='Panel normal vector'),
                                Line2D([0], [0], color='g', lw=2, label='Panel forward vector')]
            ax.legend(handles=legend_elements, loc="best")

        if particle_vectors is not None and impacts is not None:
            if len(particle_vectors) != len(impacts):
                raise ValueError("Number of specified impacts and particle vectors must match")
        if impacts is not None:  # Plot the impact locations of particles with an arrow indicating the particle direction
            vec_x_dir, vec_y_dir, vec_z_dir = self.velocity_vector_b / np.linalg.norm(self.velocity_vector_b)
            for idx in range(impacts.shape[0]):
                impact_coords = impacts[idx,:]  #TODO: VECTORIZE
                ax.scatter(impact_coords[0], impact_coords[1], impact_coords[2], marker='x', color='r')
                if particle_vectors is not None:  # If the particles have a specified direction, overwrite
                    particle_vector = particle_vectors[idx]
                    vec_x_dir, vec_y_dir, vec_z_dir = particle_vector/np.linalg.norm(particle_vector)
                ax.quiver(impact_coords[0] - vec_x_dir, impact_coords[1] - vec_y_dir, impact_coords[2] - vec_z_dir,
                          vec_x_dir, vec_y_dir, vec_z_dir)

        if show_shadow_axis_system:  # Visualise the axis system basis generated for the 2D satellite shadow
            x_vec, y_vec, z_vec, origin = self.shadow_projection_axis_system
            axis_length = 0.5  # scale for visibility
            ax.quiver(*origin, *(x_vec * axis_length), color='darkred', arrow_length_ratio=0.2, linewidth=2)
            ax.quiver(*origin, *(y_vec * axis_length), color='darkgreen', arrow_length_ratio=0.2, linewidth=2)
            ax.quiver(*origin, *(z_vec * axis_length), color='darkblue', arrow_length_ratio=0.2, linewidth=2)

        ax.set_xlabel('Length (x)')
        ax.set_ylabel('Width (y)')
        ax.set_zlabel('Height (z)')
        ax.set_xlim(-(2*self.x_len+.2),0.5)
        ax.set_ylim(-(self.y_width+.5),(self.y_width+.5))
        ax.set_zlim(-(self.z_width+.5), (self.z_width+.5))
        plt.title("Satellite configuration")
        ax.set_box_aspect([1, 1, 1])
        set_axes_equal(ax)
        plt.show()
