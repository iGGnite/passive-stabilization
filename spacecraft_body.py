import numpy as np
import matplotlib.pyplot as plt
import mapbox_earcut as earcut
from matplotlib.lines import Line2D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from shapely.geometry import Polygon, MultiPolygon
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


def extract_polygon(geom):
    if isinstance(geom, Polygon):
        return geom
    elif isinstance(geom, MultiPolygon):
        # Return the first (and hopefully only) non-empty polygon
        for poly in geom.geoms:
            if not poly.is_empty and poly.area > 1e-10:
                return poly
    raise ValueError("No valid polygon found in geometry.")



class Panel:
    """Geometry of panel is defined in this class"""
    def __init__(self, length: float, width: float, normal_vector: np.ndarray, forward_vector: np.ndarray, body: bool = True):
        self.body = body
        self.length = length
        self.width = width
        self.body_normal_vector = normal_vector
        self.body_forward_vector = forward_vector
        self.vertices_body_frame = np.array
        self.polygon = Polygon
        self.panel_area = self.length * self.width
        self.mass = 0.1 # (kg)
        ### Panel origin in centre of panel ###
        self.vertices_panel_frame = np.array((
            [-self.length/2, self.width/2,  0], # rear left
            [-self.length/2, -self.width/2, 0], # rear right
            [ self.length/2, -self.width/2, 0], # front right
            [ self.length/2, self.width/2,  0], # front left
        ))
        body_frame_basis_vectors = np.array(np.vstack((self.body_forward_vector, self.body_normal_vector,
                                                       np.cross(self.body_normal_vector, self.body_forward_vector)))).T
        self.panel_forward_vector = np.array([1, 0, 0])
        self.panel_normal_vector = np.array([0, 0, 1])
        panel_frame_basis_vectors = np.array(np.vstack((self.panel_forward_vector,
                                                        self.panel_normal_vector,
                                                        np.cross(self.panel_normal_vector, self.panel_forward_vector)))).T
        self.R_panel_to_body = body_frame_basis_vectors @ panel_frame_basis_vectors.T
        if body:  # Rotate about centre of panel (NOTE: potential flaw if body panels not perpendicular)
            self.vertices_body_frame = (self.R_panel_to_body @ self.vertices_panel_frame.T).T
        if not body:  # Rotate around hinge point (hinge point at origin)
            self.vertices_body_frame = self.vertices_panel_frame + np.array([-self.length / 2, 0, 0]).T
            self.vertices_body_frame = (self.R_panel_to_body @ self.vertices_body_frame.T).T

    def define_body_nodes(self, position_vector: np.ndarray):  #
        self.vertices_body_frame = np.array(self.vertices_body_frame + position_vector)
        self.panel_center_body_frame = np.mean(self.vertices_body_frame, axis=0)
        return

    def projected_polygon(self, projection_plane: list)->Polygon:  # Create polygon from points making up panel shadow
        x = np.dot((self.vertices_body_frame - projection_plane[-1]), projection_plane[0])
        y = np.dot((self.vertices_body_frame - projection_plane[-1]), projection_plane[1])
        projected_coords = np.vstack((x,y)).T
        self.polygon = Polygon(projected_coords)
        return self.polygon



class Satellite:
    """Geometry of satellite is defined in this class"""
    def __init__(self, settings: dict):
        geom = settings["properties"]
        self.x_len = geom["length"]
        self.y_width = geom["width"]
        self.z_width = geom["height"]
        self.panel_length = self.x_len if geom["panel_length"] is None else geom["panel_length"]
        self.panel_width = np.min(self.y_width, self.z_width) if geom["panel_width"] is None else geom["panel_width"]
        self._panel_angles = np.deg2rad(np.array(geom["panel_angles"]).astype(float))
        self.velocity = 1 # Dummy value
        self.particle_velocity_vector_aero_frame = np.array([-self.velocity, 0, 0])
        self.body_mass = geom["body_mass"]
        self.panel_mass = geom["panel_mass"] # (kg), per panel
        self.com = geom["center_of_mass"] if not geom["center_of_mass"] else np.array([-self.x_len/2, 0, 0])
        if geom["inertia"] is not None:
            self.autocalc_inertia = False
            self._inertia = np.array(geom["inertia"])
        else:
            self.autocalc_inertia = True
            self._inertia = None
        self.calc_geometric_center()

        self._R_aero_to_body = np.eye(3) # Body orientation, to be updated externally
        self.panel_polygons = [None] * 10
        self.shadow: Polygon = None # type: ignore
        self.shadow_triangle_coords = None
        self.shadow_triangle_areas: np.ndarray = None # type: ignore
        self.rear_forward_vectors: list[np.ndarray | None] = [None] * 4 # type: ignore
        self.rear_normal_vectors: list[np.ndarray | None] = [None] * 4  # type: ignore
        self.panels: list[Panel | None] = [None] * 10
        self.panel_vertices = [np.ndarray] * 10
        self.shaded_area: float = 0

        ######## CREATION OF BODY PANEL COORDINATES ########
        self.body_panel_centres = [
            np.array([0, 0, 0]), np.array([-self.x_len, 0, 0]),  # front, rear
            np.array([-self.x_len/2, self.y_width/2, 0]), np.array([-self.x_len/2, -self.y_width/2, 0]), # left, right
            np.array([-self.x_len/2, 0, self.z_width/2]), np.array([-self.x_len/2, 0, -self.z_width/2]), # top, bottom
        ]
        self.body_normal_vectors = [
            np.array([1, 0, 0]), np.array([-1, 0,  0]),  # point forward, backward
            np.array([0, 1, 0]), np.array([0,  -1, 0]),  # point left, right
            np.array([0, 0, 1]), np.array([0,  0,  -1]),  # point up, down
        ]
        self.body_forward_vectors = [
            np.array([0, 0, 1]), np.array([0, 0, 1]), # point up
            np.array([1, 0, 0]), np.array([1, 0, 0]), # point forward
            np.array([1, 0, 0]), np.array([1, 0, 0]), # point forward
        ]
        self.panel_hinges = [
            np.array([-self.x_len, self.y_width / 2, 0]),  # left
            np.array([-self.x_len, -self.y_width / 2, 0]),  # right
            np.array([-self.x_len, 0, self.z_width / 2]),  # upper
            np.array([-self.x_len, 0, -self.z_width / 2]),  # lower
        ]
        #TODO: There is likely room to make this code more succinct
        for idx, panel_center in enumerate(self.body_panel_centres):
            if idx < 2:  # front and back (0, 1)
                panel = Panel(self.y_width, self.z_width, self.body_normal_vectors[idx], self.body_forward_vectors[idx])
            elif 1 < idx < 4:  # sides (2,3)
                panel = Panel(self.x_len, self.y_width, self.body_normal_vectors[idx], self.body_forward_vectors[idx])
            else:  # top and bottom (4,5)
                panel = Panel(self.x_len, self.z_width, self.body_normal_vectors[idx], self.body_forward_vectors[idx])
            panel.define_body_nodes(panel_center)
            self.panels[idx] = panel
            self.panel_vertices[idx] = panel.vertices_body_frame  # type: ignore

        ######## CREATION OF REAR PANEL COORDINATES, WITH ABILITY TO DETERMINE ANGLE PER PANEL ########
        self.create_rear_panels()
        if self._inertia is None:
            self.calculate_inertia()
        com_to_vertex_vector = np.array(self.panel_vertices).reshape(-1,3) - self.geometric_center
        # print(np.linalg.norm(com_to_vertex_vector,axis=0))
        self.max_dist_from_com = np.max(np.linalg.norm(com_to_vertex_vector,axis=1))

        ######## PROJECT SATELLITE ONTO NORMAL PLANE VELOCITY VECTOR ########
        self.particle_velocity_vector_b = self._R_aero_to_body.T @ np.array([-self.velocity, 0, 0])
        self.shadow_projection_axis_system = [np.array([0, 1, 0]), np.array([0, 0, 1]), np.array([0, 0, 0])]
        self.project_panels()

    def create_rear_panels(self):
        self.rear_forward_vectors = [
            np.array([np.cos(self._panel_angles[0]), -np.sin(self._panel_angles[0]), 0]), # left rear panel
            np.array([np.cos(self._panel_angles[1]), np.sin(self._panel_angles[1]), 0]),  # right rear panel
            np.array([np.cos(self._panel_angles[2]), 0, -np.sin(self._panel_angles[2])]), # upper rear panel
            np.array([np.cos(self._panel_angles[3]), 0, np.sin(self._panel_angles[3])]),  # lower rear panel
        ]
        self.rear_normal_vectors = [
            np.array([np.sin(self._panel_angles[0]), np.cos(self._panel_angles[0]), 0]), # left rear panel
            np.array([np.sin(self._panel_angles[1]), -np.cos(self._panel_angles[1]), 0]),  # right rear panel
            np.array([np.sin(self._panel_angles[2]), 0, np.cos(self._panel_angles[2])]), # upper rear panel
            np.array([np.sin(self._panel_angles[3]), 0, -np.cos(self._panel_angles[3])]),  # lower rear panel
        ]
        for idx, hinge_location in enumerate(self.panel_hinges):
            panel = Panel(self.panel_length, self.panel_width, self.rear_normal_vectors[idx],
                          self.rear_forward_vectors[idx], body=False)
            panel.define_body_nodes(hinge_location)
            self.panels[6+idx] = panel
            self.panel_vertices[6 + idx] = panel.vertices_body_frame  # type: ignore
        return


    def project_panels(self):
        velocity_unit_vector_b = self.particle_velocity_vector_b / np.linalg.norm(self.particle_velocity_vector_b)
        dummy_vector = np.eye(3)[np.argmin(np.abs(velocity_unit_vector_b))]  # help construct projection plane
        origin = self.geometric_center
        x = np.cross(velocity_unit_vector_b, dummy_vector)
        x /= np.linalg.norm(x)
        y = np.cross(velocity_unit_vector_b,x)  # Crucial order to have a right-handed coordinate system!
        y /= np.linalg.norm(y)
        self.shadow_projection_axis_system = [x, y, velocity_unit_vector_b, origin] # x, y, z + origin defined in body frame
        return



    def generate_impacting_particles(self, particle_velocity_vectors: np.ndarray = None,
                                    n_particles: int = 1, method: str = "elastic") -> list[np.ndarray]:
        """
        Generate a single particle to impact the spacecraft body. As of 27/08/2025 not yet vectorised so that I can get
        a working model fast, will need to be called multiple times.
        :param particle_velocity_vectors:
            :type particle_velocity_vectors: np.ndarray
        :param n_particles: Number of particles to generate at once
            :type n_particles: int
        :param method: Type of momentum exchange between particle and panel
        - "elastic" represents an elastic collision, where no kinetic energy is lost
        :return: A list of tuples (for later vectorisation).
        - Entry 0 specifies index of impacted panel.
        - Entry 1 specifies coordinates of impact location.
        - Entry 2 specifies distance from satellite 2d projection onto plane normal to velocity vector.
            :rtype: list[tuple]
        """
        # Preallocate 3d impact points and impact registry array
        impact_points = np.zeros((10, n_particles, 3))
        impact_array = np.zeros((n_particles, 10), dtype=int)
        probs = self.shadow_triangle_areas / self.shaded_area
        # Choose a random triangle
        tri_idx = np.random.choice(len(self.shadow_triangle_coords), n_particles, p=probs)
        tri = self.shadow_triangle_coords[tri_idx]
        # Uniform sample inside triangle using barycentric coordinates
        r1 = np.random.rand(n_particles)
        r2 = np.random.rand(n_particles)
        sr1 = np.sqrt(r1)
        u = 1.0 - sr1
        v = sr1 * (1.0 - r2)
        w = sr1 * r2
        points_2d = u[:, None] * tri[:,0] + v[:, None] * tri[:,1] + w[:, None] * tri[:,2]  # random 2D point on the projected plane

        ######## Map 2D plane coordinates back to 3D using the plane basis ########
        x, y, z, origin = self.shadow_projection_axis_system
        points_3d = (np.tile(origin, (n_particles, 1)) +
                     (points_2d[:,0])[:,None] * np.tile(x, (n_particles, 1)) +
                     (points_2d[:,1])[:,None] * np.tile(y, (n_particles, 1)))

        ################## FIND IMPACTOR POINT ON SPACECRAFT BODY ##################
        """https://en.wikipedia.org/wiki/Line%E2%80%93plane_intersection"""
        if particle_velocity_vectors is None:  # If particles direction is not specified per particle
            particle_velocity_vectors = np.tile(self.particle_velocity_vector_b, (n_particles, 1))
        l = particle_velocity_vectors / np.linalg.norm(particle_velocity_vectors, axis=1)[:,None] # unit vector particle velocity
        l0 = np.squeeze(points_3d)
        d = np.full((n_particles, 10), np.nan) #NOTE: Number of panels is hard-coded to be 10
        for idx, panel in enumerate(self.panels):  # We check per panel where particles cross their infinite plane
            dot = np.einsum('ij,ij->i', np.tile(panel.body_normal_vector, (n_particles,1)), particle_velocity_vectors)
            indices_of_particles_facing_this_panel = np.where(dot < 0)[0] # Whether current panel faces incoming particles
            if len(indices_of_particles_facing_this_panel) > 0:  # If this panel faces any particles at all
                p0 = np.tile(np.squeeze(panel.panel_center_body_frame), (len(indices_of_particles_facing_this_panel), 1))
                n = np.tile(panel.body_normal_vector, (len(indices_of_particles_facing_this_panel), 1))
                l = l[indices_of_particles_facing_this_panel]
                l0 = l0[indices_of_particles_facing_this_panel]
                d[:,idx] = np.einsum('ij,ij->i',(p0 - l0), n) / np.einsum('ij,ij->i',l, n)
                p = l0 + l * (d[:,idx])[:,None]  # 3D point of intersection with the infinite plane
                impact_points[idx,indices_of_particles_facing_this_panel,:] = p  # Register that point

                ############ DETERMINE WHETHER IMPACT ON INFINITE PLANE LIES WITHIN THE EDGES OF THE PANEL ############
                panel_corners = panel.vertices_body_frame  # 4x3 array of corner nodes of panel
                panel_corners_shifted_by_one = np.vstack((panel.vertices_body_frame[1:, :], panel.vertices_body_frame[0, :]))  # 4x3 array of corner nodes of panel, moving the first row to the end
                edge_vectors = panel_corners_shifted_by_one - panel_corners # Vectors defining edges of the panel, pointing in CCW direction (looking at panel contra-normal)
                node_to_point = p[:, None, :] - panel_corners[None, :, :]  # Vector pointing from panel corner to impact point (VERIFIED working)

                # Check whether cross product of edge vector with node_to_point vector points along or opposite panel normal
                cross_corner_impact = np.cross(edge_vectors, node_to_point)
                direction = np.einsum('ijk,ik->ij',cross_corner_impact, n)
                panel_impact = np.all(direction > 0, axis=1)  # If pointing along normal for each edge, impact IN panel
                impact_array[:, idx] = panel_impact
        distances = np.where(impact_array == 1, d, np.inf)  # Distances to panel if impact, else infinity
        indices_of_impacted_panels = np.argmin(distances, axis=1)  # Determine first panel to be hit by given particle
        impact_3D_coordinates_constr_frame = impact_points[indices_of_impacted_panels,np.arange(n_particles),:]  # Retrieve impact 3d coordinates per particle
        return [indices_of_impacted_panels, impact_3D_coordinates_constr_frame, particle_velocity_vectors, points_3d]


    def generate_impacting_particles_v2(self, particle_velocity_vectors: np.ndarray = None, n_particles: int = 1):
        impact_points = np.zeros((10, n_particles, 3))
        impact_array = np.zeros((n_particles, 10), dtype=int)
        points = np.random.uniform(-self.max_dist_from_com, self.max_dist_from_com, 2 * n_particles).reshape(n_particles, 2)
        x_shadow, y_shadow, v_particle, orig = self.shadow_projection_axis_system
        points_3d = (points[:,0] * x_shadow[:,None] + points[:,1] * y_shadow[:,None]).T + orig

        ################## FIND IMPACTOR POINT ON SPACECRAFT BODY ##################
        """https://en.wikipedia.org/wiki/Line%E2%80%93plane_intersection"""
        if particle_velocity_vectors is None:  # If particles direction is not specified per particle
            particle_velocity_vectors = np.tile(self.particle_velocity_vector_b, (n_particles, 1))
        l_ = particle_velocity_vectors / np.linalg.norm(particle_velocity_vectors, axis=1)[:,None] # unit vector particle velocity
        l0_ = np.squeeze(points_3d)
        d = np.full((n_particles, 10), np.nan) #NOTE: Number of panels is hard-coded to be 10
        for idx, panel in enumerate(self.panels):  # We check per panel where particles cross their infinite plane
            dot = np.einsum('ij,ij->i', np.tile(panel.body_normal_vector, (n_particles,1)), particle_velocity_vectors)
            indices_of_particles_facing_this_panel = np.where(dot < 0)[0] # Whether current panel faces incoming particles
            if len(indices_of_particles_facing_this_panel) > 0:  # If this panel faces any particles at all
                ## Construct infinite plane
                p0 = np.tile(np.squeeze(panel.panel_center_body_frame), (len(indices_of_particles_facing_this_panel), 1))
                n = np.tile(panel.body_normal_vector, (len(indices_of_particles_facing_this_panel), 1))
                l = l_[indices_of_particles_facing_this_panel]
                l0 = l0_[indices_of_particles_facing_this_panel]
                d[:,idx] = np.einsum('ij,ij->i',(p0 - l0), n) / np.einsum('ij,ij->i',l, n)  # Distances to infinite plane
                p = l0 + l * (d[:,idx])[:,None]  # 3D point of intersection with the infinite plane
                impact_points[idx,indices_of_particles_facing_this_panel,:] = p  # Register those points

                ############ DETERMINE WHETHER IMPACT ON INFINITE PLANE LIES WITHIN THE EDGES OF THE PANEL ############
                panel_corners = panel.vertices_body_frame  # 4x3 array of corner nodes of panel
                panel_corners_shifted_by_one = np.vstack((panel.vertices_body_frame[1:, :], panel.vertices_body_frame[0, :]))  # 4x3 array of corner nodes of panel, moving the first row to the end
                edge_vectors = panel_corners_shifted_by_one - panel_corners # Vectors defining edges of the panel, pointing in CCW direction (looking at panel contra-normal)
                node_to_point = p[:, None, :] - panel_corners[None, :, :]  # Vector pointing from panel corner to impact point (VERIFIED working)

                # Check whether cross product of edge vector with node_to_point vector points along or opposite panel normal
                cross_corner_impact = np.cross(edge_vectors, node_to_point)
                direction = np.einsum('ijk,ik->ij',cross_corner_impact, n)
                if idx < 6:
                    panel_impact = np.all(direction > 0, axis=1)  # If pointing along normal for each edge, impact IN panel
                else:  # Rear panels can be hit either from the front or back
                    panel_impact = np.logical_or(
                        np.all(direction > 0, axis=1),
                        np.all(direction < 0, axis=1)
                    )
                impact_array[:, idx] = panel_impact
        distances = np.where(impact_array == 1, d, np.inf)  # Distances to panel is registered if impacted, else it is infinity
        particle_indices = (np.arange(n_particles))[~np.all(np.isinf(distances), axis=1)]  # Get indices of particles which have at least one impact
        indices_of_impacted_panels = np.argmin(distances, axis=1)  # Determine first panel to be hit by given particle
        indices_of_impacted_panels = indices_of_impacted_panels[particle_indices]
        impact_3D_coordinates_constr_frame = impact_points[indices_of_impacted_panels,particle_indices,:]  # Retrieve impact 3d coordinates per particle
        return [indices_of_impacted_panels, impact_3D_coordinates_constr_frame, particle_velocity_vectors, points_3d]

    def print_vertices(self):
        for nodes in self.panel_vertices:
            print(nodes)
        return

    def calculate_com(self):
        #TODO: Write out
        # centres = np.zeros((10,3))
        # total_surface_area = 0.0
        # for idx, panel in enumerate(self.panels):
        #     centres[idx,:] = panel.panel_area*panel.panel_center
        #     total_surface_area += panel.panel_area
        # self.com = np.sum(centres, axis=0)/total_surface_area
        self.com = np.array([0, 0, 0])
        return

    def calc_geometric_center(self):
        x = -(self.x_len + self.panel_length*max(np.cos(self._panel_angles)))/2
        self.geometric_center = np.array([x, 0, 0])
        return

    def calculate_inertia(self):
        if self.autocalc_inertia:
            self._inertia = np.zeros((3,3))
            body_inertia = self.body_mass / 12 * np.array([[self.y_width**2 + self.z_width**2, 0, 0],
                                                      [0, self.x_len**2 + self.z_width**2, 0],
                                                      [0, 0, self.x_len**2 + self.y_width**2]])
            r_to_com = self.com - np.array([self.x_len / 2, 0, 0])
            body_inertia += self.body_mass * (np.dot(r_to_com,r_to_com)*np.eye(3) - np.outer(r_to_com,r_to_com))
            self._inertia += body_inertia
            for idx, panel in enumerate(self.panels[6:10]):
                panel_inertia = panel.mass / 12 * np.array([[panel.width**2, 0, 0],
                                                            [0, panel.length**2, 0],
                                                            [0, 0, panel.length**2 + panel.width**2]])
                r_panel = np.array([panel.length/2,0,0])
                panel_inertia += panel.mass * (np.dot(r_panel,r_panel)*np.eye(3) - np.outer(r_panel,r_panel))
                panel_inertia_body_frame = panel.R_panel_to_body @ panel_inertia @ panel.R_panel_to_body.T
                r_body = self.com - self.panel_hinges[idx]
                panel_inertia_body_frame += panel.mass * (np.dot(r_body,r_body)*np.eye(3) - np.outer(r_body,r_body))
                self._inertia += panel_inertia_body_frame
        self.inertia_inv = np.linalg.inv(self._inertia)

    ###### RE-CALCULATE THE NORMAL PLANE WHEN THE VELOCITY VECTOR CHANGES #######
    #TODO: Under simulation the update may happen twice per timestep, wasting computational resources
    @property
    def R_aero_to_body(self):
        return self._R_aero_to_body

    @R_aero_to_body.setter
    def R_aero_to_body(self, new_R_a_b):
        self._R_aero_to_body = new_R_a_b
        self.particle_velocity_vector_b = new_R_a_b.T @ np.array([-self.velocity, 0, 0])
        self.project_panels()

    ###### Make panels moveable during the simulation if wanted ######
    @property
    def panel_angles(self):
        return self._panel_angles

    @panel_angles.setter
    def panel_angles(self, new_panel_angles):
        """Update panel angles in Satellite object, resulting in updating the projection and inertia."""
        self._panel_angles = np.deg2rad(new_panel_angles)
        self.create_rear_panels()
        # For changing angle, re-calculate cross-section of encountered atmosphere (fewer resources OR avoiding panels outside area of incoming particles)
        self.calc_geometric_center()
        self.project_panels()
        self.calculate_inertia()
        #TODO: Inform Simulation object of new inertia

    def get_inertia(self):
        return self._inertia


    ###### VISUALISE CUBESAT ######
    def visualise(self,
                  show_center_of_mass: bool = True,
                  show_velocity_vector: bool = False,
                  show_panel_vectors: bool = False,
                  show_shadow_axis_system: bool = False,
                  show_particle_vectors: bool = False,
                  highlight_nodes: bool = False,
                  impacts: np.ndarray = None,
                  particle_vectors: list[np.ndarray] = None,
                  p_at_impact_vectors: np.ndarray = None,
                  points_in_projection: np.ndarray = None,):
        """
        :param show_center_of_mass:
        :param show_velocity_vector:
        :param show_panel_vectors:
        :param show_shadow_axis_system:
        :param show_particle_vectors:
        :param highlight_nodes:
        :param impacts:
        :param particle_vectors:
        :param p_at_impact_vectors:
        :param points_3d:
        :return:
        """
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        for panel in self.panel_vertices: # Draw the panels which make up the cubesat
            vertices = [panel]
            panel_collection = Poly3DCollection(vertices, facecolors='skyblue', edgecolors='k', linewidths=.5, alpha=0.3)
            ax.add_collection3d(panel_collection)
            if highlight_nodes:
               ax.scatter(panel[:, 0], panel[:, 1], panel[:, 2], color='r')

        if show_center_of_mass:
            ax.scatter(self.com[0],self.com[1],self.com[2],color='orange')

        if show_velocity_vector:
            vel_vec = self.particle_velocity_vector_b / np.linalg.norm(self.particle_velocity_vector_b)/2
            ax.quiver(0-vel_vec[0], 0-vel_vec[1], 0-vel_vec[2], vel_vec[0], vel_vec[1], vel_vec[2],color='black')

        if show_panel_vectors:
            for panel, nodes in zip(self.panels, self.panel_vertices):
                center = nodes.mean(axis=0)
                n_vec = panel.body_normal_vector / np.linalg.norm(panel.body_normal_vector) * 0.3
                f_vec = panel.body_forward_vector / np.linalg.norm(panel.body_forward_vector) * 0.3
                ax.quiver(center[0], center[1], center[2],
                          n_vec[0], n_vec[1], n_vec[2],
                          color='b', arrow_length_ratio=0.2, linewidth=2)
                ax.quiver(center[0], center[1], center[2],
                          f_vec[0], f_vec[1], f_vec[2],
                          color='g', arrow_length_ratio=0.2, linewidth=2)
            legend_elements = [ Line2D([0], [0], color='b', lw=2, label='Panel normal vector'),
                                Line2D([0], [0], color='g', lw=2, label='Panel forward vector')]
            ax.legend(handles=legend_elements, loc="best")

        if show_shadow_axis_system:  # Visualise the axis system basis generated for the 2D satellite shadow
            x_vec, y_vec, z_vec, origin = self.shadow_projection_axis_system
            axis_length = 0.5  # scale for visibility
            ax.quiver(*origin, *(x_vec * axis_length), color='darkred', arrow_length_ratio=0.2, linewidth=2)
            ax.quiver(*origin, *(y_vec * axis_length), color='darkgreen', arrow_length_ratio=0.2, linewidth=2)
            ax.quiver(*origin, *(z_vec * axis_length), color='darkblue', arrow_length_ratio=0.2, linewidth=2)

        if particle_vectors is not None and impacts is not None:
            if len(particle_vectors) != len(impacts):
                raise ValueError("Number of specified impacts and particle vectors must match")
        if impacts is not None:  # Plot the impact locations of particles with an arrow indicating the particle direction
            vec_x_dir, vec_y_dir, vec_z_dir = self.particle_velocity_vector_b / np.linalg.norm(self.particle_velocity_vector_b)/5
            for idx in range(impacts.shape[0]):
                impact_coords = impacts[idx,:]  #TODO: VECTORIZE
                ax.scatter(impact_coords[0], impact_coords[1], impact_coords[2], marker='x', color='r')
                if particle_vectors is not None and show_particle_vectors is True:  # If the particles have a specified direction, overwrite
                    particle_vector = particle_vectors[idx]
                    vec_x_dir, vec_y_dir, vec_z_dir = particle_vector/np.linalg.norm(particle_vector)
                    ax.quiver(impact_coords[0] - vec_x_dir, impact_coords[1] - vec_y_dir, impact_coords[2] - vec_z_dir,
                              vec_x_dir, vec_y_dir, vec_z_dir)
                if p_at_impact_vectors is not None:
                    p_vector = p_at_impact_vectors[idx]
                    vec_x_dir, vec_y_dir, vec_z_dir = p_vector*1e4
                    ax.quiver(impact_coords[0], impact_coords[1], impact_coords[2],
                              vec_x_dir, vec_y_dir, vec_z_dir)
        if points_in_projection is not None:
            ax.scatter(points_in_projection[:,0], points_in_projection[:,1], points_in_projection[:,2], marker='o', color='purple')

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