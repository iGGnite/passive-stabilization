import numpy as np
import matplotlib.pyplot as plt
# import mapbox_earcut as earcut
from matplotlib.lines import Line2D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from shapely.geometry import Polygon, MultiPolygon
# from shapely.set_operations import unary_union

from dynamics_helper_functions import quat_to_CTM


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
    def __init__(self, length: float, width: float, normal_vector: np.ndarray, forward_vector: np.ndarray,
                 body: bool = True):
        self.body = body
        self.length = length
        self.width = width
        self.body_normal_vector = normal_vector
        self.body_forward_vector = forward_vector
        self.vertices_body_frame = np.array
        self.polygon = Polygon
        self.panel_area = self.length * self.width
        self.mass = 0.1  # (kg)
        ### Panel origin in centre of panel ###
        self.vertices_panel_frame = np.array((
            [-self.length / 2, self.width / 2, 0],  # rear left
            [-self.length / 2, -self.width / 2, 0],  # rear right
            [self.length / 2, -self.width / 2, 0],  # front right
            [self.length / 2, self.width / 2, 0],  # front left
        ))
        body_frame_basis_vectors = np.array(np.vstack((self.body_forward_vector, self.body_normal_vector,
                                                       np.cross(self.body_normal_vector, self.body_forward_vector)))).T
        self.panel_forward_vector = np.array([1, 0, 0])
        self.panel_normal_vector = np.array([0, 0, 1])
        panel_frame_basis_vectors = np.array(
            np.vstack((self.panel_forward_vector,
                       self.panel_normal_vector,
                       np.cross(self.panel_normal_vector, self.panel_forward_vector)))
        ).T
        self.R_panel_to_body = body_frame_basis_vectors @ panel_frame_basis_vectors.T
        if body:  # Rotate about centre of panel (NOTE: potential flaw if body panels not perpendicular)
            self.vertices_body_frame = (self.R_panel_to_body @ self.vertices_panel_frame.T).T
        if not body:  # Rotate around hinge point (hinge point at origin)
            self.vertices_body_frame = self.vertices_panel_frame + np.array([-self.length / 2, 0, 0]).T
            self.vertices_body_frame = (self.R_panel_to_body @ self.vertices_body_frame.T).T

    def define_body_nodes(self, position_vector: np.ndarray):  #
        """Determine location of panel vertices in body frame"""
        self.vertices_body_frame = np.array(self.vertices_body_frame + position_vector)
        self.panel_center_body_frame = np.mean(self.vertices_body_frame, axis=0)
        return

    def projected_polygon(self, projection_plane: list) -> Polygon:
        """Create polygon from points making up panel shadow. Eclipsed by random particle generation"""
        x = np.dot((self.vertices_body_frame - projection_plane[-1]), projection_plane[0])
        y = np.dot((self.vertices_body_frame - projection_plane[-1]), projection_plane[1])
        projected_coords = np.vstack((x, y)).T
        self.polygon = Polygon(projected_coords)
        return self.polygon


class CubeSat:
    """Geometry of CubeSat is defined in this class

    Class which contains the geometric and mass specifications of the CubeSat. CubeSat is made up of Panel objects, and
    contains functions relating to updating the geometry, generating a 'shadow plane' normal to the particle velocity
    vector, and calculating the inertia tensor. Some of these properties are used directly in the PassiveStabilization
    class to run a simulation.
    """
    def __init__(self, settings: dict):
        geom = settings["properties"]
        self.x_len = geom["length"]
        self.y_width = geom["width"]
        self.z_width = geom["height"]
        self.panel_length = self.x_len if geom["panel_length"] is None else geom["panel_length"]
        self.panel_width = np.min(self.y_width, self.z_width) if geom["panel_width"] is None else geom["panel_width"]
        self._panel_angles = np.deg2rad(np.array(geom["panel_angles"]).astype(float))
        self.velocity = 1  # Dummy value
        # self.particle_velocity_vector_aero_frame = np.array([-self.velocity, 0, 0])
        self.body_mass = geom["body_mass"]
        self.panel_mass = geom["panel_mass"]  # (kg), per panel
        self.com = geom["center_of_mass"] if "center_of_mass" in geom else np.array([-self.x_len / 2, 0, 0])
        if geom["inertia"] is not None:
            self.autocalc_inertia = False
            self._inertia = np.array(geom["inertia"])
        else:
            self.autocalc_inertia = True
            self._inertia = None
        self.calc_geometric_center()

        self._R_aero_to_body = np.eye(3)  # Body orientation, to be updated externally
        self.panel_polygons = [None] * 10
        # self.shadow = None # Retired feature
        # self.shadow_triangle_coords = None
        # self.shadow_triangle_areas = None
        self.rear_forward_vectors = []
        self.rear_normal_vectors = []
        self.panels = [None] * 10
        self.panel_vertices = [None] * 10

        ######## CREATION OF BODY PANEL COORDINATES ########
        self.body_panel_centres = [
            np.array([0, 0, 0]), np.array([-self.x_len, 0, 0]),  # front, rear
            np.array([-self.x_len / 2, self.y_width / 2, 0]), np.array([-self.x_len / 2, -self.y_width / 2, 0]), # left, right
            np.array([-self.x_len / 2, 0, self.z_width / 2]), np.array([-self.x_len / 2, 0, -self.z_width / 2]), # top, bottom
        ]
        self.body_normal_vectors = [
            np.array([1, 0, 0]), np.array([-1, 0, 0]),  # point forward, backward
            np.array([0, 1, 0]), np.array([0, -1, 0]),  # point left, right
            np.array([0, 0, 1]), np.array([0, 0, -1]),  # point up, down
        ]
        self.body_forward_vectors = [
            np.array([0, 0, 1]), np.array([0, 0, 1]),  # point up
            np.array([1, 0, 0]), np.array([1, 0, 0]),  # point forward
            np.array([1, 0, 0]), np.array([1, 0, 0]),  # point forward
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
        com_to_vertex_vector = np.array(self.panel_vertices).reshape(-1, 3) - self.geometric_center
        # print(np.linalg.norm(com_to_vertex_vector,axis=0))
        self.max_dist_from_geom_center = np.max(np.linalg.norm(com_to_vertex_vector, axis=1))
        self.max_dist_from_long_axis = np.max(np.linalg.norm(com_to_vertex_vector[:, 1:3], axis=1))

        ######## PROJECT CUBESAT ONTO NORMAL PLANE VELOCITY VECTOR ########
        self.particle_velocity_vector_b = self._R_aero_to_body.T @ np.array([-self.velocity, 0, 1e-12]) # small value to avoid div by 0
        self.shadow_projection_axis_system = [np.array([0, 1, 0]), np.array([0, 0, 1]), np.array([0, 0, 0])]
        self.project_panels()

    def create_rear_panels(self):
        """Given specified geometry, generate aft panels as Panel objects and add these to CubeSat object"""
        self.rear_forward_vectors = [
            np.array([np.cos(self._panel_angles[0]), -np.sin(self._panel_angles[0]), 0]),  # left rear panel
            np.array([np.cos(self._panel_angles[1]), np.sin(self._panel_angles[1]), 0]),  # right rear panel
            np.array([np.cos(self._panel_angles[2]), 0, -np.sin(self._panel_angles[2])]),  # upper rear panel
            np.array([np.cos(self._panel_angles[3]), 0, np.sin(self._panel_angles[3])]),  # lower rear panel
        ]
        self.rear_normal_vectors = [
            np.array([np.sin(self._panel_angles[0]), np.cos(self._panel_angles[0]), 0]),  # left rear panel
            np.array([np.sin(self._panel_angles[1]), -np.cos(self._panel_angles[1]), 0]),  # right rear panel
            np.array([np.sin(self._panel_angles[2]), 0, np.cos(self._panel_angles[2])]),  # upper rear panel
            np.array([np.sin(self._panel_angles[3]), 0, -np.cos(self._panel_angles[3])]),  # lower rear panel
        ]
        for idx, hinge_location in enumerate(self.panel_hinges):
            panel = Panel(self.panel_length, self.panel_width, self.rear_normal_vectors[idx],
                          self.rear_forward_vectors[idx], body=False)
            panel.define_body_nodes(hinge_location)
            self.panels[6 + idx] = panel
            self.panel_vertices[6 + idx] = panel.vertices_body_frame  # type: ignore
        return

    def project_panels(self):
        """Construct 2D plane normal to velocity vector in which impacting particles can be generated"""
        velocity_unit_vector_b = self.particle_velocity_vector_b / np.linalg.norm(self.particle_velocity_vector_b)
        origin = self.geometric_center
        x = np.cross(np.cross(velocity_unit_vector_b, np.array([1, 0, 0])),
                     velocity_unit_vector_b)  # Put x_shadow as close as possible to x_body
        x /= np.linalg.norm(x)
        y = np.cross(velocity_unit_vector_b, x)  # Complete right-handed coordinate system
        y /= np.linalg.norm(y)
        self.shadow_projection_axis_system = [x, y, velocity_unit_vector_b,
                                              origin]  # shadow x, y, (z) + origin defined in body frame
        return

    def generate_impacting_particles(self, n_particles: int = 1):
        """Generate particle impacts

        Creates particles in plane normal to incoming particle velocity vector, and determines where along that line the
        particle intersects the infinite plane of each CubeSat panel. Then determines whether impact in infinite plane
        lies in the physical panel. Returns 3D coordinates of impact for first panel encountered (if any), index of the
        respective panel, and original coordinates at which the particle was generated in the velocity normal plane.

        :param n_particles: Number of particles to generate
        :return: list with, per impacting particle, impacted panel index, impact coordinates, original creation coordinates
        :rtype: list
        """
        impact_points = np.zeros((10, n_particles, 3))
        impact_array = np.zeros((n_particles, 10), dtype=int)
        points_x = np.random.uniform(-self.max_dist_from_geom_center, self.max_dist_from_geom_center, n_particles)
        points_y = np.random.uniform(-self.max_dist_from_long_axis, self.max_dist_from_long_axis, n_particles)
        x_shadow, y_shadow, v_particle, orig = self.shadow_projection_axis_system
        points_3d = (points_x * x_shadow[:, None] + points_y * y_shadow[:, None]).T + orig

        ################## FIND IMPACTOR POINT ON SPACECRAFT BODY ##################
        """https://en.wikipedia.org/wiki/Line%E2%80%93plane_intersection"""
        particle_velocity_vectors = np.tile(self.particle_velocity_vector_b, (n_particles, 1))
        l_ = np.tile(self.particle_velocity_vector_b / np.linalg.norm(self.particle_velocity_vector_b),
                     (n_particles, 1))  # unit vector particle velocity
        l0_ = np.squeeze(points_3d)
        d = np.full((n_particles, 10), np.nan)  #NOTE: Number of panels is hard-coded to be 10
        for idx, panel in enumerate(self.panels):  # We check per panel where particles cross their infinite plane
            dot = np.einsum('ij,ij->i', np.tile(panel.body_normal_vector, (n_particles, 1)), particle_velocity_vectors)
            if idx < 6:
                indices_of_particles_facing_this_panel = np.where(dot < 0)[
                    0]  # Whether current panel faces incoming particles
            else:  # Rear panels can be hit from either side
                indices_of_particles_facing_this_panel = np.arange(n_particles)
            if indices_of_particles_facing_this_panel.size > 0:  # If this panel faces any particles at all
                ## Construct infinite plane
                p0 = np.tile(np.squeeze(panel.panel_center_body_frame),
                             (len(indices_of_particles_facing_this_panel), 1))
                n = np.tile(panel.body_normal_vector, (len(indices_of_particles_facing_this_panel), 1))
                l = l_[indices_of_particles_facing_this_panel]
                l0 = l0_[indices_of_particles_facing_this_panel]
                d[:, idx] = np.einsum('ij,ij->i', (p0 - l0), n) / np.einsum('ij,ij->i', l,
                                                                            n)  # Distances to infinite plane
                p = l0 + l * (d[:, idx])[:, None]  # 3D point of intersection with the infinite plane
                impact_points[idx, indices_of_particles_facing_this_panel, :] = p  # Register those points

                ############ DETERMINE WHETHER IMPACT ON INFINITE PLANE LIES WITHIN THE EDGES OF THE PANEL ############
                panel_corners = panel.vertices_body_frame  # 4x3 array of corner nodes of panel
                panel_corners_shifted_by_one = np.vstack((panel.vertices_body_frame[1:, :], panel.vertices_body_frame[
                    0, :]))  # 4x3 array of corner nodes of panel, moving the first row to the end
                edge_vectors = panel_corners_shifted_by_one - panel_corners  # Vectors defining edges of the panel, pointing in CCW direction (looking at panel contra-normal)
                node_to_point = p[:, None, :] - panel_corners[
                    None, :, :]  # Vector pointing from panel corner to impact point (VERIFIED working)

                # Check whether cross product of edge vector with node_to_point vector points along or opposite panel normal
                cross_corner_impact = np.cross(edge_vectors, node_to_point)
                direction = np.einsum('ijk,ik->ij', cross_corner_impact, n)
                if idx < 6:
                    panel_impact = np.all(direction > 0,
                                          axis=1)  # If pointing along normal for each edge, impact IN panel
                else:  # Rear panels can be hit either from the front or back
                    panel_impact = np.logical_or(
                        np.all(direction > 0, axis=1),
                        np.all(direction < 0, axis=1))
                impact_array[:, idx] = panel_impact
        distances = np.where(impact_array == 1, d,
                             np.inf)  # Distances to panel is registered if impacted, else it is infinity
        particle_indices = (np.arange(n_particles))[
            ~np.all(np.isinf(distances), axis=1)]  # Get indices of particles which have at least one impact
        indices_of_impacted_panels = np.argmin(distances, axis=1)  # Determine first panel to be hit by given particle
        indices_of_impacted_panels = indices_of_impacted_panels[particle_indices]
        impact_3D_coordinates_constr_frame = impact_points[
            indices_of_impacted_panels, particle_indices, :]  # Retrieve impact 3d coordinates per particle
        return [indices_of_impacted_panels, impact_3D_coordinates_constr_frame, points_3d]

    def calculate_com(self):
        """Automatically calculate center of mass of satellite. Not yet implemented"""
        raise NotImplemented("Automatic center of mass calculation not yet implemented")

    def calc_geometric_center(self):
        """Determine the 'geometric center', which is meant to help minimise the size of the shadow plane"""
        self.geometric_center = np.array(
            [-(self.x_len + self.panel_length * max(np.cos(self._panel_angles))) / 2, 0, 0])
        return

    def calculate_inertia(self):
        """Automatically calculate CubeSat inertia tensor using specified geometry and mass distribution"""
        if self.autocalc_inertia:
            self._inertia = np.zeros((3, 3))
            body_inertia = self.body_mass / 12 * np.array([[self.y_width ** 2 + self.z_width ** 2, 0, 0],
                                                           [0, self.x_len ** 2 + self.z_width ** 2, 0],
                                                           [0, 0, self.x_len ** 2 + self.y_width ** 2]])
            r_to_com = self.com - np.array([self.x_len / 2, 0, 0])
            body_inertia += self.body_mass * (np.dot(r_to_com, r_to_com) * np.eye(3) - np.outer(r_to_com, r_to_com))
            self._inertia += body_inertia
            for idx, panel in enumerate(self.panels[6:10]):
                panel_inertia = panel.mass / 12 * np.array([[panel.width ** 2, 0, 0],
                                                            [0, panel.length ** 2, 0],
                                                            [0, 0, panel.length ** 2 + panel.width ** 2]])
                r_panel = np.array([panel.length / 2, 0, 0])
                panel_inertia += panel.mass * (np.dot(r_panel, r_panel) * np.eye(3) - np.outer(r_panel, r_panel))
                panel_inertia_body_frame = panel.R_panel_to_body @ panel_inertia @ panel.R_panel_to_body.T
                r_body = self.com - self.panel_hinges[idx]
                panel_inertia_body_frame += panel.mass * (np.dot(r_body, r_body) * np.eye(3) - np.outer(r_body, r_body))
                self._inertia += panel_inertia_body_frame
        else:
            raise ValueError("Inertia not meant to be calculated according to CubeSat config file")
        self.inertia_inv = np.linalg.inv(self._inertia)

    ###### RE-CALCULATE THE NORMAL PLANE WHEN THE VELOCITY VECTOR CHANGES #######
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
        """Update panel angles in CubeSat object, and then derivative attributes"""
        self._panel_angles = np.deg2rad(new_panel_angles)
        self.create_rear_panels()
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
        for panel in self.panel_vertices:  # Draw the panels which make up the cubesat
            vertices = [panel]
            panel_collection = Poly3DCollection(vertices, facecolors='skyblue', edgecolors='k', linewidths=.5,
                                                alpha=0.3)
            ax.add_collection3d(panel_collection)
            if highlight_nodes:
                ax.scatter(panel[:, 0], panel[:, 1], panel[:, 2], color='r')

        if show_center_of_mass:
            ax.scatter(self.com[0], self.com[1], self.com[2], color='orange')

        if show_velocity_vector:
            vel_vec = self.particle_velocity_vector_b / np.linalg.norm(self.particle_velocity_vector_b) / 2
            ax.quiver(0 - vel_vec[0], 0 - vel_vec[1], 0 - vel_vec[2], vel_vec[0], vel_vec[1], vel_vec[2], color='black')

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
            legend_elements = [Line2D([0], [0], color='b', lw=2, label='Panel normal vector'),
                               Line2D([0], [0], color='g', lw=2, label='Panel forward vector')]
            ax.legend(handles=legend_elements, loc="best")

        if show_shadow_axis_system:  # Visualise the axis system basis generated for the 2D CubeSat shadow
            x_vec, y_vec, z_vec, origin = self.shadow_projection_axis_system
            axis_length = 0.5  # scale for visibility
            ax.quiver(*origin, *(x_vec * axis_length), color='darkred', arrow_length_ratio=0.2, linewidth=2)
            ax.quiver(*origin, *(y_vec * axis_length), color='darkgreen', arrow_length_ratio=0.2, linewidth=2)
            ax.quiver(*origin, *(z_vec * axis_length), color='darkblue', arrow_length_ratio=0.2, linewidth=2)

        if particle_vectors is not None and impacts is not None:
            if len(particle_vectors) != len(impacts):
                raise ValueError("Number of specified impacts and particle vectors must match")
        if impacts is not None:  # Plot the impact locations of particles with an arrow indicating the particle direction
            vec_x_dir, vec_y_dir, vec_z_dir = self.particle_velocity_vector_b / np.linalg.norm(
                self.particle_velocity_vector_b) / 5
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
            vertices = np.array([[-self.max_dist_from_geom_center, -self.max_dist_from_long_axis],
                                 [-self.max_dist_from_geom_center, self.max_dist_from_long_axis],
                                 [self.max_dist_from_geom_center, self.max_dist_from_long_axis],
                                 [self.max_dist_from_geom_center, -self.max_dist_from_long_axis],
                                 ])
            print(vertices[:, 0])
            vertices = ((vertices[:, 0] * (self.shadow_projection_axis_system[0])[:, None] +
                         vertices[:, 1] * (self.shadow_projection_axis_system[1])[:, None]).T +
                        self.shadow_projection_axis_system[3])
            print(vertices)
            ax.add_collection3d(
                Poly3DCollection([vertices], facecolors='cyan', linewidths=1, edgecolors='r', alpha=0.5))
        ax.set_xlabel('Length (x)')
        ax.set_ylabel('Width (y)')
        ax.set_zlabel('Height (z)')
        ax.set_xlim(-(2 * self.x_len + .2), 0.5)
        ax.set_ylim(-(self.y_width + .5), (self.y_width + .5))
        ax.set_zlim(-(self.z_width + .5), (self.z_width + .5))
        plt.title("CubeSat configuration")
        ax.set_box_aspect([1, 1, 1])
        set_axes_equal(ax)
        plt.show()
