import numpy as np
from sympy.physics.mechanics import angular_momentum

from spacecraft_body import *
from dynamics_helper_functions import *
from time import time
from yaml import load, CLoader

class PassiveStabilization:
    """
    Simulation object
    """
    def __init__(self, config: str = "DefaultSimulator"):
        """Create simulation object"""
        self.sat = None
        self.step = 0
        ######## LOAD SIMULATOR SETTINGS ########
        with open("SimulatorConfigs/" + str(config) + ".yaml", "r") as f:
            settings = (load(f, Loader=CLoader))
            sim_settings = settings["simulator_settings"]
            state_init = settings["state"]
        self.dt = sim_settings["timestep"]  # Time step (s)
        self.simulation_duration = sim_settings["runtime"]  # Simulation time (s)
        self.air_density = sim_settings["atmospheric_density"]  # Air density (kg/m^3)
        self.particle_mass = sim_settings["particle_mass"] # Mass of single particle (kg)
        if sim_settings["impact_type"] == "elastic" or sim_settings["impact_type"] == 2:
            self.impact_type = 2
        elif sim_settings["impact_type"] == "inelastic" or sim_settings["impact_type"] == 1:
            self.impact_type = 1
        else:
            raise ValueError("Impact type is currently limited to elastic or inelastic")

        ######## LOAD INITIAL STATE ########
        self.altitude = state_init["altitude"]  # Altitude (m)
        self._inertial_to_body_quat = eul_to_quat(np.deg2rad(np.array(state_init["attitude"])))
        self.particle_velocity = state_init["velocity"]  # Total velocity of incoming particles (m/s)
        self.omega_ib_b = np.deg2rad(np.array(state_init["rotation_rates"]))  # rad/s

        ######## CALCULATE DERIVATIVE PARAMETERS ########
        self.R_inertial_to_body = quat_to_CTM(self._inertial_to_body_quat)
        self.v_particle_inertial_frame = np.array([-self.particle_velocity, 0, 0])
        self.v_particle_body_frame = self.R_inertial_to_body @ self.v_particle_inertial_frame
        self.particles_per_cubic_meter = self.air_density / self.particle_mass

        self.inertia = None
        self.inertia_inv = None
        self.com = np.ndarray

    def create_cubesat(self, config: str = "DefaultSat"):
        with open("SatelliteGeometries/" + str(config) + ".yaml", "r") as f:
            sat_config = load(f, Loader=CLoader)
        self.sat = Satellite(sat_config)
        self.sat.velocity = self.particle_velocity
        self.sat.R_aero_to_body = self.R_inertial_to_body  # TODO: Distinguish aero and inertial frames = quat_to_CTM(self._inertial_to_body_quat)
        self.com = self.sat.com
        self.inertia = self.sat.get_inertia()
        self.inertia_inv = np.linalg.inv(self.inertia)

    def run_simulation(self,visualise_each_timestep=False):
        start = time()
        time_points = np.arange(0, self.simulation_duration, self.dt)
        state = np.zeros((len(time_points), 8))
        state[:, 0] = time_points
        for t_idx, t in enumerate(time_points):
            self.simulate_timestep(visualise_timestep=visualise_each_timestep)
            state[t_idx, 1:4] = self.omega_ib_b
            state[t_idx, 4:8] = self._inertial_to_body_quat
            if np.linalg.norm(self.omega_ib_b) > 2*np.pi:
                print(f"Angular rate exceeds 1 full rotation per second at t: {t}s")
                return state[:t_idx,:]
        stop = time()
        print(f"Simulation took {stop - start} seconds for {len(time_points)} timesteps, or {(stop - start)/len(time_points)} s/step")
        return state


    def simulate_timestep(self,
                          visualise_timestep=False,
                          visualise_2d=False,
                          particle_velocity_vector=None,
                          impact_type: str = "elastic"):
        swept_volume = self.sat.max_dist_from_com ** 2 * self.particle_velocity * self.dt  # Volume of space swept out by vehicle in timestep dt
        n_particles = int(self.particles_per_cubic_meter * swept_volume)
        # self.sat.generate_impacting_particles_v2(n_particles=n_particles)
        print(f"n_particles: {n_particles} in this time step")
        impact_panel_indices, impact_coordinates, particle_velocity_vectors, points_in_projection = (
            self.sat.generate_impacting_particles_v2(n_particles=n_particles))

        d_p, d_L, ps = self.calculate_momentum_exchange(impact_panel_indices=impact_panel_indices,
                                         impact_coordinates=impact_coordinates,
                                         impact_type=impact_type)
        print(ps.shape)
        # #TODO: Do something with the linear momentum change for orbital decay and such
        torque = d_L/self.dt
        omega_ib_b_dot = self.inertia_inv.dot(torque - np.cross(self.omega_ib_b,self.inertia @ self.omega_ib_b))
        self.omega_ib_b += omega_ib_b_dot*self.dt
        self.inertial_to_body_quat = (
            q_update(self._inertial_to_body_quat,
                     self.omega_ib_b, # omega_ib_i seen in b-frame
                     self.dt))

        if visualise_timestep and self.step % int(5/self.dt) == 0:
            self.visualise_3d(show_velocity_vector=True,
                              impacts=impact_coordinates,
                              p_at_impacts=ps,
                              points_in_projection=points_in_projection,)



    def calculate_momentum_exchange(self,
                                    impact_panel_indices: list[int] = None,
                                    impact_coordinates: np.ndarray = None,
                                    particle_velocity_vectors: np.ndarray = None,
                                    impact_type: str = "elastic"):
        n_particles = len(impact_panel_indices)
        all_panel_normals = np.zeros((10, 3))
        #TODO: Consider constant velocity for all particles to speed up computation
        impact_moment_arms = impact_coordinates - self.com
        for idx, panel in enumerate(self.sat.panels):
            all_panel_normals[idx, :] = panel.body_normal_vector # containing all 10 panels
        if particle_velocity_vectors is None:
            particle_velocity_vectors = np.tile(self.v_particle_body_frame, (n_particles,1))
        particle_momentum_vectors = particle_velocity_vectors * self.particle_mass
        panel_normals = all_panel_normals[impact_panel_indices]
        p_normal_to_panel = (np.einsum('ij,ij->i', particle_momentum_vectors, panel_normals)[:,None]*
                             panel_normals)
        linear_momentum_transfer = self.impact_type * np.sum(p_normal_to_panel, axis=0)

        total_angular_momentum = np.sum(np.cross(impact_moment_arms, p_normal_to_panel),axis=0)
        # print(f"total_linear_momentum (inertial): {total_linear_momentum}")
        # print(f"total_angular_momentum (inertial): {total_angular_momentum}")
        # particle_velocity_out = (particle_velocity_vectors -
        #                          2*np.einsum('ij,ij->i', particle_velocity_vectors, panel_normals)[:,None]*
        #                          panel_normals)
        #TODO: Implement second collision check/simulation, using particle_velocity_out
        return linear_momentum_transfer, total_angular_momentum, p_normal_to_panel





    def show_attitude(self):
        aero_axes = np.eye(3)
        body_axes = self.R_inertial_to_body @ np.eye(3)
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.quiver(0,0,0,body_axes[:, 0], body_axes[:, 1], body_axes[:, 2],color=['red','green','blue'])
        ax.quiver(0,0,0,aero_axes[:, 0], aero_axes[:, 1], aero_axes[:, 2],color=['orange','darkgreen','lightblue'])
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')
        plt.show()


    def visualise_3d(self,
                     show_velocity_vector: bool = False,
                     impacts: np.ndarray = None,
                     p_at_impacts: np.ndarray = None,
                     points_in_projection: np.ndarray = None,):
        """
        :param show_velocity_vector: Whether to show the velocity vector
        :type show_velocity_vector: bool
        :param impacts: Array containing location of particle impacts
        :type impacts: np.ndarray
        :param p_at_impacts: Array containing vectors representing momentum transfer at particle impacts
        :type p_at_impacts: np.ndarray
        :param points_in_projection: Array containing points generated on projection plane
        :type points_in_projection: np.ndarray
        :return:
        """
        self.sat.visualise(show_velocity_vector=show_velocity_vector, impacts=impacts,show_shadow_axis_system=False,
                           p_at_impact_vectors=p_at_impacts,points_in_projection=points_in_projection)
        plt.show()

    @property
    def inertial_to_body_quat(self):
        return self._inertial_to_body_quat

    @inertial_to_body_quat.setter
    def inertial_to_body_quat(self, new_quaternion_attitude):
        """Redefines satellite body attitude wrt inertial frame, and updates associated parameters elsewhere."""
        self._inertial_to_body_quat = new_quaternion_attitude
        self.R_inertial_to_body = quat_to_CTM(self._inertial_to_body_quat)
        self.v_particle_body_frame = self.R_inertial_to_body.T @ self.v_particle_inertial_frame
        self.sat.R_aero_to_body = self.R_inertial_to_body #TODO: Untangle aerodynamic and inertial orientation in future expansion

    def set_panel_angles(self, panel_angles: np.ndarray):
        """Sets new panel angles for rear panels"""
        self.sat.panel_angles = panel_angles
