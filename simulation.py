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
        self.n_particles = sim_settings["particle_number"] # Predetermined number of particles in a given timestep
        if sim_settings["impact_type"] == "elastic" or sim_settings["impact_type"] == 2:
            self.impact_type = 2
        elif sim_settings["impact_type"] == "inelastic" or sim_settings["impact_type"] == 1:
            self.impact_type = 1
        else:
            raise ValueError("Impact type is currently limited to elastic or inelastic")

        ######## LOAD INITIAL STATE ########
        self.altitude = state_init["altitude"]  # Altitude (m)
        if state_init["attitude"][1] == 0 and state_init["attitude"][2] == 0: # if pitch and yaw are both exactly zero, sim breaks
            self._inertial_to_body_quat = eul_to_quat(np.deg2rad(np.array(state_init["attitude"])+np.array([0,0,1e-10])))
        else:
            self._inertial_to_body_quat = eul_to_quat(np.deg2rad(np.array(state_init["attitude"])))
        self.particle_velocity = state_init["velocity"]  # Total velocity of incoming particles (m/s)
        self.omega_ib_b = np.deg2rad(np.array(state_init["rotation_rates"]))  # rad/s

        ######## CALCULATE DERIVATIVE PARAMETERS ########
        self.R_inertial_to_body = quat_to_CTM(self._inertial_to_body_quat)
        self.v_particle_inertial_frame = np.array([-self.particle_velocity, 0, 0])
        self.v_particle_body_frame = self.R_inertial_to_body @ self.v_particle_inertial_frame
        self.particles_per_cubic_meter = self.air_density * 1000 / self.particle_mass

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

    def run_simulation(self,visualise_timesteps=False):
        start = time()
        time_points = np.arange(0, self.simulation_duration, self.dt)
        state = np.zeros((len(time_points), 14))
        impact_history = np.zeros((len(time_points), 3))
        state[:, 0] = time_points
        impact_history[:, 0] = time_points
        for t_idx, t in enumerate(time_points):
            self.step = t_idx
            state[t_idx, 1:], impact_history[t_idx, 1:] = self.simulate_timestep(visualise_timestep=visualise_timesteps)
            # if np.linalg.norm(self.omega_ib_b) > 2*np.pi:
            #     print(f"Angular rate exceeds 1 full rotation per second at t: {t}s")
            #     return state[:t_idx,:]
        stop = time()
        print(f"Simulation took {round(stop - start,5)}s for {len(time_points)} timesteps, or {round((stop - start)/len(time_points),6)} s/step")
        return state, impact_history


    def simulate_timestep(self,
                          visualise_timestep=False,
                          particle_velocity_vector=None):
        swept_mass = self.sat.max_dist_from_geom_center * self.sat.max_dist_from_long_axis * self.particle_velocity * self.dt*self.air_density  # Volume of space swept out by vehicle in timestep dt
        self.particle_mass = swept_mass/self.n_particles
        impact_panel_indices, impact_coordinates, particle_velocity_vectors, points_in_projection = (
            self.sat.generate_impacting_particles(n_particles=self.n_particles))

        d_p, d_L, n_impacts = self.calculate_momentum_exchange(impact_panel_indices=impact_panel_indices,
                                         impact_coordinates=impact_coordinates)
        # #TODO: Do something with the linear momentum change for orbital decay and such
        torque = d_L/self.dt
        kinematics = np.cross(self.omega_ib_b,self.inertia @ self.omega_ib_b)
        omega_ib_b_dot = self.inertia_inv.dot(torque - kinematics)
        self.omega_ib_b += omega_ib_b_dot*self.dt
        self.inertial_to_body_quat = q_update(self._inertial_to_body_quat,self.omega_ib_b,self.dt)

        if visualise_timestep and self.step % int(100/self.dt) == 0:
            self.visualise_3d(show_velocity_vector=True,
                              impacts=impact_coordinates,
                              p_at_impacts=None,
                              points_in_projection=None,)
        return np.hstack((self._inertial_to_body_quat, self.omega_ib_b, torque, -kinematics)), np.array([self.n_particles, n_impacts])


    def calculate_momentum_exchange(self,
                                    impact_panel_indices: list[int] = None,
                                    impact_coordinates: np.ndarray = None,
                                    particle_velocity_vectors: np.ndarray = None):
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
        # particle_velocity_out = (particle_velocity_vectors -
        #                          2*np.einsum('ij,ij->i', particle_velocity_vectors, panel_normals)[:,None]*
        #                          panel_normals)
        #TODO: Implement second collision check/simulation, using particle_velocity_out
        return linear_momentum_transfer, total_angular_momentum, p_normal_to_panel.shape[0]





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
