import numpy as np
from spacecraft_body import *
from dynamics_helper_functions import *

class PassiveStabilization:
    """
    Simulation object
    """
    def __init__(self, settings=None):
        self.sat = None
        self.dt = 0.1  # Time step (s)
        self.simulation_duration = 0.1*60*60  # Simulation time (s)
        self.altitude = 400000  # Altitude (m)
        self.air_density = 1e-11  # Air density (kg/m^3)
        self.single_particle_mass = 5e-10 # Mass of single particle (kg)
        self.particles_per_cubic_meter = self.air_density / self.single_particle_mass

        self.particle_velocity = 7.8e3  # Total velocity of incoming particles (m/s)

        self.com = np.ndarray
        self.aero_to_body_eul = np.array([0, 0, 0])

        self._aero_to_body_quat = eul_to_quat(self.aero_to_body_eul)
        self.aero_to_body_CTM = quat_to_CTM(self._aero_to_body_quat)
        self.v_particle_aero_frame = np.array([-self.particle_velocity, 0, 0])
        self.v_particle_body_frame = self.aero_to_body_CTM @ self.v_particle_aero_frame

        self.inertia = 1*np.eye(3)  # TODO: obtain a reasonable estimate
        self.inertia_inv = np.linalg.inv(self.inertia)  #TODO: re-estimate with changing panel angle
        self.angular_momentum = np.zeros(3)  # TODO: Could be an initial value for tumbling and despin
        self.angular_rate = self.inertia_inv @ self.angular_momentum



    def create_cubesat(self, length: float, width: float, height: float, panel_angles: np.ndarray):
        self.sat = Satellite(length, width, height, panel_angles)
        self.sat.velocity = self.particle_velocity
        self.sat.C_b = quat_to_CTM(self._aero_to_body_quat)
        self.com = self.sat.com

    def run_simulation(self,visualise_each_timestep=False):
        time = np.arange(0, self.simulation_duration, self.dt)
        state = np.zeros((len(time), 8))
        state[:, 0] = time
        for t_idx, t in enumerate(time):
            self.simulate_timestep(visualise_timestep=visualise_each_timestep)
            state[t_idx, 1:4] = self.angular_rate
            state[t_idx, 4:8] = self._aero_to_body_quat
        return state


    def simulate_timestep(self,
                          visualise_timestep=False,
                          particle_velocity_vector=None,
                          impact_type: str = "elastic"):
        swept_volume = self.sat.shaded_area * self.particle_velocity * self.dt
        n_particles = int(self.particles_per_cubic_meter * swept_volume)
        print(f"n_particles: {n_particles} in this time step")
        impact_panel_indices, impact_coordinates, particle_velocity_vectors = (
            self.sat.generate_impacting_particle(n_particles=n_particles))
        if visualise_timestep:
            self.visualise_3d(show_velocity_vector=True,
                              impacts=impact_coordinates)
        d_p, d_L = self.calculate_momentum_exchange(impact_panel_indices=impact_panel_indices,
                                         impact_coordinates=impact_coordinates,
                                         impact_type=impact_type)
        #TODO: Do something with the linear momentum change for deorbiting and such
        self.angular_momentum += d_L
        self.angular_rate = self.inertia_inv @ self.angular_momentum

        self.aero_to_body_quat += q_dot(self._aero_to_body_quat, self.angular_rate) * self.dt
        self.aero_to_body_quat /= np.linalg.norm(self.aero_to_body_quat)
        # print(f"new attitude: {np.rad2deg(quat_to_eul(self.aero_to_body_quat))}")
        # print(f"new angular_rate: {self.angular_rate}")


    def calculate_momentum_exchange(self,
                                    impact_panel_indices: list[int] = None,
                                    impact_coordinates: np.ndarray = None,
                                    particle_velocity_vectors: np.ndarray = None,
                                    impact_type: str = "elastic"):
        n_particles = len(impact_panel_indices)
        all_panel_normals = np.zeros((10, 3))
        impact_moment_arms = impact_coordinates - self.com
        for idx, panel in enumerate(self.sat.panels):
            all_panel_normals[idx, :] = panel.body_normal_vector
        if particle_velocity_vectors is None:
            particle_velocity_vectors = np.tile(self.v_particle_body_frame, (n_particles,1))
        particle_momentum_vector = particle_velocity_vectors * self.single_particle_mass
        if impact_type == "elastic":
            alpha = 2
        elif impact_type == "inelastic" or "absorbed":
            alpha = 1
        else:
            ValueError("Invalid impact type")
        panel_normals = all_panel_normals[impact_panel_indices]
        p_normal_to_panel = (np.einsum('ij,ij->i', particle_momentum_vector, panel_normals)[:,None]*
                             panel_normals)
        total_linear_momentum = alpha * np.sum(p_normal_to_panel,axis=0)
        total_angular_momentum = np.sum(np.cross(impact_moment_arms, total_linear_momentum),axis=0)
        # print(f"total_linear_momentum: {total_linear_momentum}")
        # print(f"total_angular_momentum: {total_angular_momentum}")
        particle_velocity_out = (particle_velocity_vectors -
                                 2*np.einsum('ij,ij->i', particle_velocity_vectors, panel_normals)[:,None]*
                                 panel_normals)
        #TODO: Implement second collision check/simulation
        return total_linear_momentum, total_angular_momentum

    def show_attitude(self):
        aero_axes = np.eye(3)
        body_axes = self.aero_to_body_CTM @ np.eye(3)
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
                     impacts: np.ndarray = None):
        satellite = self.sat
        #TODO: Investigate (graphical representation of) attitude definition
        satellite.visualise(show_velocity_vector=show_velocity_vector, impacts=impacts,show_shadow_axis_system=False)
        plt.show()

    @property
    def aero_to_body_quat(self):
        return self._aero_to_body_quat

    @aero_to_body_quat.setter
    def aero_to_body_quat(self, new_quaternion_attitude):
            self._aero_to_body_quat = new_quaternion_attitude
            self.aero_to_body_CTM = quat_to_CTM(self._aero_to_body_quat)
            self.sat.aero_body_CTM = self.aero_to_body_CTM
            # self.v_particle_body_frame = self.aero_to_body_CTM @ self.v_particle_aero_frame
            # self.sat.
