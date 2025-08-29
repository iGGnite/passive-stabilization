import numpy as np
from spacecraft_body import *

# class SimulationSettings:
#     def __init__(self):


class PassiveStabilization:
    """
    Simulation object
    """
    def __init__(self, settings=None):
        self.sat = None
        self.dt = 0.01  # Time step (s)
        self.simulation_duration = 10  # Simulation time (s)
        self.altitude = 400000  # Altitude (m)
        self.air_density = 1e-11  # Air density (kg/m^3)
        self.single_particle_mass = 1e-10 # Mass of single particle (kg)
        self.particles_per_cubic_meter = self.air_density / self.single_particle_mass
        print(f"Particles per cubic meter: {self.particles_per_cubic_meter}")
        self.vehicle_velocity_vector = np.array([7.8e3, 1, 1])  # (Orbital) velocity of vehicle (frame not currently defined)
        self.particle_velocity_magnitude = np.linalg.norm(self.vehicle_velocity_vector)  # Total velocity of incoming particles (m/s)
        self.v_particle_b_frame = -self.vehicle_velocity_vector  # Particle appears to move opposite to the spacecraft
        self.com = np.ndarray
        self.inertia = 0.1*np.eye(3)  # TODO: obtain a reasonable estimate
        self.inertia_inv = np.linalg.inv(self.inertia)
        self.angular_momentum = np.zeros(3)  # TODO: Could be an initial value for tumbling and despin
        self.angular_rate = self.inertia_inv @ self.angular_momentum

    def create_cubesat(self, length: float, width: float, height: float, panel_angles: np.ndarray):
        self.sat = Satellite(length, width, height, panel_angles)
        self.sat.velocity_vector_i = self.v_particle_b_frame  # TODO: These orientation definitions are all messed up
        self.com = self.sat.com

    def run_simulation(self):
        time = np.arange(0, self.simulation_duration, self.dt)
        state = np.zeros((len(time), 4))
        state[:, 0] = time
        for t_idx, t in enumerate(time):
            self.simulate_timestep()
            state[t_idx, 1:4] = self.angular_rate
            #TODO: Implement attitude dynamics equations
        return state


    def simulate_timestep(self,
                          visualise_timestep=False,
                          particle_velocity_vector=None,
                          impact_type: str = "elastic"):
        swept_volume = self.sat.shaded_area * self.particle_velocity_magnitude * self.dt
        n_particles = int(self.particles_per_cubic_meter * swept_volume)
        print(f"n_particles: {n_particles} in this time step")
        impact_panel_indices = []
        impact_coordinates = np.zeros((n_particles, 3))
        for n in range(n_particles):
            impact = self.sat.generate_impacting_particle()[0]  #TODO: Currently a list of tuples, hence [0]
            impact_panel_indices.append(impact[0])
            impact_coordinates[n,:] = impact[1]
        if visualise_timestep:
            self.visualise_3d(show_velocity_vector=False,
                              impacts=impact_coordinates
                              )
        d_p, d_L = self.calculate_momentum_exchange(impact_panel_indices=impact_panel_indices,
                                         impact_coordinates=impact_coordinates,
                                         impact_type=impact_type)
        #TODO: Do something with the linear momentum change for deorbiting and such
        self.angular_momentum += d_L
        #TODO: Investigate how expensive inverting the inertia is
        self.angular_rate = self.angular_rate + self.inertia_inv @ self.angular_momentum
        print(f"new angular_rate: {self.angular_rate}")


    def calculate_momentum_exchange(self,
                                    impact_panel_indices: list[int] = None,
                                    impact_coordinates: np.ndarray = None,
                                    particle_velocity_vectors: np.ndarray = None,
                                    impact_type: str = "elastic"):
        n_particles = len(impact_panel_indices)
        panel_normals = np.zeros((10, 3))
        moment_arms = impact_coordinates - self.com
        for idx, panel in enumerate(self.sat.panels):
            # print(panel.body_normal_vector)
            panel_normals[idx, :] = panel.body_normal_vector
        if particle_velocity_vectors is None:
            particle_velocity_vectors = np.tile(-self.vehicle_velocity_vector, (n_particles,1))
        particle_momentum_vector = particle_velocity_vectors * self.single_particle_mass
        # print(f"particle_momentum_vector: {particle_momentum_vector}")
        if impact_type == "elastic":
            alpha = 2
        elif impact_type == "inelastic" or "absorbed":
            alpha = 1
        else:
            ValueError("Invalid impact type")
        total_linear_momentum = np.zeros(3)
        total_angular_momentum = np.zeros(3)
        for n in range(n_particles):  # TODO: can be vectorised (seriously, do this)
            particle_velocity = particle_velocity_vectors[n, :]
            panel_idx = impact_panel_indices[n]
            r = moment_arms[n]
            panel_normal = panel_normals[panel_idx, :]
            # print(f"panel_normal: {panel_normal}")

            p_normal_to_panel = np.dot(particle_momentum_vector[n,:], panel_normal)*panel_normal
            lin_mom_to_cubesat = alpha*p_normal_to_panel
            total_linear_momentum += lin_mom_to_cubesat
            total_angular_momentum += np.cross(r, lin_mom_to_cubesat)
            particle_velocity_out = particle_velocity - 2*np.dot(particle_velocity,panel_normal)*panel_normal
            #TODO: Implement second collision check/simulation
        return total_linear_momentum, total_angular_momentum

    def visualise_3d(self,
                     show_velocity_vector: bool = False,
                     impacts: np.ndarray = None):
        satellite = self.sat
        satellite.visualise(show_velocity_vector=show_velocity_vector, impacts=impacts)
        plt.show()

