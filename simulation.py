import numpy as np
from spacecraft_body import *

# class SimulationSettings:
#     def __init__(self):


class PassiveStabilization:
    """
    Simulation object
    """
    def __init__(self, settings=None):
        self.sat = Satellite
        self.dt = 0.01  # Time step (s)
        self.simulation_duration = 10  # Simulation time (s)
        self.altitude = 400000  # Altitude (m)
        self.air_density = 1e-10  # Air density (kg/m^3)
        self.single_particle_mass = 1e-10 # Mass of single particle (kg)
        self.particles_per_cubic_meter = self.air_density / self.single_particle_mass
        print(f"Particles per cubic meter: {self.particles_per_cubic_meter}")
        self.vehicle_velocity_vector = np.array([5.3e3, 1300, 300])  # (Orbital) velocity of vehicle (frame not currently defined)
        self.particle_velocity_magnitude = np.linalg.norm(self.vehicle_velocity_vector)  # Total velocity of incoming particles (m/s)
        self.v_particle_b_frame = -self.vehicle_velocity_vector  # Particle appears to move opposite to the spacecraft

    def create_cubesat(self, length: float, width: float, height: float, panel_angles: np.ndarray):
        self.sat = Satellite(length, width, height, panel_angles)
        self.sat.velocity_vector_i = self.v_particle_b_frame  #TODO: These orientation definitions are all messed up

    def run_simulation(self):
        time = np.arange(0, self.simulation_duration, self.dt)
        for dt in time:
            self.simulate_timestep()

    def simulate_timestep(self, visualise_timestep=False):
        swept_volume = self.sat.shaded_area * self.particle_velocity_magnitude * self.dt
        n_particles = int(self.particles_per_cubic_meter * swept_volume)
        print(f"n_particles: {n_particles} in this time step")
        impact_coordinates = np.zeros((n_particles, 3))
        for n in range(n_particles):
            # results = (self.sat.generate_impacting_particle())[0][1]
            impact_coordinates[n,:] = (self.sat.generate_impacting_particle())[0][1]
        if visualise_timestep:
            self.visualise_3d(show_velocity_vector=True,
                              # impacts=impact_coordinates
                              )
        #TODO: Implement calculation of moment arm from center of mass
        # m = self.calculate_moment()
        ...

    def calculate_moment(self, impact_coordinates: np.ndarray = None):
        moment = ...
        return

    def simulate_dynamics(self, moment):
        #TODO: Implement equations of rotational motion
        inertia = self.sat.inertia

    def visualise_3d(self,
                     show_velocity_vector: bool = False,
                     impacts: np.ndarray = None):
        satellite = self.sat
        satellite.visualise(show_velocity_vector=show_velocity_vector, impacts=impacts)
        plt.show()
