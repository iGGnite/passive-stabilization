import numpy as np
import matplotlib.pyplot as plt

class Panel:
    def __init__(self, length: float, width: float, normal_vector: np.array, forward_vector: np.array, body: bool = True):
        A_inv = np.matrix(
            [[1, 0, 0],
             [0, 1, 0],
             [0, 0, 1]]
        )
        self.length = length
        self.width = width
        self.n = normal_vector
        self.f = forward_vector
        ### Panel origin in centre of panel ###
        self.local_nodes = np.array((
            [-self.length/2, self.width/2,  0],
            [-self.length/2, -self.width/2, 0],
            [ self.length/2, self.width/2,  0],
            [ self.length/2, -self.width/2, 0],
        ))
        n_cross_f = np.cross(self.n, self.f)
        B = np.matrix(np.vstack((self.f, n_cross_f, self.n))).T
        self.initial_forward_vector = np.array([1, 0, 0])
        self.initial_normal_vector = np.array([0, 0, 1])
        new_normal_vector = B @ self.initial_normal_vector
        new_forward_vector = B @ self.initial_forward_vector
        self.local_nodes = (B @ self.local_nodes.T).T

    def define_global_nodes(self, position_vector: np.array):
        self.global_nodes = np.array(self.local_nodes + position_vector)

class Satellite:
    def __init__(self, x_len: float, y_width: float, z_width: float, panel_length: float, panel_width: float,
                 panel_angles: np.array, inertia: np.array, com: np.array = np.array([-.5, 0, 0])):
        self.x_len = x_len
        self.y_width = y_width
        self.z_width = z_width
        self.panel_length = panel_length
        self.panel_width = panel_width
        for idx, panel_angle in enumerate(panel_angles):
            while panel_angle < 0 or panel_angle > 90:
                ValueError("Panel angle must be between 0 and 90 degrees, but is " + str(panel_angle) + " degrees.")
                panel_angle = np.deg2rad(input("Provide a new panel angle in degrees:"))
            else:
                self.panel_angles = np.deg2rad(panel_angles)
        self.com = com
        self.inertia = inertia
        self.panel_hinges = [
            np.array([-x_len, y_width / 2, 0]),
            np.array([-x_len, 0, z_width / 2]),
            np.array([-x_len, -y_width / 2, 0]),
            np.array([-x_len, 0, -z_width / 2]),
        ]
        self.body_panels = []
        ### COORDINATE SYSTEM STARTS AT TIP OF SPACECRAFT ###
        self.body_panel_centres = [
            np.array([0, 0, 0]), np.array([-self.x_len, 0, 0]),
            np.array([-self.x_len/2, self.y_width/2, 0]), np.array([-self.x_len/2, -self.y_width/2, 0]),
            np.array([-self.x_len/2, 0, self.z_width/2]), np.array([-self.x_len/2, 0, -self.z_width/2]),
        ]
        self.body_forward_vectors = [
            np.array([0, 0, 1]), np.array([0, 0, 1]),
            np.array([1, 0, 0]), np.array([1, 0, 0]),
            np.array([1, 0, 0]), np.array([1, 0, 0]),
        ]
        self.body_normal_vectors = [
            np.array([1, 0, 0]), np.array([-1, 0, 0]),  # x forward, x backward
            np.array([0, 1, 0]), np.array([0, -1, 0]),  # y right, y left
            np.array([0, 0, 1]), np.array([0, 0, -1]),  # z down, z up
        ]
        self.panel_nodes = []
        for idx, panel_center in enumerate(self.body_panel_centres):
            if idx < 2:  # front and back
                panel = Panel(self.y_width, self.z_width, self.body_normal_vectors[idx], self.body_forward_vectors[idx])
            elif 1 < idx < 4:  # sides
                panel = Panel(self.x_len, self.y_width, self.body_normal_vectors[idx], self.body_forward_vectors[idx])
            else:  # top and bottom
                panel = Panel(self.x_len, self.z_width, self.body_normal_vectors[idx], self.body_forward_vectors[idx])
            panel.define_global_nodes(panel_center)
            self.panel_nodes.append(panel.global_nodes)
        # print(self.panel_nodes)


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

        # if com == np.array([-.5, 0, 0]):
        #     new_com()

    # def visualise(self):



if __name__ == "__main__":
    satellite = Satellite(1, .3, .3, .8, .3,
                          np.array([30, 30, 30, 30]), None)
    satellite.print_nodes()