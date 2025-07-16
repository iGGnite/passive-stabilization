from spacecraft_body import *

#TODO: Determine global coordinate frame to determine direction of incoming particles
def projected_area(panel: Panel, C_ib, velocity_vector_inertial_frame: np.ndarray):
    panel_nodes = panel.body_nodes
    velocity_vector_body_frame = C_ib @ velocity_vector_inertial_frame
    velocity_unit_vector = velocity_vector_body_frame / np.linalg.norm(velocity_vector_body_frame)
    for node in panel_nodes:
        projected_nodes = node - np.dot(node, velocity_unit_vector) * velocity_unit_vector
        print(projected_nodes)

#TODO: Project panels onto normal plane of atmospheric velocity vector
#TODO: Determine whether panel has normal vector component along velocity vector wrt atmosphere
#TODO: Consider area summation as panels cover one another