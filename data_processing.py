import matplotlib.pyplot as plt

def plot_angular_momentum(state):
    time = state[:,0]
    angular_momentum = state[:,1:4]
    plt.plot(time, angular_momentum[:,0])
    plt.plot(time, angular_momentum[:,1])
    plt.plot(time, angular_momentum[:,2])
    plt.show()
