# CubeSat Passive Attitude Stabilisation
Simulator in development to evaluate stability and dynamics of Earth-orbiting CubeSat with moveable aft solar panels.
More information on the [Wiki](https://github.com/iGGnite/passive-stabilization/wiki/About).

### Currently contains: 
- Definition of CubeSat geometry, creating satellite object from config file
- Automatic option to calculate of CubeSat inertia as a function of panel angle
- Simulation of particles impacting CubeSat, found by generating particles in a plane normal to the direction of flight
- Propagation of CubeSat attitude in time

### Roadmap
- [x] Enable satellite/simulation settings to be loaded from (custom) settings files
  - [x] File defining satellite geometry, mass, and inertia properties
  - [x] File defining simulation parameters, like timestep, duration, initial attitude, number of particles, and atmospheric density
- [ ] Vary type of particle/plane interaction beyond elastic/inelastic momentum exchange
- [ ] Improve performance
  - [x] Generate particles uniformly in 'shadow plane' normal to direction of flight
  - [x] Minimize size of 'shadow plane'
  - [ ] Determine (through MC analysis), relation between time-step size, number of particles, and accuracy
- [ ] Simulate orbital travel
  - [ ] Vary direction of incoming particles over the course of multiple orbits
  - [ ] Consider particles' linear momentum for change in orbital parameters
(Far future)
- [ ] Add ability to implement control systems, e.g.
  - [ ] Control system manipulating angles of individual panels, like airbrakes
  - [ ] Active attitude control (reaction wheels)