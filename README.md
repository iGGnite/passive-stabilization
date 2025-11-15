# CubeSat Passive Attitude Stabilisation
Simulator in development to evaluate stability and dynamics of Earth-orbiting CubeSat with moveable aft solar panels.
More information on the [Wiki](https://github.com/iGGnite/passive-stabilization/wiki/About).

### Currently contains: 
- Definition of CubeSat geometry, creating satellite object from config file
- Automatic option to calculate of CubeSat inertia as a function of panel angle
- Simulation of particles impacting CubeSat, found by generating particles in a plane normal to the direction of flight
- Propagation of CubeSat attitude in time, currently limited to forward Euler

### Roadmap
- [x] Enable satellite/simulation settings to be loaded from (custom) settings files
  - [x] File defining satellite geometry, mass, and inertia properties
  - [x] File defining simulation 
- [ ] Varying types of particle interactions beyond basic momentum exchange
- [ ] Implementing integration schemes other than forward Euler
- [ ] Improve performance
  - [ ] Determine (through MC analysis), minimum number of particle impacts to mimic 'flow' 
  - [ ] Sort particle impacts according to panels
  - [x] Generate particles uniformly in plane normal to direction of flight
- [ ] Adding orbital mechanics
  - [ ] To consider particles' linear momentum for change in orbital parameters
  - [ ] To vary direction of incoming particles over the course of multiple orbits
- [ ] Add ability to implement control systems, e.g.
  - [ ] Control system manipulating angles of individual panels, like airbrakes
  - [ ] Internal active attitude control (reaction wheels?)
