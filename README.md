## THIS IS RESEARCH CODE AND IS NOT OPTIMISED
This is the code that accompanies "A Distributed Multi-Robot Framework for Exploration, Information Acquisition and Consensus" by Aalok Patwardhan and Andrew J Davison, presented at IEEE ICRA 2024.

[Paper](https://arxiv.org/abs/2310.01930)

[Project Page](https://aalpatya.github.io/gbpstack/)

## Initial Setup
Install Raylib dependencies as mentioned in https://github.com/raysan5/raylib#build-and-installation
This will be platform dependent

[OpenMP](https://www.openmp.org/) is required (included with gcc in Linux, you may need to install on other platforms)

Clone the repository *with the submodule dependencies:*
```shell
git clone https://github.com/aalpatya/gbpstack.git --recurse-submodules
cd gbpstack
```
Use CMAKE to set up the build environment and then run 'make':
```shell
mkdir build
cd build
cmake ..
make
```

## Run examples
Make any changes to the simulations you want in config/config.json and then run:
```shell
./gbpstack
```

Examples config files are in config directory, and include:
- ```config.json``` (default: robots travel to the opposite sides of a circle in free space)
- ```circle_cluttered.json``` (robots travel to the opposite sides of a circle around some obstacles)

Run the simulation:
```shell
./gbpstack --cfg ../config/circle_cluttered.json
```

Or create your own config_file.json and run:
```shell
./gbpstack --cfg ../config_file.json
```

### During simulation
**Press 'H' to display help and tips!**

Use the mouse wheel to change the camera view (scroll : zoom, drag : pan, shift+drag : rotate)

Press 'spacebar' to transition between camera keyframes which were set in ```src/Graphics.cpp```.

## Play with the code
Edit the parameters in the config files and see the effects on the simulations!

### Own formations
You may want to create your own formations and scenarios for the robots.

Towards the end of ```src/Simulator.cpp``` there is a function called ```createOrDeleteRobots()```, where you may add your own case.


