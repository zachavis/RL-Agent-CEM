To run all programs:

Must be on Windows 10 64 bit.
Must have Visual Studio 2019
it may be the case that you can directly run the built binaries in x64/release to see training i.e. the reward increase and tests run.

For CEM CUDA:
Must have CUDA 10.1 with support for visual studio 2019 installed, and added to system path
Although it isn't used, must download Eigen library: http://eigen.tuxfamily.org/index.php?title=Main_Page
In project properties, in config properties, in VCC++ directories, set the path for your eigen download. The default is C:\Toolkits\eigen-3.3.7\
build and run, however, I didn't have luck running from visual studio, and had to navigate to the build directory and run from the command line.

For CEM Vehicle Update:
Must download Eigen library: http://eigen.tuxfamily.org/index.php?title=Main_Page
In project properties, in config properties, in VCC++ directories, set the path for your eigen download. The default is C:\Toolkits\eigen-3.3.7\
build and run

For MotionPlanning.ipynb simulator/visualizer
must pip install numpy and matplotlib
must have file called "dd_parameters.m" in the same directory
if CPU params, set USE_CPU to True, else False
must load via python notebook software such as jupyter
run all cells but the final, unless you want a video saved.

