Two Views
=========
This is an exercise to check my understanding of photogrammetry concepts and C++ in general. I've implemented something akin to PMVS2 for the special case of rectified stereo images from the Middlebury Stereo Evaluation dataset. Feel free to reuse this code in any way you want.

Example
=======
The reconstruction below was generated from the "Crusade" image pair from [this dataset](https://vision.middlebury.edu/stereo/data/scenes2014/). The point cloud file is included in the sample/ folder.

![Final reconstruction](/sample/optimized2.png)

The patch expansion and optimization process results in a much denser point cloud, compared to the initial triangulation:

![Initial triangulation](/sample/initial.png)

The effect of the optimization step can be seen in the top view of the reddish planar surface, where the point cloud becomes more tightly aligned (right):

![Top view comparison, before and after optimization](/sample/initial_vs_optimized1.png)

Setup Notes
===========
The C++ implementation should work cross-platform, but I've configured CMakeLists.txt for Windows.

 1. Download and install Visual Studio 2019
 2. Follow the instructions [here](https://docs.opencv.org/master/d3/d52/tutorial_windows_install.html) to build OpenCV 4.5.3 from source, but change the script to use generator "Visual Studio 16 2019".
 3. Download the half-resolution 2014 dataset from the [Middlebury Stereo Evaluation](https://vision.middlebury.edu/stereo/submit3/) page, and extract into this folder.
 4. Create a `build` folder here and run `cmake . -B build`
 5. Run `cmake --build build --config Release --target ALL_BUILD -j 10 --`
 6. Find `opencv_world453.dll` in the OpenCV build folder and copy it into `build/Release/`

After running the program, I recommend using [CloudCompare](https://www.danielgm.net/cc/) to view the generated point clouds.
