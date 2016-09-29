Hydra
=========

<img src="Hydra.jpg" width="200">

What is it?
-----------
Hydra is an header only library designed for data analysis. The library makes use of Thrust and can deploy OpenMP
threads, CUDA and Xeon Phi cores. It is focused on performance and precision. 

The library core algorithms follow as close as is possible the implementations widely available in frameworks like ROOT, and libraries 
like GSL.

Main features
-------------

Currently Hydra supports:

* Generation of phase-space Monte Carlo Samples with any number of particles in the final states.
* Generation of sequential decays (phase-space).
* Sampling of multidimensional pdfs.
* Multidimensional fitting using binned and unbinned data sets.
* Evaluation of multidimensional functions over heterogeneous data sets. 
* Numerical integration of multidimensional functions using plain or self-adaptive (Vegas-like) Monte Carlo methods.

Hydra also provides a bunch of custom types, containers and an increasing number of algorithms
to maximize performance, avoid unnecessary usage of memory and grant flexibility and portability between 
host and device calculations and deployment scenarios. 

Just changing `.cu` to `.cpp` in any source code written only using the Hydra constructs is enough
to compile your application for OpenMP using GCC in a machine without a NVIDIA GPU installed.

Many other possibilities and functionalities can be achieved combining the core components, bounded only by the creativity of the users. 

The Latest Version
------------------

The latest version can be found on the 
[project releases page](https://github.com/MultithreadCorner/Hydra/releases).

Documentation
-------------

The complete and updated [Doxygen](http://www.doxygen.org/) source code documentation of this release is available in HTML format on the
[reference manual](http://multithreadcorner.github.io/Hydra/) webpage.
Users can also browse the documentation by class, file or name using the following links:

1.[classes](http://multithreadcorner.github.io/Hydra/classes.html)

2.[files](http://multithreadcorner.github.io/Hydra/files.html)

3.[names](http://multithreadcorner.github.io/Hydra/namespacemembers.html)

Installation and requirements 
-----------------------------

Hydra is a header only library, so no build process is necessary to install it. 
Just place the `hydra` folder and its contents where your system can find it.
The library run on Linux systems and requires C++11 and the varidadic version of the 
[Thrust library](https://github.com/andrewcorrigan/thrust-multi-permutation-iterator/tree/variadic) and [boost::format](http://www.boost.org/doc/libs/1_61_0/libs/format/doc/format.html). 

Some examples demonstrating the basic features of the library are included in the `src` folder. 
These code samples require [ROOT](https://root.cern.ch/) and [TCLAP](http://tclap.sourceforge.net/) library. 
CUDA based projects will require a local installation of [CUDA Tookit](https://developer.nvidia.com/cuda-toolkit)
 with version 6.5 or higher.   
Alternatively, projects targeting [OpenMP](http://openmp.org/wp/) backend can be compiled with either nvcc or gcc. 
The CUDA installation is not required to use OpemMP. 

Examples
--------

Some example code samples demonstrating the basic usage of the library are stored in the `src` directory, in the project source tree. 
These samples can be built using [CMAKE](https://cmake.org/) according the following instructions:

1. clone the git repository: `git clone https://github.com/MultithreadCorner/Hydra.git`
2. go to Hydra directory: `cd Hydra`
3. create a build directory: `mkdir build` 
4. go to build directory: `cd build`
4. `cmake ../`
5. `make`


The examples are named according to the convention `HYDRA_Example_<BACKEND AND COMPILER>_<EXAMPLE NAME>`. To run an example do `./example-name`.
The examples are described below:

1. __PhaseSpace__ : Takes arguments from the command line and generates a 3-body decay and calculates some observables.   
The program print some events and timing information to stdout.

2. __Evaluate__ : Takes arguments from the command line, generates some samples and perform calculations 
using lambda functions (requires CUDA 8.0 to run on the GPU). 
The program print some results and timing information to stdout.

3. __Fit__: Takes arguments from the command line, generates a samples and perform a extended likelihood fit. 
The program print some results and timing information to stdout.

4. __Random__: Takes arguments from the command line, generates some samples  in one, two and three. 
The program print some results, draw plots and timing information to stdout.


Licensing
---------

Hydra is released under the [GNU General Public License version 3](http://www.gnu.org/licenses/gpl-3.0.en.html). 
Please see the file called [COPYING](https://github.com/MultithreadCorner/Hydra/blob/master/COPYING).

Contact the developers
----------------------
Here’s what you should do if you need help or would like to contribute:

1. If you need help or would like to ask a general question, subscribe and use https://groups.google.com/forum/#!forum/hydra-libray-users.
2. If you found a bug, use GitHub issues.
3. If you have an idea, use GitHub issues.
4. If you want to contribute, submit a pull request.

Author
--------

Hydra was created and is maintained by [Antonio Augusto Alves Jr](@AAAlvesJr).

Acknowledgement
---------------

Hydra's development has been supported by the [National Science Foundation](http://nsf.gov/index.jsp) 
under the grant number [PHY-1414736](http://nsf.gov/awardsearch/showAward?AWD_ID=1414736). 
Any opinions, findings, and conclusions or recommendations expressed in this material are those of 
the developers and do not necessarily reflect the views of the National Science Foundation.
