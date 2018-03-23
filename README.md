[![Documentation Status](https://readthedocs.org/projects/hydra-documentation/badge/?version=latest)](http://hydra-documentation.readthedocs.io/en/latest/?badge=latest)

-----------

<img src="logo_Hydra.png" width="500">

Table of Contents
=================

  * [What is it?](#what-is-it)
  * [Main features](#main-features)
  * [Hydra and Thrust](#hydra-and-thrust)
  * [Supported Parallel Backends](#supported-parallel-backends)
  * [The Latest Version](#the-latest-version)
  * [Documentation](#documentation)
    * [Reference manual](#reference-manual)
    * [Users guide](#users-guide)
  * [Installation and requirements](#installation-and-requirements)
  * [Examples](#examples)
  * [Licensing](#licensing)
  * [Contact the developers](#contact-the-developers)
  * [Author](#author)
  * [Acknowledgement](#acknowledgement)
  

What is it?
-----------

Hydra is a C++11 compliant and header only framework designed to perform commom data analysis tasks on massivelly parallel platforms. Hydra provides a collection of containers and algorithms commomly used in HEP data analysis, which can deploys transparently OpenMP, CUDA and TBB enabled devices, allowing the user to re-use the same code across a large range of available multi-core CPU and accelerators. The framework design is focused on performance and precision. 

The core algorithms follow as close as is possible the implementations widely used in framworks like ROOT and libraries 
like GSL.

Main features
-------------

Currently Hydra supports:

* Generation of phase-space Monte Carlo samples with any number of particles in the final states. Sequential decays, calculation of integrals of models over the corresponding phase-space and production of weighted and unweighted samples, which can be flat or distributed following a model provided by the user.  
* Sampling of multidimensional pdfs.
* Multidimensional maximum likelihood fits using binned and unbinned data sets.
* Calculation of S-Plots, a popular technique for statistical unfolding of populations contributing to a sample.   
* Evaluation of multidimensional functions over heterogeneos data sets. 
* Numerical integration of multidimensional functions using self-adaptive Monte Carlo and quadrature methods.
* Multidimensional sparse and dense histograming of large samples. 

Hydra also provides a bunch of custom types, optimized containers and number of algorithms and constructs to maximaze performance, avoiding unecessary usage of memory and without losing the flexibility and protability to compile and run the same code across different platforms and deployment scenarios. 
For example the plots below 


For example, just changing .cu to .cpp in any source code writen only using the Hydra and standard C++11 is enough
to compile your application for OpenMP or TBB compatible devices using GCC or other compiler in a machine without a NVIDIA GPU installed.

In summary, using Hydra, the user can transparently typical bottle-neck calculations to a suitable parallel device and get speed-up factors ranging from some dozens to hundeds.  



Hydra and Thrust
----------------

Hydra is implemented on top of the [Thrust library](https://thrust.github.io/) and rely strongly on Thrust's containers, algorithms and backend managment systems.
The official version of Thrust supports tuples with maximum ten elements. In order to overcome this limitation, Hydra uses the 
[unoficial version, forked from the original, by Andrew Currigan and collaborators](https://github.com/andrewcorrigan/thrust-multi-permutation-iterator). 
This version implements variadic tuples and related classes, as well as provides some additional functionalities, which are missing in the official Thrust.

The version of Thrust distributed with Hydra is maintained by [MultithreadCorner](https://github.com/MultithreadCorner). It is basically a fork of Currigan's repository, which was merged with the latest official release available in github (Thrust 1.8.3). 

***Hydra does not depend or conflits with the official Thrust library distributed with the CUDA-SDK.***


Supported Parallel Backends
---------------------------

Hydra uses the underlying Thrust's "backend systems" to control how the algorithms algorithms get
mapped to and executed on the parallel processors and accelerators available to a given application. 
When is necessary, the backends can be specified using the symbols hydra::device::sys_t, hydra::host::sys_t,
hydra::omp::sys_t, hydra::tbb::sys_t, hydra::cuda::sys_t, hydra::cpp::sys_t.
The backends hydra::device::sys_t and hydra::host::sys_t are selected in compile time using the macros ```HYDRA_HOST_SYSTEM``` and ```HYDRA_DEVICE_SYSTEM```.
The following possibilities are available:
 
* host: CPP, OMP, TBB
* device: CPP, OMP, TBB, CUDA

For example, this will compile ```my_program.cu``` using OpenMP as host backend and CUDA as device backend:

```bash
nvcc -Xcompiler -fopenmp -DHYDRA_HOST_SYSTEM=OMP -DHYDRA_DEVICE_SYSTEM=CUDA  my_program.cu
```
The available "host" and "device" backends can be freely combined. 
Two important features related to the Hydra's design and the backend configuration:

* If CUDA backend is not used, [NVCC and the CUDA runtime](https://developer.nvidia.com/cuda-toolkit) are not necessary. The programs can be compiled with GCC, CLang or other host compiler compatible with C++11 directly.
* Programs written using only Hydra, Thrust, STL and standard c++ constructs, it means programs without any raw CUDA code or calls to the CUDA runtime API, can be compiled with NVCC, to run on CUDA backends, or a suitable host compiler to run on OpenMP , TBB and CPP backends. **Just change the source file extention from .cu to .cpp, or something else the host compiler understands.**        


The Latest Version
------------------

Documentation
-------------

### Reference manual

The complete and updated [Doxygen](http://www.doxygen.org/) source code documentation in HTML format is available at the
[Reference documentation](https://multithreadcorner.github.io/Hydra/index.html) web-page.
It is also possible to browse the documentation by class, file or name using the links:

1.[classes](http://multithreadcorner.github.io/Hydra/classes.html)
2.[files](http://multithreadcorner.github.io/Hydra/files.html)
3.[names](http://multithreadcorner.github.io/Hydra/namespacemembers.html)


### User's Guide

The Hydra User's Guide is available in the following formats:

* [HTML](https://hydra-documentation.readthedocs.io/en/latest/)
* [PDF](https://readthedocs.org/projects/hydra-documentation/downloads/pdf/latest/)
* [EPUB](https://readthedocs.org/projects/hydra-documentation/downloads/epub/latest/)

Installation and requirements 
-----------------------------

Hydra is a header only library, so no build process is necessary to install it. 
Just place the `hydra` folder and its contents where your system can find it.
The library run on Linux systems and requires at least a host compiler supporting C++11. To use NVidia's GPUs, CUDA 8 or higher is required.  
A suite of examples demonstrating the basic features of the library are included in the `examples` folder.
All the examples are organized in .inl files, which implements the `main()` function. These files are included by .cpp and .cu
files, which are compiled according with the availability of backends. TBB and CUDA backends requires the a installation of the corresponding libraries and runtimes.
These code samples uses, but does not requires [ROOT](https://root.cern.ch/) for graphics, and [TCLAP](http://tclap.sourceforge.net/) library for process command line arguments. 

Examples
--------
The examples are built using [CMAKE](https://cmake.org/) following the following instructions:

1. clone the git repository: `git clone https://github.com/MultithreadCorner/Hydra.git`
2. go to the Hydra repository: `cd Hydra`  
3. create a build directory: `mkdir build` 
4. go to build directory: `cd build`
5. `cmake ..`
6. `make`


The compiled examples will be placed in the build/examples folder. The sub-directories are named according to the functionalities they ilustrates.

The examples are listed below:

1. __async__ : async_mc
2. __fit__ : basic_fit, multidimensional_fit, extended_logLL_fit, fractional_logLL_fit, phsp_unweighting_functor_and_fit, splot
3. __histograming__ : dense_histogram, sparse_histogram
4. __misc__ : multiarray_container, multivector_container, variant_types
5. __numerical_integration__ : adaptive_gauss_kronrod, gauss_kronrod, plain_mc, vegas
6. __phase_space__ : phsp_averaging_functor, phsp_evaluating_functor, phsp_reweighting, phsp_basic, phsp_unweighting, phsp_chain, phsp_unweighting_functor
7. __random__ :  basic_distributions, sample_distribution
8. __root_macros__ :  macros to run examples in ROOT

Each compiled example executable will have an postfix (ex.: _cuda, _omp, _tbb) to indicate the deployed device backend.  
All examples use CPP as host backend. 


Recent publications and presentations at conferences and workshops
------------------------------------------------------------------

1. [A. A. Alves Junior, *Hydra: a C++11 framework for data analysis in massively parallel platforms*, Proceedings of the 18th International Workshop on Advanced Computing and Analysis Techniques in Physics Research, 21-25 August 2017 Seattle,USA](https://inspirehep.net/record/1636201/files/arXiv:1711.05683.pdf),
2. [A. A. Alves Junior, *Hydra: Accelerating Data Analysis in Massively Parallel Platforms* - University of Washington, 21-25 August 2017, Seattle](https://indico.cern.ch/event/567550/contributions/2638690/)
3. [A. A. Alves Junior, *Hydra: A Framework for Data Analysis in Massively Parallel Platforms* - NVIDIA’s GPU Technology Conference, May 8-11, 2017 - Silicon Valley, USA]()
4. [A. A. Alves Junior, *Hydra* - HSF-HEP analysis ecosystem workshop, 22-24 May 2017 Amsterdam, Netherlands](https://indico.cern.ch/event/613842/)
5. [A. A. Alves Junior, *MCBooster and Hydra: two libraries for high performance computing and data analysis in massively parallel platforms* -Perspectives of GPU computing in Science September 2016, Rome, Italy](http://www.roma1.infn.it/conference/GPU2016/pdf/talks/AlvesJr.pdf)

Licensing
---------

Hydra is released under the [GNU General Public License version 3](http://www.gnu.org/licenses/gpl-3.0.en.html). 
Please see the file called [COPYING](https://github.com/MultithreadCorner/Hydra/blob/master/COPYING).

Contact the developers
----------------------
Here’s what you should do if you need help or would like to contribute:

1. If you need help or would like to ask a general question, subscribe and use https://groups.google.com/forum/#!forum/hydra-library-users.
2. If you found a bug, use GitHub issues.
3. If you have an idea, suggestion of whatever, use GitHub issues.
4. If you want to contribute, submit a pull request https://github.com/MultithreadCorner/Hydra.

Author
--------

Hydra was created and is mantained by [Antonio Augusto Alves Jr](@AAAlvesJr).

Acknowledgement
---------------

Hydra's development has been supported by the [National Science Foundation](http://nsf.gov/index.jsp) 
under the grant number [PHY-1414736](http://nsf.gov/awardsearch/showAward?AWD_ID=1414736). 
Any opinions, findings, and conclusions or recommendations expressed in this material are those of 
the developers and do not necessarily reflect the views of the National Science Foundation.
