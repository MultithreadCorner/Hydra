
Introduction
============

Hydra framework
---------------

Despite the ongoing efforts of modernization, a large fraction of the software used in HEP remain based on legacy. It mostly consists of libraries assembling single threaded,
Fortran and C++03 mono-platform routines :cite:`cernlib`. Concomitantly, HEP experiments keep collecting samples with unprecedented large statistics and
data analyses become increasingly complex. Are not rare the situations where computers spend days performing calculations to reach a result, which very often needs re-tune.

On the other hand, computer processors will not increase clock frequency any more in order to reach higher performance. Indeed, the current road-map to improve overall
performance is to deploy different levels of concurrency, which for example has been leading to the proliferation of multi-thread friendly and multi-platform environments
among HPC data-centers. Unfortunately, HEP software is not completely prepared yet to fully exploit concurrency and to deploy more opportunistic computing strategies.

The Hydra framework proposes a computing model to approach these issues. The Hydra provides collection of parallelized high-level algorithms, addressing some of of typical computing bottlenecks commonly found in HEP, and a set of optimized containers and types, through a modern and functional interface, allowing to enhance HEP software productivity and performance and at same time keeping the portability between NVidia GPUs, multi-core CPUs and other devices compatible with CUDA :cite:`cuda`, TBB :cite:`tbb` and OpenMP :cite:`openmp` computing models.

Design highlights
-----------------

Hydra is basically a header-only C++11 template framework organized using a variety of static polymorphism idioms and patterns. This ensure the predictability of the stack at compile time, which is critical for stability and performance when running on GPUs and minimizes the overhead introduced by the user interface when engaging the actual calculations. Furthermore, the implementation of static polymorphism via extensive usage of templates
allows to expose the maximum amount of code to the compiler, in the context in which the code will be used, contributing to activate many compile time optimizations that could not be accessible otherwise. Hydra's interface and implementation details extensively deploys patterns and idioms
that enforce thread-safeness and efficient in memory access and management. The following list summarizes some of the main design choices adopted in Hydra:


  * Hydra provides a set of optimized STL-like containers that can store multidimensional datasets using :cite:`soa`  layout.
  * Data handled using iterators and all classes manages resources using RAII idiom.
  * The framework is type and thread-safe.
  * There is no limitation on the maximum number of dimensions that containers and algorithms can handle.

The types of devices which Hydra can be deployed are classified by back-end type, according with the device compatibility with certain computing models.
Currently, Hydra supports four back-ends, which are CPP :cite:`cpp`, OpenMP :cite:`openmp`, CUDA :cite:`cuda` and TBB :cite:`tbb`. Code can be dispatched and executed in all supported back-ends concurrently and asynchronously in the same program, using the suitable policies represented by the symbols ``hydra::omp::sys`` , ``hydra::cuda::sys``, ``hydra::tbb::sys``, ``hydra::cpp::sys`` , ``hydra::host::sys`` and ``hydra::device::sys``. Where applicable, these policies define the memory space where resources should be allocated to run algorithms and store data.

For mono-backend applications, source files written using Hydra and standard C++ compile for GPU and CPU just exchanging the extension from .cu to .cpp and one or two compiler flags. So, basically, there is no need to refractory code to deploy different back-ends.

Basic features
--------------

Currently, Hydra provides collection of
parallelized high-level algorithms, addressing some computing-intensive tasks commonly found in data analyses in HEP.
The available high-level algorithms are listed below,

 * Interface to **Minuit2** minimization package :cite:`minuit`, allowing to accelerate maximum likelihood fits over multidimensional large data-sets.
 * Parallel implementation of the **SPlot** technique, a very popular procedure for statistical unfolding of data distributions :cite:`splot` .
 * Phase-space Monte Carlo generation, integration and modeling.
 * Multidimensional p.d.f. sampling.
 * Parallel function evaluation on multidimensional data-sets.
 * Five fully parallelized numerical integration algorithms: Genz-Malik :cite:`genzmalik,berntsen`, self-adaptive and static Gauss-Kronrod quadratures,
   plain, self-adaptive importance sampling and phase-space Monte Carlo integration.

How does this manual is organized?
----------------------------------

By the time it was written, this manual covers the usage of most of the Hydra features. This manual was written to be read sequentially.
The sections are organized by subject and are sorted to make available the functionality described in a given section usable in the next parts.

