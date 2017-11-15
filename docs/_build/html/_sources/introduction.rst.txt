Introduction
============

Hydra framework
---------------

Despite the ongoing efforts of modernization, a large fraction of the software used in HEP is legacy.
It mostly consists of libraries assembling single threaded,
Fortran and C++03 mono-platform routines. Concomitantly, HEP experiments keep collecting samples with unprecedented large statistics and
data analyses become increasingly complex. Are not rare the situations where computers spend days performing
calculations to reach a result, which very often needs re-tune.

On the other hand, computer processors will not increase clock frequency any more in order to reach higher performance. Indeed, the current road-map to improve overall
performance is to deploy different levels of concurrency, which for example has been leading to the proliferation of multi-thread friendly and multi-platform environments
among HPC data-centers. Unfortunately, HEP software is not completely prepared yet to fully exploit concurrency and to deploy more opportunistic computing strategies.

Hydra proposes a computing model to approach these issues. The Hydra framework provides collection of
parallelized high-level algorithms, addressing some of of typical computing bottlenecks commonly found in HEP,
and a set of optimized containers and types, through a modern and functional interface, allowing to enhance HEP software productivity and
performance, keeping the portability between NVidia GPUs, multicore CPUs and other devices compatible with CUDA, TBB and OpenMP computing models.

Design highlights
-----------------

Hydra is basically a C++11 template framework organized using a variety of static polymorphism idioms and patterns. This ensure the predictability of the stack size at compile time,
which is critical for stability and performance when running on GPUs and minimizes the overhead introduced by the user interface
when engaging the actual calculations. Furthermore, the combination of static polymorphism and templates
allow to expose the maximum amount of code to the compiler, in the context in which the code will be used, contributing to activate many
compile time optimizations that could not be accessible otherwise. Hydra's interface and implementation details extensively deploys patterns and idioms
that enforce thread-safeness and efficient memory access and management. The following list summarizes some of the main design choices adopted in Hydra:


  * Hydra provides a set of optimized STL-like containers that can store multidimensional datasets using SoA [#soa]_.
  * Data handled using iterators and all classes manages resources using RAII idiom.
  * The framework is type and thread-safe.
  * There is no limitation on the maximum number of dimensions that containers and algorithms can handle.


The types of devices which Hydra can be deployed are classified by back-end type, according with the device compatibility with certain computing models.
Currently, Hydra supports four back-ends, which are CPP, OpemMP, CUDA and TBB. Code can be dispached and executed in all supported back-ends concurrently and asynchronously
in the same program, using the suitable policies represented by the symbols ``hydra::omp::sys`` , ``hydra::cuda::sys``, ``hydra::tbb::sys``, ``hydra::cpp::sys`` , ``hydra::host::sys``
and ``hydra::device::sys``. These policies define the memory space where resources should be allocated to run algorithms and store data.

For mono-backend applications, source files written using Hydra and standard C++ compile for GPU and CPU just
exchanging the extension from .cu to .cpp and one or two compiler flags. There is no need to
refractory code.

Basic features
--------------

Currently, Hydra provides collection of
parallelized high-level algorithms, addressing some computing-intensive tasks commonly found in data analyses in HEP.
The available high-level algorithms are listed below,

 * Interface to ``ROOT::Minuit2`` [#minuit]_ minimization package, allowing to accelerate maximum likelihood fits over multidimensional large data-sets.
 * Parallel implementation of the S-Plots [#splot]_ technique prescription, for statistical unfolding data distributions.
 * Phase-space Monte Carlo generation, integration and modeling.
 * Multidimensional p.d.f. sampling.
 * Parallel function evaluation on multidimensional data-sets.
 * Five fully parallelized numerical integration algorithms: Genz-Malik, self-adaptive and static Gauss-Kronrod quadratures,
   plain, self-adaptive importance sampling and phase-space Monte Carlo integration.

How does this manual is organized?
----------------------------------

The next sections of this manual cover the usage of each Hydra feature. This manual was written to be read sequentialy.
The sections are organized by subject and are sorted to make available the functionality described in a given section
usable in the next parts.

References
----------

.. [#soa] Structure of arrays or SoA is a layout separating elements of a structure into one parallel array per field. This ease the data manipulation with SIMD instructions and, if only a specific field of the structure is needed,
          only this field can to be iterated over, allowing more data to fit onto a single cache line.
          For more information see `Wikipedia SoA page <https://en.wikipedia.org/wiki/AOS_and_SOA>`_ .
.. [#minuit] Minuit2 is a new object-oriented implementation, written in C++, of the popular MINUIT minimization package.
             For more information see `Minuit page <https://root.cern.ch/root/html/MATH_MINUIT2_Index.html>`_ .
.. [#splot] S-Plot: A statistical tool to unfold data distributions. `<https://doi.org/10.1016/j.nima.2005.08.106>`_
