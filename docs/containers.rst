Containers
==========

Hydra framework provides an one-dimensional STL-like vector container for each supported back-end, aliasing the underlying Thrust types. The framework also implements two native multidimensional containers called hydra::multivector`` and   ``hydra::multiarray`` .
  
  In these containers, the data corresponding to each dimension is stored in contiguous memory addresses that can be traversed in a CPU/GPU cache friendly way, independently of the other dimensions. In the case of multidimensional containers, when the data is traversed each entry is accessed as 
a ``hydra::tuple`` object, where each field holds a value corresponding to a dimension. 
 
One-dimensional containers
--------------------------

Hydra's one-dimensional containers are aliases to the corresponding [Thrust]_ vectors and are defined for each supported back-end. They are: 

	1. ``hydra::device::vector`` : storage allocated in the device back-end defined at compile time using the macro ``HYDRA_DEVICE_SYSTEM``
	2. ``hydra::host::vector`` : storage allocated in the device back-end defined at compile time using the macro ``HYDRA_HOST_SYSTEM``
	3. ``hydra::omp::vector`` : storage allocated in the [OpenMP]_ back-end. Usually the CPU memory space.  
	4. ``hydra::tbb::vector`` : storage allocated in the [TBB]_ back-end. Usually the CPU memory space.
	5. ``hydra::cuda::vector`` : storage allocated in the [CUDA]_ back-end. The GPU memory space.
	6. ``hydra::cpp::vector`` : storage allocated in the [CPP]_ back-end. Usually the CPU memory 
	
The usage of these containers is extensively documented in STL and [Thrust]_ library. Hydra also implements range-semantics for many of these containers.  

Multi-dimensional containers
----------------------------

Hydra implements two multidimensional containers:``hydra::multivector`` and ``hydra::multiarray``. 
These containers store data using [SoA]_ layout and provides a STL-vector compliant interface.

Both classes provides constant and non-constant accessors for the single dimensional data. The container 
``hydra::multivector`` is suitable to store data-sets where the dimensions are represented by entries with different POD types. ``hydra::multiarray`` is designed to store data-sets where all dimensions are represented by fields of the same type. Data is always copyable across different back-ends and movable between containers on the same back-end. 

The best way to understand how these containers operate is to visualize them as a table, there each row corresponds to a entry and each column to a dimension. The design of ``hydra::multivector`` and ``hydra::multiarray`` makes possible to iterate over the container to access a complete row
or to iterate over one or more columns to access only the data of interest in a given entry, without loading the entire row. 

When the user iterates over the whole container, each entry (row) is returned as a ``hydra::tuple``. If the user iterates over one single column, the entries have the type of the column. If two or more columns are accessed, entry's data is returned as again as  ``hydra::tuple`` containing only the elements of interest. Hydra's multi-dimensional containers can hold any type of data per dimension, but there is not real gain using these containers for describing dimensions with non-POD or non alignable data. 

These containers can store the state of arbitrary objects and perform type conversions on-the-fly, using suitable overloaded iterators and ``push_back()`` methods. 


``hydra::multivector``
......................


``hydra::multivector`` templates are instantiated passing the type list corresponding to each dimension via a ``hydra::tuple`` and the back-end where memory will be allocated. The snippet 
:ref:`below <multivector-example1>` show how to instantiate a ``hydra::multivector`` to store four-dimensional data, two columns for integers and two columns for doubles:

.. code-block:: cpp
	:name: multivector-example1
	
	#include <hydra/device/System.h>
	#include<hydra/multivector.h>

	...

	hydra::multivector<hydra::tuple<int, int, double, double>, hydra::device::sys_t> mvector;

	for(int i=0; i<10;i++){
		mvector.push_back(hydra::make_tuple( i, 2*i, i, 2*i));
	}
    
   	for(auto x:mvector) std::cout << x << std::endl;


this will print in stdout something like it :

.. code-block:: text
	
	(0, 0, 0.0, 0.0)
	(1, 2, 1.0, 2.0)
	(2, 4, 2.0, 4.0)
	...
	(9, 18, 9.0, 18.0)

To access the columns the user needs ``hydra::placeholders``: _0, _1, _2,..., _99; 

.. code-block:: cpp
	:name: multivector-example2
	
	#include <hydra/device/System.h>
	#include<hydra/multivector.h>
	#include<hydra/Placeholders.h>

	using namespace hydra::placeholders;

	...

	hydra::multivector<hydra::tuple<int, int, double, double>, hydra::device::sys_t> mvector;

	for(int i=0; i<10;i++){
		mvector.push_back(hydra::make_tuple( i, 2*i, i, 2*i));
	}
    
   	for(auto x = mvector.begin(_1, _3);
   			 x != mvector.end(_1, _3); ++x )
   			  
   				std::cout << *x << std::endl;

now in stdout the user will get:

.. code-block:: text
	
	(0, 0.0)
	(2, 2.0)
	(4, 4.0)
	...
	(18, 18.0)

Now suppose that one want to interpret the data stored in ``mvector`` as a pair of complex numbers, represented by the types ``hydra::complex<int>`` and ``hydra::complex<double>``. 
It is not necessary to access each field stored in each entry to perform a conversion invoking the corresponding constructors. The next example shows how this can be accomplished in a more elegant way using a lambda function:

.. code-block:: cpp 
		
	#include <hydra/device/System.h>
	#include<hydra/multivector.h>
	#include<hydra/Complex.h>

	...

	hydra::multivector<hydra::tuple<int, int, double, double>, hydra::device::sys_t> mvector;

	for(int i=0; i<10;i++){
		mvector.push_back(hydra::make_tuple( i, 2*i, i, 2*i));
	}
    
   	auto caster = [] __host__ device__ ( hydra::tuple<int, int, double, double>& entry )
   	{

    	hydra::complex<int> cint(hydra::get<0>(entry), hydra::get<1>(entry));
    	hydra::complex<double> cdouble(hydra::get<2>(entry), hydra::get<2>(entry));
    	
    	return hydra::make_pair(  cint, cdouble ); 
    };

   	for(auto x = mvector.begin(caster); x != mvector.end(caster); x++ ) 
   		std::cout << *x << std::endl;

stdout will look like:

.. code-block:: text
	
	((0, 0), (0.0, 0.0))
	((1, 2), (1.0, 2.0))
	((2, 4), (2.0, 4.0))
	...
	((9, 18), (9.0, 18.0))


``hydra::multiarray``
  .................


``hydra::multiarray`` templates are instantiated passing the type and the number of dimensions via and the back-end where memory will be allocated. The snippet 
:ref:`below <multiarray-example1>` show how to instantiate a ``hydra::multiarray`` to store four-dimensional data, two columns for integers and two columns for doubles:

.. code-block:: cpp
	:name: multiarray-example1
	
	#include <hydra/device/System.h>
	#include<hydra/multiarray.h>

	...

	hydra::multiarray<4, double, hydra::device::sys_t> marray;

	for(int i=0; i<10;i++){
		marray.push_back(hydra::make_tuple( i, 2*i, 4*i, 8*i));
	}
    
   	for(auto x:marray) std::cout << x << std::endl;


this will print in stdout something like it :

.. code-block:: text
	
	(0.0, 0.0, 0.0, 0.0)
	(1.0, 2.0, 4.0, 8.0)
	(2.0, 4.0, 8.0, 16.0)
	...
	(9.0, 18.0, 36.0, 72.0)

To access the columns the user can deploy ``hydra::placeholders``: _0, _1, _2...
or use ``unsigned it`` indexes. 

.. code-block:: cpp
	:name: multiarray-example2
	
	#include <hydra/device/System.h>
	#include<hydra/multiarray.h>
	#include<hydra/Placeholders.h>

	using namespace hydra::placeholders;

	...

	hydra::multiarray<4, double, hydra::device::sys_t> marray;

	for(int i=0; i<10;i++){
		marray.push_back(hydra::make_tuple( i, 2*i, i, 2*i));
	}
    
   	for(auto x = marray.begin(_1, _3);
   			 x != marray.end(_1, _3); x++ ) 
   				std::cout << *x << std::endl;

now in stdout the user will get:

.. code-block:: text
	
	(0.0, 0.0)
	(2.0, 8.0)
	(4.0, 16.0)
	...
	(18.0, 72.0)

Now suppose that one want to interpret the data stored in mvector as a pair of complex numbers, represented by the types ``hydra::complex<double>`` and ``hydra::complex<double>``. 
It is not necessary to access each field stored in each entry to perform a conversion invoking the corresponding constructors. The next example shows how this can be accomplished in a more elegant way using a lambda function:

.. code-block:: cpp 
	
	#include <hydra/device/System.h>
	#include<hydra/multiarray.h>
	#include<hydra/Complex.h>

	...

	hydra::multiarray<4, double, hydra::device::sys_t> marray;
	
	for(int i=0; i<10;i++){
		marray.push_back(hydra::make_tuple( i, 2*i, i, 2*i));
	}
    
  	auto caster = [] __host__ device__ (hydra::tuple<double, double, double, double>& entry ){
  	
    	hydra::complex<double> c1(hydra::get<0>(entry), hydra::get<1>(entry));
    	hydra::complex<double> c2(hydra::get<2>(entry), hydra::get<2>(entry));
    	return hydra::make_pair(  c1, c2); 
    
    };
    
	for(auto x = marray.begin(caster); x != marray.end(caster); x++ ) 
   		std::cout << *x << std::endl;


stdout will look like:


.. code-block:: text
	
	((0, 0), (0.0, 0.0))
	((1, 2), (1.0, 2.0))
	((2, 4), (2.0, 4.0))
	...
	((9, 18), (9.0, 18.0))
