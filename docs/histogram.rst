Histograms
==========

Hydra implements two classes dedicated to calculate multidimensional histograms in parallel.
One class for dense histograms and other for sparse histograms. These classes provides only the basic functionality to calculate the histogram using one of the supported parallel back-ends. 
Once calculated, the histogram contents can be exported to external libraries, like ROOT, for
drawing etc. 

The histograms classes does not process event-by-event. They takes iterators pointing to containers storing the data and process it at once. This approach is orders of magnitude more efficient than iterate over the container and histogram in entry-by-entry basis. 

If a explicit policy is not specified, the histogram is processed in the back-end the data is stored. 

Binning convention
------------------

In Hydra, a histogram with N bins is stored in a array with length N+2. In range contents are indexed starting from 0 to N-1. Underflow contents are stored in bin N and overflow contents are stored in bin N+1.  


Global and dimensional binning
------------------------------

The histogram contents is organized in a linear array of length N+2, where N is total number of bins, obtained multiplying the number of bins configured in  for each dimension.
The conversion between global bin number and dimensional bin numbers is performed 
by the methods ``GetBin(...)`` and ``GetIndexes(...)``, implemented in both classes.
The internal indexing convention used in Hydra in general does not match the one used in other 
libraries and interfaces. Users are advised to always export the histogram contents using the 
bin numbers per bin.   


Dense histograms
----------------

Dense histograms store all bins, including ones with zero content. In Hydra, they are represented by the class ``hydra::DenseHistogram<<NDimensions, Type>``, where ``NDimensions`` is the number of dimensions and ``Type`` is the type  of the histogram's  values.

.. code-block:: cpp
	
	#include <hydra/device/System.h>
	#include <hydra/multiarray.h>
	#include <hydra/DenseHistogram.h>
	#include <array>

	...

	hydra::multiarray<4, double, hydra::device::sys_t> mvector;
	
	...
	// fill mvector with the data of interest... 
	...

	//histogram ranges
	std::array<double, 4>max{ 1.0,  2.0,  3.0,  4.0};
	std::array<double, 4>min{-1.0, -2.0, -3.0, -4.0};
	
	//bins per dimension
	std::array<size_t, 3> nbins{10, 20, 30, 40};
	
	//create histogram
	hydra::DenseHistogram<3, double> Histogram(nbins, min, max);

	Histogram.Fill( mvector.begin(), mvector.end());

	//getting bin content [0, 2, 3, 1]
	Histogram.GetBinContent({0, 2, 3, 1});


Sparse histograms 
-----------------

Sparse histograms store only bins with non-zero content. In Hydra, they are represented by the class ``hydra::SparseHistogram<NDimensions, Type>``, where ``NDimensions`` is the number of dimensions and ``Type`` is the type  of the histogram's  values.

.. code-block:: cpp

	#include <hydra/device/System.h>
	#include <hydra/multiarray.h>
	#include <hydra/SparseHistogram.h>
	#include <array>

	...

	hydra::multiarray<4, double, hydra::device::sys_t> mvector;
	
	...
	// fill mvector with the data of interest... 
	...

	//histogram ranges
	std::array<double, 4>max{ 1.0,  2.0,  3.0,  4.0};
	std::array<double, 4>min{-1.0, -2.0, -3.0, -4.0};
	
	//bins per dimension
	std::array<size_t, 3> nbins{10, 20, 30, 40};
	
	//create histogram
	hydra::SparseHistogram<3, double> Histogram(nbins, min, max);

	Histogram.Fill( mvector.begin(), mvector.end());

	//getting bin content [0, 2, 3, 1]
	Histogram.GetBinContent({0, 2, 3, 1});




