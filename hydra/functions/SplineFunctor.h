/*----------------------------------------------------------------------------
 *
 *   Copyright (C) 2016 - 2023 Antonio Augusto Alves Junior
 *
 *   This file is part of Hydra Data Analysis Framework.
 *
 *   Hydra is free software: you can redistribute it and/or modify
 *   it under the terms of the GNU General Public License as published by
 *   the Free Software Foundation, either version 3 of the License, or
 *   (at your option) any later version.
 *
 *   Hydra is distributed in the hope that it will be useful,
 *   but WITHOUT ANY WARRANTY; without even the implied warranty of
 *   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *   GNU General Public License for more details.
 *
 *   You should have received a copy of the GNU General Public License
 *   along with Hydra.  If not, see <http://www.gnu.org/licenses/>.
 *
 *---------------------------------------------------------------------------*/

/*
 * SpilineFunctor.h
 *
 *  Created on: 22/12/2018
 *      Author: Antonio Augusto Alves Junior
 */

#ifndef SPILINEFUNCTOR_H_
#define SPILINEFUNCTOR_H_



#include <hydra/detail/Config.h>
#include <hydra/detail/BackendPolicy.h>
#include <hydra/Types.h>
#include <hydra/Function.h>
#include <hydra/Spline.h>
#include <hydra/DenseHistogram.h>
#include <hydra/SparseHistogram.h>
#include <hydra/Placeholders.h>
#include <hydra/detail/utility/CheckValue.h>
#include <hydra/detail/external/hydra_thrust/copy.h>
#include <hydra/detail/external/hydra_thrust/iterator/zip_iterator.h>
#include <hydra/detail/external/hydra_thrust/execution_policy.h>
#include <hydra/detail/external/hydra_thrust/binary_search.h>
#include <hydra/detail/external/hydra_thrust/extrema.h>
#include <math.h>
#include <algorithm>
#include <memory>

namespace hydra {

/**
 * \class SpilineFunctor
 *
 * A simple method for a one—dimensional
interpolation on a given set of data points (xi, yi). In each interval
(xi,x,x_{i+1}) the interpolation function is assumed to be a third-
order polynomial passing through the data points. The slope at
each grid point is determined in such a way as to guarantee
a monotonic behavior of the interpolating function. The result
is a smooth curve with continuous ﬁrst-order derivatives that
passes through any given set of data points without spurious
oscillations. Local extrema can occur only at grid points where
they are given by the data, but not in between two adjacent
grid points. The method gives exact results if the data points
correspond to a second-order polynomial.

Reference: M. Steffen, Astron. Astrophys. 239, 443—450 (1990).
*/
template<typename Iterator1, typename Iterator2,typename ArgType, typename Signature=double(ArgType)>
class SplineFunctor: public BaseFunctor<SplineFunctor<Iterator1, Iterator2, ArgType>, Signature, 0>
{


public:

	SplineFunctor() = delete;

	SplineFunctor( Iterator1 first, Iterator1 last, Iterator2 output ):
		BaseFunctor<SplineFunctor<Iterator1, Iterator2, ArgType>, Signature, 0>(),
		fSize(hydra::thrust::distance(first, last)),
		fX(first),
		fY(output)
		{}

	__hydra_host__ __hydra_device__
	SplineFunctor(SplineFunctor<Iterator1, Iterator2, ArgType> const& other ):
	BaseFunctor<SplineFunctor<Iterator1, Iterator2, ArgType>, Signature, 0>(other),
	fSize(other.GetSize()),
	fX(other.GetX()),
	fY(other.GetY())
	{ }

	__hydra_host__ __hydra_device__ inline
	SplineFunctor<Iterator1, Iterator2, ArgType>&
	operator=(SplineFunctor<Iterator1, Iterator2, ArgType> const& other )
	{
		if(this == &other) return *this;

		BaseFunctor<SplineFunctor<Iterator1, Iterator2,  ArgType>, Signature, 0>::operator=(other);

		fSize=other.GetSize();
		fX=other.GetX();
		fY=other.GetY();

		return *this;
	}

	__hydra_host__ __hydra_device__
	size_t GetSize() const
	{
		return fSize;
	}

	__hydra_host__ __hydra_device__
	Iterator1 GetX() const
	{
		return fX;
	}

	__hydra_host__ __hydra_device__
	Iterator2 GetY() const
	{
		return fY;
	}


	__hydra_host__ __hydra_device__
	inline double Evaluate(ArgType X)  const {


		Iterator1 fXN = fX + fSize;

		double r = spline(fX,  fXN, fY,  X);

		return  CHECK_VALUE( r, " r=%f ", r) ;
	}



private:


	size_t fSize;
	Iterator1 fX;
	Iterator2 fY;
};


template<typename ArgType, typename Iterator1, typename Iterator2>
inline SplineFunctor<Iterator1, Iterator2, ArgType>
make_spline(Iterator1 firstX, Iterator1 lastX, Iterator2 firstY )
{

	return SplineFunctor<Iterator1, Iterator2, ArgType>( firstX, lastX, firstY);
}

template<typename ArgType, typename Iterable1, typename Iterable2 >
inline typename std::enable_if<
          hydra::detail::is_iterable<Iterable1>::value &&
          hydra::detail::is_iterable<Iterable2>::value,
          SplineFunctor< decltype(std::declval<Iterable1>().begin()),
                          decltype(std::declval<Iterable2>().begin()), ArgType> >::type
make_spline(Iterable1&& x, Iterable2&& y)
{

typedef  decltype(std::declval<Iterable1>().begin()) Iterator1;
typedef  decltype(std::declval<Iterable2>().begin()) Iterator2;

	return SplineFunctor<Iterator1, Iterator2, ArgType>(
			std::forward<Iterable1>(x).begin(),
			std::forward<Iterable1>(x).end(),
			std::forward<Iterable2>(y).begin());
}


template<typename T, hydra::detail::Backend BACKEND>
inline SplineFunctor<
decltype( std::declval<	DenseHistogram<T, 1,  hydra::detail::BackendPolicy<BACKEND>, detail::unidimensional > >().GetBinsCenters().begin()),
decltype(std::declval< DenseHistogram<T, 1,  hydra::detail::BackendPolicy<BACKEND>, detail::unidimensional > >().GetBinsContents().begin()), T>
make_spline( DenseHistogram<T, 1,  hydra::detail::BackendPolicy<BACKEND>, detail::unidimensional >  const& histogram )
{

typedef  decltype(std::declval<DenseHistogram<T, 1,  hydra::detail::BackendPolicy<BACKEND>, detail::unidimensional > >().GetBinsCenters().begin()) Iterator1;
typedef  decltype(std::declval<DenseHistogram<T, 1,  hydra::detail::BackendPolicy<BACKEND>, detail::unidimensional > >().GetBinsContents().begin()) Iterator2;

	return SplineFunctor<Iterator1, Iterator2, T>(
			histogram.GetBinsCenters().begin(),
			histogram.GetBinsCenters().end(),
			histogram.GetBinsContents().begin());
}


template<typename T, hydra::detail::Backend BACKEND >
inline SplineFunctor< decltype(std::declval<SparseHistogram<T, 1,  hydra::detail::BackendPolicy<BACKEND>, detail::unidimensional > >().GetBinsCenters().begin()),
                       decltype(std::declval<SparseHistogram<T, 1,  hydra::detail::BackendPolicy<BACKEND>, detail::unidimensional > >().GetBinsContents().begin()), T>
make_spline( SparseHistogram<T, 1,  hydra::detail::BackendPolicy<BACKEND>, detail::unidimensional > const& histogram )
{

typedef  decltype(std::declval< SparseHistogram<T, 1,  hydra::detail::BackendPolicy<BACKEND>, detail::unidimensional >>().GetBinsCenters().begin()) Iterator1;
typedef  decltype(std::declval< SparseHistogram<T, 1,  hydra::detail::BackendPolicy<BACKEND>, detail::unidimensional >>().GetBinsContents().begin()) Iterator2;

	return SplineFunctor<Iterator1, Iterator2, T>(
			histogram.GetBinsCenters().begin(),
			histogram.GetBinsCenters().end(),
			histogram.GetBinsContents().begin());
}


}  // namespace hydra



#endif /* SPILINEFUNCTOR_H_ */
