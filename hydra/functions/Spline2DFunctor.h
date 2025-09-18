/*----------------------------------------------------------------------------
 *
 *   Copyright (C) 2016 - 2025 Antonio Augusto Alves Junior
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
 * Spiline2DFunctor.h
 *
 *  Created on: 18 de set. de 2023
 *      Author: Antonio Augusto Alves Junior
 */

#ifndef SPILINE2DFUNCTOR_H_
#define SPILINE2DFUNCTOR_H_



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

template<typename IteratorX, typename IteratorY, typename IteratorZ, typename ArgType1, typename ArgType2, typename Signature=double(ArgType1, ArgType1)>
class Spline2DFunctor:
		public BaseFunctor< Spline2DFunctor<IteratorX, IteratorY, IteratorZ, ArgType1, ArgType2>, Signature, 0>
{


public:

	Spline2DFunctor() = delete;

	Spline2DFunctor( IteratorX xfirst, IteratorX xlast, IteratorY yfirst, IteratorY ylast, IteratorZ zfirst ):
		BaseFunctor< Spline2DFunctor<IteratorX, IteratorY, IteratorZ, ArgType1, ArgType2>, Signature, 0 >(),
		fSizeX(hydra::thrust::distance(xfirst, xlast)),
		fSizeY(hydra::thrust::distance(yfirst, ylast)),
		fX(xfirst),
		fY(yfirst),
		fZ(zfirst)
		{}

	__hydra_host__ __hydra_device__
	Spline2DFunctor(Spline2DFunctor< IteratorX, IteratorY, IteratorZ, ArgType1, ArgType2 > const& other ):
	BaseFunctor<Spline2DFunctor<IteratorX, IteratorY, IteratorZ, ArgType1, ArgType2>, Signature, 0>(other),
	fSizeX(other.GetSizeX()),
	fSizeY(other.GetSizeY()),
	fX(other.GetX()),
	fY(other.GetY()),
	fZ(other.GetZ())
	{ }

	__hydra_host__ __hydra_device__ inline
	Spline2DFunctor< IteratorX, IteratorY, IteratorZ, ArgType1, ArgType2 >&
	operator=(Spline2DFunctor< IteratorX, IteratorY, IteratorZ, ArgType1, ArgType2 > const& other )
	{
		if(this == &other) return *this;

		BaseFunctor<Spline2DFunctor< IteratorX, IteratorY, IteratorZ, ArgType1, ArgType2  >, Signature, 0>::operator=(other);

		fSizeX=other.GetSizeX();
		fSizeY=other.GetSizeY();
		fX=other.GetX();
		fY=other.GetY();
		fZ=other.Getz();

		return *this;
	}

	__hydra_host__ __hydra_device__
    inline size_t GetSizeX() const
	{
		return fSizeX;
	}

	__hydra_host__ __hydra_device__
	inline size_t GetSizeY() const
		{
			return fSizeY;
		}

	__hydra_host__ __hydra_device__
	inline IteratorX GetX() const
	{
		return fX;
	}

	__hydra_host__ __hydra_device__
	inline IteratorY GetY() const
	{
		return fY;
	}

	__hydra_host__ __hydra_device__
	inline IteratorZ GetZ() const
	{
		return fZ;
	}


	__hydra_host__ __hydra_device__
	inline double Evaluate(ArgType1 X, ArgType2 Y)  const {


		IteratorX fXN = fX + fSizeX;
		IteratorY fYN = fY + fSizeY;

		double r = spline2D(fX,  fXN, fY, fYN, fZ,  X, Y);

		return  CHECK_VALUE( r, " r=%f ", r) ;
	}



private:


	size_t fSizeX;
	size_t fSizeY;
	IteratorX fX;
	IteratorY fY;
	IteratorZ fZ;

};


template<typename ArgTypeX, typename ArgTypeY, typename IteratorX, typename IteratorY,  typename IteratorZ>
inline Spline2DFunctor<  IteratorX, IteratorY, IteratorZ, ArgTypeX, ArgTypeY >
make_spline2D(IteratorX firstX, IteratorX lastX, IteratorY firstY, IteratorY lastY, IteratorZ firstZ )
{

	return Spline2DFunctor<  IteratorX, IteratorY, IteratorZ, ArgTypeX, ArgTypeY  >( firstX, lastX, firstY, lastY, firstZ );
}

template<typename ArgTypeX, typename ArgTypeY, typename IterableX, typename IterableY, typename IterableZ >
inline typename std::enable_if<
          hydra::detail::is_iterable<IterableX>::value &&
		  hydra::detail::is_iterable<IterableY>::value &&
		  hydra::detail::is_iterable<IterableZ>::value,
          Spline2DFunctor< decltype(std::declval<IterableX>().begin()) ,decltype(std::declval<IterableY>().begin()),
                          decltype(std::declval<IterableZ>().begin()), ArgTypeX, ArgTypeY> >::type
make_spline2D(IterableX&& x, IterableY&& y, IterableZ&& z)
{

typedef  decltype(std::declval<IterableX>().begin()) IteratorX;
typedef  decltype(std::declval<IterableY>().begin()) IteratorY;
typedef  decltype(std::declval<IterableZ>().begin()) IteratorZ;

	return Spline2DFunctor<IteratorX, IteratorY, IteratorZ, ArgTypeX, ArgTypeY>(
			std::forward<IterableX>(x).begin(),
			std::forward<IterableX>(x).end(),
			std::forward<IterableY>(y).begin(),
			std::forward<IterableY>(y).end(),
			std::forward<IterableZ>(z).begin());
}


template<typename T, hydra::detail::Backend BACKEND>
inline typename std::enable_if< std::is_convertible<T, double>::value,
Spline2DFunctor<
decltype(std::declval<DenseHistogram<T, 2,  hydra::detail::BackendPolicy<BACKEND>, detail::multidimensional> >().GetBinsCenters(placeholders::_0).begin()),
decltype(std::declval<DenseHistogram<T, 2,  hydra::detail::BackendPolicy<BACKEND>, detail::multidimensional> >().GetBinsCenters(placeholders::_1).begin()),
decltype(std::declval<DenseHistogram<T, 2,  hydra::detail::BackendPolicy<BACKEND>, detail::multidimensional > >().GetBinsContents().begin()), double, double>>::type
make_spline( DenseHistogram<T, 2,  hydra::detail::BackendPolicy<BACKEND>, detail::multidimensional >  const& histogram )
{

typedef  DenseHistogram<T, 2,  hydra::detail::BackendPolicy<BACKEND>, detail::multidimensional > histogram_type;

typedef  decltype(std::declval<histogram_type>().GetBinsCenters().begin(placeholders::_0)) IteratorX;
typedef  decltype(std::declval<histogram_type>().GetBinsCenters().begin(placeholders::_1)) IteratorY;
typedef  decltype(std::declval<histogram_type>().GetBinsContents().begin()) IteratorM;

	return Spline2DFunctor<IteratorX, IteratorY, IteratorM, double, double>(
			histogram.GetBinsCenters(placeholders::_0).begin(), histogram.GetBinsCenters(placeholders::_0).end(),
			histogram.GetBinsCenters(placeholders::_1).begin(), histogram.GetBinsCenters(placeholders::_1).end(),
			histogram.GetBinsContents().begin());
}



template<typename T, hydra::detail::Backend BACKEND>
inline typename std::enable_if< std::is_convertible<T, double>::value,
Spline2DFunctor<
decltype(std::declval<SparseHistogram<T, 2,  hydra::detail::BackendPolicy<BACKEND>, detail::multidimensional> >().GetBinsCenters(placeholders::_0).begin()),
decltype(std::declval<SparseHistogram<T, 2,  hydra::detail::BackendPolicy<BACKEND>, detail::multidimensional> >().GetBinsCenters(placeholders::_1).begin()),
decltype(std::declval<SparseHistogram<T, 2,  hydra::detail::BackendPolicy<BACKEND>, detail::multidimensional > >().GetBinsContents().begin()), double, double>>::type
make_spline( SparseHistogram<T, 2,  hydra::detail::BackendPolicy<BACKEND>, detail::multidimensional >  const& histogram )
{

typedef  SparseHistogram<T, 2,  hydra::detail::BackendPolicy<BACKEND>, detail::multidimensional > histogram_type;

typedef  decltype(std::declval<histogram_type>().GetBinsCenters().begin(placeholders::_0)) IteratorX;
typedef  decltype(std::declval<histogram_type>().GetBinsCenters().begin(placeholders::_1)) IteratorY;
typedef  decltype(std::declval<histogram_type>().GetBinsContents().begin()) IteratorM;

	return Spline2DFunctor<IteratorX, IteratorY, IteratorM, double, double>(
			histogram.GetBinsCenters(placeholders::_0).begin(),histogram.GetBinsCenters(placeholders::_0).end(),
			histogram.GetBinsCenters(placeholders::_1).begin(),histogram.GetBinsCenters(placeholders::_1).end(),
			histogram.GetBinsContents().begin());
}

}  // namespace hydra


#endif /* SSPILINE2DFUNCTOR_H_ */
