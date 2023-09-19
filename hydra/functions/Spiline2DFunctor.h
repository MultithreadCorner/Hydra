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
#include <hydra/Spiline.h>
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
class Spiline2DFunctor:
		public BaseFunctor< Spiline2DFunctor<IteratorX, IteratorY, IteratorZ, ArgType1, ArgType2>, Signature, 0>
{


public:

	Spiline2DFunctor() = delete;

	Spiline2DFunctor( IteratorX xfirst, IteratorX xlast, IteratorY yfirst, IteratorZ zfirst ):
		BaseFunctor< Spiline2DFunctor<IteratorX, IteratorY, IteratorZ, ArgType1, ArgType2>, Signature, 0 >(),
		fSize(hydra::thrust::distance(xfirst, xlast)),
		fX(xfirst),
		fY(yfirst),
		fZ(zfirst)
		{}

	__hydra_host__ __hydra_device__
	Spiline2DFunctor(Spiline2DFunctor< IteratorX, IteratorY, IteratorZ, ArgType1, ArgType2 > const& other ):
	BaseFunctor<Spiline2DFunctor<IteratorX, IteratorY, IteratorZ, ArgType1, ArgType2>, Signature, 0>(other),
	fSize(other.GetSize()),
	fX(other.GetX()),
	fY(other.GetY()),
	fZ(other.GetZ())
	{ }

	__hydra_host__ __hydra_device__ inline
	Spiline2DFunctor< IteratorX, IteratorY, IteratorZ, ArgType1, ArgType2 >&
	operator=(Spiline2DFunctor< IteratorX, IteratorY, IteratorZ, ArgType1, ArgType2 > const& other )
	{
		if(this == &other) return *this;

		BaseFunctor<Spiline2DFunctor< IteratorX, IteratorY, IteratorZ, ArgType1, ArgType2  >, Signature, 0>::operator=(other);

		fSize=other.GetSize();
		fX=other.GetX();
		fY=other.GetY();
		fZ=other.Getz();

		return *this;
	}

	__hydra_host__ __hydra_device__
    inline size_t GetSize() const
	{
		return fSize;
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


		IteratorX fXN = fX + fSize;

		double r = spiline2D(fX,  fXN, fY, fZ,  X, Y);

		return  CHECK_VALUE( r, " r=%f ", r) ;
	}



private:


	size_t fSize;
	IteratorX fX;
	IteratorY fY;
	IteratorZ fZ;

};


template<typename ArgTypeX, typename ArgTypeY, typename IteratorX, typename IteratorY,  typename IteratorZ>
inline Spiline2DFunctor<  IteratorX, IteratorY, IteratorZ, ArgTypeX, ArgTypeY >
make_spiline2D(IteratorX firstX, IteratorX lastX, IteratorY firstY,  IteratorZ firstZ )
{

	return Spiline2DFunctor<  IteratorX, IteratorY, IteratorZ, ArgTypeX, ArgTypeY  >( firstX, lastX, firstY, firstZ );
}

template<typename ArgTypeX, typename ArgTypeY, typename IterableX, typename IterableY, typename IterableZ >
inline typename std::enable_if<
          hydra::detail::is_iterable<IterableX>::value &&
		  hydra::detail::is_iterable<IterableY>::value &&
		  hydra::detail::is_iterable<IterableZ>::value,
          Spiline2DFunctor< decltype(std::declval<IterableX>().begin()) ,decltype(std::declval<IterableY>().begin()),
                          decltype(std::declval<IterableZ>().begin()), ArgTypeX, ArgTypeY> >::type
make_spiline2D(IterableX&& x, IterableY&& y, IterableZ&& z)
{

typedef  decltype(std::declval<IterableX>().begin()) IteratorX;
typedef  decltype(std::declval<IterableY>().begin()) IteratorY;
typedef  decltype(std::declval<IterableZ>().begin()) IteratorZ;

	return Spiline2DFunctor<IteratorX, IteratorY, IteratorZ, ArgTypeX, ArgTypeY>(
			std::forward<IterableX>(x).begin(),
			std::forward<IterableX>(x).end(),
			std::forward<IterableY>(y).begin(),
			std::forward<IterableZ>(z).begin());
}

/*
template<typename T, hydra::detail::Backend BACKEND>
inline SpilineFunctor<
decltype( std::declval<	DenseHistogram<T, 1,  hydra::detail::BackendPolicy<BACKEND>, detail::unidimensional > >().GetBinsCenters().begin()),
decltype(std::declval< DenseHistogram<T, 1,  hydra::detail::BackendPolicy<BACKEND>, detail::unidimensional > >().GetBinsContents().begin()), T>
make_spiline( DenseHistogram<T, 1,  hydra::detail::BackendPolicy<BACKEND>, detail::unidimensional >  const& histogram )
{

typedef  decltype(std::declval<DenseHistogram<T, 1,  hydra::detail::BackendPolicy<BACKEND>, detail::unidimensional > >().GetBinsCenters().begin()) Iterator1;
typedef  decltype(std::declval<DenseHistogram<T, 1,  hydra::detail::BackendPolicy<BACKEND>, detail::unidimensional > >().GetBinsContents().begin()) Iterator2;

	return SpilineFunctor<Iterator1, Iterator2, T>(
			histogram.GetBinsCenters().begin(),
			histogram.GetBinsCenters().end(),
			histogram.GetBinsContents().begin());
}


template<typename T, hydra::detail::Backend BACKEND >
inline SpilineFunctor< decltype(std::declval<SparseHistogram<T, 1,  hydra::detail::BackendPolicy<BACKEND>, detail::unidimensional > >().GetBinsCenters().begin()),
                       decltype(std::declval<SparseHistogram<T, 1,  hydra::detail::BackendPolicy<BACKEND>, detail::unidimensional > >().GetBinsContents().begin()), T>
make_spiline( SparseHistogram<T, 1,  hydra::detail::BackendPolicy<BACKEND>, detail::unidimensional > const& histogram )
{

typedef  decltype(std::declval< SparseHistogram<T, 1,  hydra::detail::BackendPolicy<BACKEND>, detail::unidimensional >>().GetBinsCenters().begin()) Iterator1;
typedef  decltype(std::declval< SparseHistogram<T, 1,  hydra::detail::BackendPolicy<BACKEND>, detail::unidimensional >>().GetBinsContents().begin()) Iterator2;

	return SpilineFunctor<Iterator1, Iterator2, T>(
			histogram.GetBinsCenters().begin(),
			histogram.GetBinsCenters().end(),
			histogram.GetBinsContents().begin());
}
*/

}  // namespace hydra


#endif /* SSPILINE2DFUNCTOR_H_ */
