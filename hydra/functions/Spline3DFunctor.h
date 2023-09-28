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
 * Spiline3DFunctor.h
 *
 *  Created on: 18 de set. de 2023
 *      Author: Antonio Augusto Alves Junior
 */

#ifndef SPILINE3DFUNCTOR_H_
#define SPILINE3DFUNCTOR_H_



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

template<typename IteratorX, typename IteratorY, typename IteratorZ,  typename IteratorM, typename ArgType1, typename ArgType2, typename ArgType3, typename Signature=double(ArgType1, ArgType2, ArgType3)>
class Spline3DFunctor:
		public BaseFunctor< Spline3DFunctor<IteratorX, IteratorY, IteratorZ, IteratorM, ArgType1, ArgType2, ArgType3>, Signature, 0>
{


public:

	Spline3DFunctor() = delete;

	Spline3DFunctor( IteratorX xfirst, IteratorX xlast, IteratorY yfirst, IteratorY ylast, IteratorZ zfirst, IteratorZ zlast, IteratorM mfirst  ):
		BaseFunctor< Spline3DFunctor<IteratorX, IteratorY, IteratorZ, IteratorM, ArgType1, ArgType2, ArgType3>, Signature, 0 >(),
		fSizeX(hydra::thrust::distance(xfirst, xlast)),
		fSizeY(hydra::thrust::distance(yfirst, ylast)),
		fSizeZ(hydra::thrust::distance(zfirst, zlast)),
		fX(xfirst),
		fY(yfirst),
		fZ(zfirst),
		fM(mfirst)
		{}

	__hydra_host__ __hydra_device__
	Spline3DFunctor(Spline3DFunctor< IteratorX, IteratorY, IteratorZ, IteratorM, ArgType1, ArgType2, ArgType3 > const& other ):
	BaseFunctor<Spline3DFunctor<IteratorX, IteratorY, IteratorZ, IteratorM, ArgType1, ArgType2, ArgType3>, Signature, 0>(other),
	fSizeX(other.GetSizeX()),
	fSizeY(other.GetSizeY()),
	fSizeZ(other.GetSizeZ()),
	fX(other.GetX()),
	fY(other.GetY()),
	fZ(other.GetZ()),
	fM(other.GetM())
	{ }

	__hydra_host__ __hydra_device__ inline
	Spline3DFunctor< IteratorX, IteratorY, IteratorZ,  IteratorM, ArgType1, ArgType2, ArgType3 >&
	operator=(Spline3DFunctor< IteratorX, IteratorY, IteratorZ,  IteratorM, ArgType1, ArgType2, ArgType3 > const& other )
	{
		if(this == &other) return *this;

		BaseFunctor<Spline3DFunctor< IteratorX, IteratorY, IteratorZ,  IteratorM, ArgType1, ArgType2, ArgType3 >, Signature, 0>::operator=(other);

		fSizeX=other.GetSizeX();
		fSizeY=other.GetSizeY();
		fSizeZ=other.GetSizeZ();
		fX=other.GetX();
		fY=other.GetY();
		fZ=other.GetZ();
		fM=other.GetM();

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
		inline size_t GetSizeZ() const
			{
				return fSizeZ;
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
	inline IteratorM GetM() const
	{
		return fM;
	}


	__hydra_host__ __hydra_device__
	inline double Evaluate(ArgType1 X, ArgType2 Y, ArgType3 Z)  const {


		IteratorX fXN = fX + fSizeX;
		IteratorY fYN = fY + fSizeY;
		IteratorZ fZN = fZ + fSizeZ;

		double r = spline3D(fX,  fXN, fY, fYN, fZ, fZN, fM, X, Y, Z);

		return  CHECK_VALUE( r, " r=%f ", r) ;
	}



private:


	size_t fSizeX;
	size_t fSizeY;
	size_t fSizeZ;
	IteratorX fX;
	IteratorY fY;
	IteratorZ fZ;
	IteratorM fM;

};


template<typename ArgTypeX, typename ArgTypeY, typename ArgTypeZ, typename IteratorX, typename IteratorY, typename IteratorZ, typename IteratorM>
inline Spline3DFunctor<  IteratorX, IteratorY, IteratorZ, IteratorM, ArgTypeX, ArgTypeY, ArgTypeZ  >
make_spline3D(IteratorX firstX, IteratorX lastX, IteratorY firstY, IteratorY lastY, IteratorZ firstZ, IteratorZ lastZ,  IteratorM measurements_first )
{

	return Spline3DFunctor<  IteratorX, IteratorY, IteratorZ, IteratorM, ArgTypeX, ArgTypeY, ArgTypeZ >( firstX, lastX, firstY, lastY, firstZ, lastZ,  measurements_first );
}

template<typename ArgTypeX, typename ArgTypeY,typename ArgTypeZ, typename IterableX, typename IterableY, typename IterableZ, typename IterableM >
inline typename std::enable_if<
          hydra::detail::is_iterable<IterableX>::value &&
		  hydra::detail::is_iterable<IterableY>::value &&
		  hydra::detail::is_iterable<IterableZ>::value &&
		  hydra::detail::is_iterable<IterableM>::value,
          Spline3DFunctor< decltype(std::declval<IterableX>().begin()) ,decltype(std::declval<IterableY>().begin()),
                          decltype(std::declval<IterableZ>().begin()), decltype(std::declval<IterableM>().begin()), ArgTypeX, ArgTypeY, ArgTypeZ> >::type
make_spline3D(IterableX&& x, IterableY&& y, IterableZ&& z, IterableM&& measurements)
{

typedef  decltype(std::declval<IterableX>().begin()) IteratorX;
typedef  decltype(std::declval<IterableY>().begin()) IteratorY;
typedef  decltype(std::declval<IterableZ>().begin()) IteratorZ;
typedef  decltype(std::declval<IterableM>().begin()) IteratorM;

	return Spline3DFunctor<IteratorX, IteratorY, IteratorZ, IteratorM, ArgTypeX, ArgTypeY, ArgTypeZ>(
			std::forward<IterableX>(x).begin(),
			std::forward<IterableX>(x).end(),
			std::forward<IterableY>(y).begin(),
			std::forward<IterableY>(y).end(),
			std::forward<IterableZ>(z).begin(),
			std::forward<IterableZ>(z).end(),
			std::forward<IterableM>(measurements).begin()
	);
}

/*
template<typename T, hydra::detail::Backend BACKEND>
inline SpilineFunctor<
decltype( std::declval<	DenseHistogram<T, 1,  hydra::detail::BackendPolicy<BACKEND>, detail::unidimensional > >().GetBinsCenters().begin()),
decltype(std::declval< DenseHistogram<T, 1,  hydra::detail::BackendPolicy<BACKEND>, detail::unidimensional > >().GetBinsContents().begin()), T>
make_spline( DenseHistogram<T, 1,  hydra::detail::BackendPolicy<BACKEND>, detail::unidimensional >  const& histogram )
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
make_spline( SparseHistogram<T, 1,  hydra::detail::BackendPolicy<BACKEND>, detail::unidimensional > const& histogram )
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
