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
 * Spline4DFunctor.h
 *
 *  Created on: 18 de set. de 2023
 *      Author: Antonio Augusto Alves Junior
 */

#ifndef SPLINE3DFUNCTOR_H_
#define SPLINE3DFUNCTOR_H_



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

template<typename IteratorX, typename IteratorY, typename IteratorW, typename IteratorZ, typename IteratorM,
         typename ArgType1, typename ArgType2,  typename ArgType3, typename ArgType4,
		 typename Signature=double(ArgType1, ArgType2, ArgType3, ArgType4)>
class Spline4DFunctor:
		public BaseFunctor< Spline4DFunctor<IteratorX, IteratorY, IteratorW, IteratorZ, IteratorM, ArgType1, ArgType2, ArgType3, ArgType4>, Signature, 0>
{


public:

	Spline4DFunctor() = delete;

	Spline4DFunctor( IteratorX xfirst, IteratorX xlast,
			IteratorY yfirst, IteratorY ylast,
			IteratorW wfirst, IteratorW wlast,
			IteratorZ zfirst, IteratorZ zlast,
			IteratorM mfirst  ):
		BaseFunctor< Spline4DFunctor<IteratorX, IteratorY, IteratorW, IteratorZ, IteratorM, ArgType1, ArgType2, ArgType3, ArgType4>, Signature, 0 >(),
		fSizeX(hydra::thrust::distance(xfirst, xlast)),
		fSizeY(hydra::thrust::distance(yfirst, ylast)),
		fSizeW(hydra::thrust::distance(wfirst, wlast)),
		fSizeZ(hydra::thrust::distance(zfirst, zlast)),
		fX(xfirst),
		fY(yfirst),
		fW(wfirst),
		fZ(zfirst),
		fM(mfirst)
		{}

	__hydra_host__ __hydra_device__
	Spline4DFunctor(Spline4DFunctor< IteratorX, IteratorY, IteratorW, IteratorZ, IteratorM, ArgType1, ArgType2, ArgType3, ArgType4 > const& other ):
	BaseFunctor<Spline4DFunctor<IteratorX, IteratorY, IteratorW, IteratorZ, IteratorM, ArgType1, ArgType2, ArgType3, ArgType4>, Signature, 0>(other),
	fSizeX(other.GetSizeX()),
	fSizeY(other.GetSizeY()),
	fSizeW(other.GetSizeW()),
	fSizeZ(other.GetSizeZ()),
	fX(other.GetX()),
	fY(other.GetY()),
	fW(other.GetW()),
	fZ(other.GetZ()),
	fM(other.GetM())
	{ }

	__hydra_host__ __hydra_device__ inline
	Spline4DFunctor< IteratorX, IteratorY, IteratorW, IteratorZ,  IteratorM, ArgType1, ArgType2, ArgType3, ArgType4 >&
	operator=(Spline4DFunctor< IteratorX, IteratorY, IteratorW, IteratorZ, IteratorM, ArgType1, ArgType2, ArgType3, ArgType4 > const& other )
	{
		if(this == &other) return *this;

		BaseFunctor<Spline4DFunctor< IteratorX, IteratorY, IteratorW, IteratorZ,  IteratorM, ArgType1, ArgType2, ArgType3, ArgType4 >, Signature, 0>::operator=(other);

		fSizeX=other.GetSizeX();
		fSizeY=other.GetSizeY();
		fSizeW=other.GetSizeW();
		fSizeZ=other.GetSizeZ();
		fX=other.GetX();
		fY=other.GetY();
		fW=other.GetW();
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
	inline size_t GetSizeW() const
	{
		return fSizeW;
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
	inline IteratorY GetW() const
	{
		return fW;
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
	inline double Evaluate(ArgType1 X, ArgType2 Y, ArgType3 W, ArgType4 Z)  const {


		IteratorX fXN = fX + fSizeX;
		IteratorY fYN = fY + fSizeY;
		IteratorY fWN = fW + fSizeW;
		IteratorZ fZN = fZ + fSizeZ;

		double r = spline4D(fX,  fXN, fY, fYN,  fW, fWN, fZ, fZN, fM, X, Y, W, Z);

		return  CHECK_VALUE( r, " r=%f ", r) ;
	}



private:


	size_t fSizeX;
	size_t fSizeY;
	size_t fSizeW;
	size_t fSizeZ;
	IteratorX fX;
	IteratorY fY;
	IteratorY fW;
	IteratorZ fZ;
	IteratorM fM;

};


template<typename ArgTypeX, typename ArgTypeY, typename ArgTypeW, typename ArgTypeZ,
         typename IteratorX, typename IteratorY, typename IteratorW, typename IteratorZ,
		 typename IteratorM>
inline Spline4DFunctor<  IteratorX, IteratorY, IteratorW, IteratorZ, IteratorM, ArgTypeX, ArgTypeY, ArgTypeW, ArgTypeZ  >
make_spline4D(IteratorX firstX, IteratorX lastX,
		      IteratorY firstY, IteratorY lastY,
			  IteratorW firstW, IteratorW lastW,
			  IteratorZ firstZ, IteratorZ lastZ,
			  IteratorM measurements_first )
{

	return Spline4DFunctor<  IteratorX, IteratorY, IteratorW, IteratorZ,
			                 IteratorM, ArgTypeX, ArgTypeY, ArgTypeW, ArgTypeZ >( firstX, lastX,
			                		 firstY, lastY,
									 firstW, lastW,
									 firstZ, lastZ,
									 measurements_first );
}

template<typename ArgTypeX, typename ArgTypeY, typename ArgTypeW, typename ArgTypeZ,
         typename IterableX, typename IterableY, typename IterableW, typename IterableZ,
		 typename IterableM >
inline typename std::enable_if<
          hydra::detail::is_iterable<IterableX>::value &&
		  hydra::detail::is_iterable<IterableY>::value &&
		  hydra::detail::is_iterable<IterableW>::value &&
		  hydra::detail::is_iterable<IterableZ>::value &&
		  hydra::detail::is_iterable<IterableM>::value,
          Spline4DFunctor<
		   decltype(std::declval<IterableX>().begin()),
		   decltype(std::declval<IterableY>().begin()),
		   decltype(std::declval<IterableW>().begin()),
           decltype(std::declval<IterableZ>().begin()),
		   decltype(std::declval<IterableM>().begin()),
		   ArgTypeX, ArgTypeY, ArgTypeW, ArgTypeZ> >::type
make_spline4D(IterableX&& x, IterableY&& y, IterableW&& w, IterableZ&& z, IterableM&& measurements)
{

typedef  decltype(std::declval<IterableX>().begin()) IteratorX;
typedef  decltype(std::declval<IterableY>().begin()) IteratorY;
typedef  decltype(std::declval<IterableW>().begin()) IteratorW;
typedef  decltype(std::declval<IterableZ>().begin()) IteratorZ;
typedef  decltype(std::declval<IterableM>().begin()) IteratorM;

	return Spline4DFunctor<IteratorX, IteratorY, IteratorW, IteratorZ, IteratorM, ArgTypeX, ArgTypeY, ArgTypeW, ArgTypeZ>(
			std::forward<IterableX>(x).begin(),	std::forward<IterableX>(x).end(),
			std::forward<IterableY>(y).begin(),	std::forward<IterableY>(y).end(),
			std::forward<IterableW>(w).begin(),	std::forward<IterableW>(w).end(),
			std::forward<IterableZ>(z).begin(),	std::forward<IterableZ>(z).end(),
			std::forward<IterableM>(measurements).begin()
	);
}


template<typename T, hydra::detail::Backend BACKEND>
inline typename std::enable_if< std::is_convertible<T, double>::value,
Spline4DFunctor<
decltype(std::declval<DenseHistogram<T, 4,  hydra::detail::BackendPolicy<BACKEND>, detail::multidimensional> >().GetBinsCenters(placeholders::_0).begin()),
decltype(std::declval<DenseHistogram<T, 4,  hydra::detail::BackendPolicy<BACKEND>, detail::multidimensional> >().GetBinsCenters(placeholders::_1).begin()),
decltype(std::declval<DenseHistogram<T, 4,  hydra::detail::BackendPolicy<BACKEND>, detail::multidimensional> >().GetBinsCenters(placeholders::_2).begin()),
decltype(std::declval<DenseHistogram<T, 4,  hydra::detail::BackendPolicy<BACKEND>, detail::multidimensional> >().GetBinsCenters(placeholders::_3).begin()),
decltype(std::declval<DenseHistogram<T, 4,  hydra::detail::BackendPolicy<BACKEND>, detail::multidimensional > >().GetBinsContents().begin()),
double, double, double, double>>::type
make_spline( DenseHistogram<T, 4,  hydra::detail::BackendPolicy<BACKEND>, detail::multidimensional >  const& histogram ){

typedef  DenseHistogram<T, 4,  hydra::detail::BackendPolicy<BACKEND>, detail::multidimensional > histogram_type;

typedef  decltype(std::declval<histogram_type>().GetBinsCenters().begin(placeholders::_0)) IteratorX;
typedef  decltype(std::declval<histogram_type>().GetBinsCenters().begin(placeholders::_1)) IteratorY;
typedef  decltype(std::declval<histogram_type>().GetBinsCenters().begin(placeholders::_2)) IteratorW;
typedef  decltype(std::declval<histogram_type>().GetBinsCenters().begin(placeholders::_3)) IteratorZ;
typedef  decltype(std::declval<histogram_type>().GetBinsContents().begin()) IteratorM;

	return Spline4DFunctor<IteratorX, IteratorY, IteratorW, IteratorZ, IteratorM, double, double, double, double>(
			histogram.GetBinsCenters(placeholders::_0).begin(),histogram.GetBinsCenters(placeholders::_0).end(),
			histogram.GetBinsCenters(placeholders::_1).begin(),histogram.GetBinsCenters(placeholders::_1).end(),
			histogram.GetBinsCenters(placeholders::_2).begin(),histogram.GetBinsCenters(placeholders::_2).end(),
			histogram.GetBinsCenters(placeholders::_3).begin(),histogram.GetBinsCenters(placeholders::_3).end(),
			histogram.GetBinsContents().begin());
}



template<typename T, hydra::detail::Backend BACKEND>
inline typename std::enable_if< std::is_convertible<T, double>::value,
Spline4DFunctor<
decltype(std::declval<SparseHistogram<T, 4,  hydra::detail::BackendPolicy<BACKEND>, detail::multidimensional> >().GetBinsCenters(placeholders::_0).begin()),
decltype(std::declval<SparseHistogram<T, 4,  hydra::detail::BackendPolicy<BACKEND>, detail::multidimensional> >().GetBinsCenters(placeholders::_1).begin()),
decltype(std::declval<SparseHistogram<T, 4,  hydra::detail::BackendPolicy<BACKEND>, detail::multidimensional> >().GetBinsCenters(placeholders::_2).begin()),
decltype(std::declval<SparseHistogram<T, 4,  hydra::detail::BackendPolicy<BACKEND>, detail::multidimensional> >().GetBinsCenters(placeholders::_3).begin()),
decltype(std::declval<SparseHistogram<T, 4,  hydra::detail::BackendPolicy<BACKEND>, detail::multidimensional > >().GetBinsContents().begin()),
double, double, double, double>>::type
make_spline( SparseHistogram<T, 4,  hydra::detail::BackendPolicy<BACKEND>, detail::multidimensional >  const& histogram ){

typedef  SparseHistogram<T, 4,  hydra::detail::BackendPolicy<BACKEND>, detail::multidimensional > histogram_type;

typedef  decltype(std::declval<histogram_type>().GetBinsCenters().begin(placeholders::_0)) IteratorX;
typedef  decltype(std::declval<histogram_type>().GetBinsCenters().begin(placeholders::_1)) IteratorY;
typedef  decltype(std::declval<histogram_type>().GetBinsCenters().begin(placeholders::_2)) IteratorW;
typedef  decltype(std::declval<histogram_type>().GetBinsCenters().begin(placeholders::_3)) IteratorZ;
typedef  decltype(std::declval<histogram_type>().GetBinsContents().begin()) IteratorM;

	return Spline4DFunctor<IteratorX, IteratorY, IteratorW, IteratorZ, IteratorM, double, double, double, double>(
			histogram.GetBinsCenters(placeholders::_0).begin(),histogram.GetBinsCenters(placeholders::_0).end(),
			histogram.GetBinsCenters(placeholders::_1).begin(),histogram.GetBinsCenters(placeholders::_1).end(),
			histogram.GetBinsCenters(placeholders::_2).begin(),histogram.GetBinsCenters(placeholders::_2).end(),
			histogram.GetBinsCenters(placeholders::_3).begin(),histogram.GetBinsCenters(placeholders::_3).end(),
			histogram.GetBinsContents().begin());
}


}  // namespace hydra


#endif /* SSPILINE2DFUNCTOR_H_ */
