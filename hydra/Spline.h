/*----------------------------------------------------------------------------
 *
 *   Copyright (C) 2016 - 2026 Antonio Augusto Alves Junior
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
 * Spiline.h
 *
 *  Created on: 16/12/2018
 *      Author: Antonio Augusto Alves Junior
 */

#ifndef SPILINE_H_
#define SPILINE_H_


#include <hydra/detail/Config.h>
#include <hydra/detail/BackendPolicy.h>
#include <hydra/Types.h>
#include <hydra/Function.h>
#include <hydra/detail/utility/CheckValue.h>
#include <hydra/detail/external/hydra_thrust/copy.h>
#include <hydra/detail/external/hydra_thrust/iterator/zip_iterator.h>
#include <hydra/detail/external/hydra_thrust/execution_policy.h>
#include <hydra/detail/external/hydra_thrust/binary_search.h>
#include <hydra/detail/external/hydra_thrust/extrema.h>
#include <hydra/detail/external/hydra_thrust/iterator/iterator_traits.h>
#include <hydra/detail/Iterable_traits.h>

#include <math.h>
#include <algorithm>
#include <type_traits>
#include <hydra/detail/IteratorConcepts.h>

namespace hydra {

/**
 * @fn Type spline(Iterator1, Iterator1, Iterator2, Type)
 * @brief Cubic monotone spline interpolation
 *
 * @tparam Iterator1 Iterator type of the abcissae. Convertible to double
 * @tparam Iterator2 Iterator type of data to interpolate. Convertible to double
 * @tparam Type interpolation point data. Convertible to double
 * @param first iterator pointing to the first element of the abcissae range
 * @param last  iterator pointing to the last element of the abcissae range
 * @param measurements iterator pointing to the first element of the data range with at least (last - first) elements
 * @param value point where to calculate the interpolation
 * @return Interpolated value
 */
template<typename Iterator1, typename Iterator2, typename Type>
requires (
	std::is_convertible_v<typename hydra::thrust::iterator_traits<Iterator1>::value_type, double > &&
	std::is_convertible_v<typename hydra::thrust::iterator_traits<Iterator2>::value_type, double > &&
	std::is_convertible_v<Type, double >
)
__hydra_host__ __hydra_device__ inline Type
spline(Iterator1 first, Iterator1 last,  Iterator2 measurements, Type value);

/**
 * @fn Type spline(Iterable1&&, Iterable2&&, Type)
 * @brief
 *
 * @pre
 * @post
 * @tparam Iterable1  Iterable range of the abcissae. Convertible to double
 * @tparam Iterable2  Iterable range of data to interpolate. Convertible to double
 * @tparam Type       Interpolation point data. Convertible to double
 * @param abiscissae  range of the abcissae
 * @param measurements range of data to interpolate. Same size of abcissae
 * @param value  point where to calculate the interpolation
 * @return Interpolated value
 */
template<typename Iterable1, typename Iterable2, typename Type>
requires (
	hydra::detail::Iterable<Iterable1> &&
	hydra::detail::Iterable<Iterable2> &&
	std::is_convertible_v<typename Iterable1::value_type, double > &&
	std::is_convertible_v<typename Iterable2::value_type, double > &&
	std::is_convertible_v<Type, double >
)
__hydra_host__ __hydra_device__ inline Type
spline(Iterable1&& abiscissae,  Iterable2&& measurements, Type value);

/**
 * @fn Type spline2D(IteratorX, IteratorX, IteratorY, IteratorY, IteratorM, Type, Type)
 * @brief  Two-dimensional Cubic monotone spline interpolation
 *
 * @tparam IteratorX
 * @tparam IteratorY
 * @tparam IteratorM
 * @tparam Type
 * @param firstx
 * @param lastx
 * @param firsty
 * @param lasty
 * @param measurements
 * @param x
 * @param y
 * @return
 */
template<typename IteratorX, typename IteratorY, typename IteratorM, typename TypeX, typename TypeY>
requires (
	std::is_convertible_v<typename hydra::thrust::iterator_traits<IteratorX>::value_type, double > &&
	std::is_convertible_v<typename hydra::thrust::iterator_traits<IteratorY>::value_type, double > &&
	std::is_convertible_v<typename hydra::thrust::iterator_traits<IteratorM>::value_type, double > &&
	std::is_convertible_v<TypeX, double > &&
	std::is_convertible_v<TypeY, double >
)
__hydra_host__ __hydra_device__ inline double
spline2D(IteratorX firstx, IteratorX lastx, IteratorY firsty, IteratorY lasty, IteratorM measurements, TypeX x, TypeY y);


template<typename IterableX, typename IterableY,typename IterableM, typename TypeX,typename TypeY >
requires (
	hydra::detail::Iterable<IterableX> &&
	hydra::detail::Iterable<IterableY> &&
	hydra::detail::Iterable<IterableM> &&
	std::is_convertible_v<typename IterableX::value_type, double > &&
	std::is_convertible_v<typename IterableY::value_type, double > &&
	std::is_convertible_v<typename IterableM::value_type, double > &&
	std::is_convertible_v<TypeX, double > &&
	std::is_convertible_v<TypeY, double >
)
__hydra_host__ __hydra_device__ inline double
spline(IterableX&& abcissa_x,  IterableY&& abcissa_y, IterableM measurements, TypeX x, TypeX y);

template<typename IteratorX, typename IteratorY, typename IteratorZ, typename IteratorM, typename TypeX, typename TypeY, typename TypeZ >
requires (
	std::is_convertible_v<typename hydra::thrust::iterator_traits<IteratorX>::value_type, double > &&
	std::is_convertible_v<typename hydra::thrust::iterator_traits<IteratorY>::value_type, double > &&
	std::is_convertible_v<typename hydra::thrust::iterator_traits<IteratorZ>::value_type, double > &&
	std::is_convertible_v<typename hydra::thrust::iterator_traits<IteratorM>::value_type, double > &&
	std::is_convertible_v<TypeX, double > &&
	std::is_convertible_v<TypeY, double > &&
	std::is_convertible_v<TypeZ, double >)
__hydra_host__ __hydra_device__ inline double
spline3D(IteratorX firstx, IteratorX lastx,
		 IteratorY firsty, IteratorY lasty,
		 IteratorY firstz, IteratorY lastz,
		 IteratorM measurements, TypeX x, TypeY y, TypeZ z);

template<typename IterableX, typename IterableY,typename IterableZ,typename IterableM, typename TypeX,typename TypeY, typename TypeZ >
requires (
	hydra::detail::Iterable<IterableX> &&
	hydra::detail::Iterable<IterableY> &&
	hydra::detail::Iterable<IterableZ> &&
	hydra::detail::Iterable<IterableM> &&
	std::is_convertible_v<typename IterableX::value_type, double > &&
	std::is_convertible_v<typename IterableY::value_type, double > &&
	std::is_convertible_v<typename IterableZ::value_type, double > &&
	std::is_convertible_v<typename IterableM::value_type, double > &&
	std::is_convertible_v<TypeX, double > &&
	std::is_convertible_v<TypeY, double > &&
	std::is_convertible_v<TypeZ, double >
)
__hydra_host__ __hydra_device__ inline double
spline3D(IterableX&& abscissa_x,  IterableY&& abscissa_y, IterableZ&& abscissa_z, IterableM measurements, TypeX x, TypeX y, TypeZ z );

template<typename IteratorX, typename IteratorY, typename IteratorW, typename IteratorZ, typename IteratorM,
          typename TypeX, typename TypeY, typename TypeW, typename TypeZ >
requires (
	std::is_convertible_v<typename hydra::thrust::iterator_traits<IteratorX>::value_type, double > &&
	std::is_convertible_v<typename hydra::thrust::iterator_traits<IteratorY>::value_type, double > &&
	std::is_convertible_v<typename hydra::thrust::iterator_traits<IteratorW>::value_type, double > &&
	std::is_convertible_v<typename hydra::thrust::iterator_traits<IteratorZ>::value_type, double > &&
	std::is_convertible_v<typename hydra::thrust::iterator_traits<IteratorM>::value_type, double > &&
	std::is_convertible_v<TypeX, double > &&
	std::is_convertible_v<TypeY, double > &&
	std::is_convertible_v<TypeW, double > &&
	std::is_convertible_v<TypeZ, double >
)
__hydra_host__ __hydra_device__ inline double
spline4D(IteratorX firstx, IteratorX lastx,
		 IteratorY firsty, IteratorY lasty,
		 IteratorW firstw, IteratorW lastw,
		 IteratorZ firstz, IteratorZ lastz,
		 IteratorM measurements, TypeX x, TypeY y, TypeW w, TypeZ z);

template<typename IterableX, typename IterableY,typename IterableW,typename IterableZ,typename IterableM, typename TypeX,typename TypeY, typename TypeW, typename TypeZ >
requires (
	hydra::detail::Iterable<IterableX> &&
	hydra::detail::Iterable<IterableY> &&
	hydra::detail::Iterable<IterableW> &&
	hydra::detail::Iterable<IterableZ> &&
	hydra::detail::Iterable<IterableM> &&
	std::is_convertible_v<typename IterableX::value_type, double > &&
	std::is_convertible_v<typename IterableY::value_type, double > &&
	std::is_convertible_v<typename IterableW::value_type, double > &&
	std::is_convertible_v<typename IterableZ::value_type, double > &&
	std::is_convertible_v<typename IterableM::value_type, double > &&
	std::is_convertible_v<TypeX, double > &&
	std::is_convertible_v<TypeY, double > &&
	std::is_convertible_v<TypeW, double > &&
	std::is_convertible_v<TypeZ, double >
)
__hydra_host__ __hydra_device__ inline double
spline3D(IterableX&& abscissa_x,  IterableY&& abscissa_y, IterableW&& abscissa_w, IterableZ&& abscissa_z, IterableM measurements, TypeX x, TypeX y, TypeW w, TypeZ z );

} // namespace hydra

#include <hydra/detail/Spline.inl>
#include <hydra/detail/Spline2D.inl>
#include <hydra/detail/Spline3D.inl>
#include <hydra/detail/Spline4D.inl>
#endif /* SPILINE_H_ */
