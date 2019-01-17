/*----------------------------------------------------------------------------
 *
 *   Copyright (C) 2016 - 2019 Antonio Augusto Alves Junior
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
 * CountingRange.inl
 *
 *  Created on: 20/05/2018
 *      Author: Antonio Augusto Alves Junior
 */

#ifndef COUNTINGRANGE_INL_
#define COUNTINGRANGE_INL_

#include <hydra/detail/Config.h>
#include <hydra/detail/external/thrust/iterator/counting_iterator.h>
#include <type_traits>

namespace hydra {

namespace detail {

namespace range {

template<typename T>
struct Shift
{
	Shift(T min,  T delta):
		fMin(min),
		fDelta(delta)
	{}

	__hydra_host__ __hydra_device__
	Shift( Shift<T> const& other):
	fMin(other.fMin),
	fDelta(other.fDelta)
	{}

	inline T operator()(unsigned bin){
		return fMin + bin*fDelta;
	}

private:
	T fMin;
	T fDelta;
};

}  // namespace range

}  // namespace detail

Range<HYDRA_EXTERNAL_NS::thrust::counting_iterator<long int>>
range(long int first, long int last ){

	return make_range( HYDRA_EXTERNAL_NS::thrust::counting_iterator<long int>(first),
			HYDRA_EXTERNAL_NS::thrust::counting_iterator<long int>(last) );
}

template<typename T>
inline typename std::enable_if< std::is_floating_point<T>::value,
   Range<HYDRA_EXTERNAL_NS::thrust::counting_iterator<unsigned>, detail::range::Shift<T>> >::type
range(T min, T max, unsigned nbins ){

	T delta = (max-min)/nbins;

	return make_range( HYDRA_EXTERNAL_NS::thrust::counting_iterator<unsigned>(0),
			HYDRA_EXTERNAL_NS::thrust::counting_iterator<unsigned>(nbins),
			detail::range::Shift<T>(min, delta) );
}



}  // namespace hydra

#endif /* COUNTINGRANGE_INL_ */
