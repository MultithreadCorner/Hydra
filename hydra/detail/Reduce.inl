/*----------------------------------------------------------------------------
 *
 *   Copyright (C) 2016-2017 Antonio Augusto Alves Junior
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
 * Reduce.inl
 *
 *  Created on: 11/06/2018
 *      Author: Antonio Augusto Alves Junior
 */

#ifndef REDUCE_INL_
#define REDUCE_INL_

#include <hydra/detail/Config.h>
#include <hydra/detail/BackendPolicy.h>
#include <utility>
#include <hydra/detail/external/hydra_thrust/reduce.h>
#include <hydra/detail/external/hydra_thrust/iterator/iterator_traits.h>
#include <hydra/Range.h>

namespace hydra {

template<typename Iterable>
typename std::enable_if<hydra::detail::is_iterable<Iterable>::value,
typename hydra_thrust::iterator_traits<decltype(std::declval<Iterable>().begin())>::value_type >::type
reduce(Iterable&& iterable){

	return hydra_thrust::reduce(std::forward<Iterable>(iterable).begin(),
			std::forward<Iterable>(iterable).end() );
}

template<typename Iterable, typename Functor,
 	 	 typename T = typename hydra_thrust::iterator_traits<
		     decltype(std::declval<Iterable>().begin())>::value_type >
typename std::enable_if<hydra::detail::is_iterable<Iterable>::value, T >::type
reduce(Iterable&& iterable, T const& init, Functor const& binary_functor){


	return hydra_thrust::reduce(std::forward<Iterable>(iterable).begin(),
			std::forward<Iterable>(iterable).end(), init,
			binary_functor);
}

}  // namespace hydra

#endif /* REDUCE_INL_ */
