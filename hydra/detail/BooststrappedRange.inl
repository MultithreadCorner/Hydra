/*----------------------------------------------------------------------------
 *
 *   Copyright (C) 2016 - 2020 Antonio Augusto Alves Junior
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
 * BooststrappedRange.inl
 *
 *  Created on: 04/11/2018
 *      Author: Antonio Augusto Alves Junior
 */

#ifndef BOOSTSTRAPPEDRANGE_INL_
#define BOOSTSTRAPPEDRANGE_INL_

#include <hydra/detail/Config.h>
#include <hydra/detail/functors/RandomUtils.h>
#include <hydra/detail/external/hydra_thrust/iterator/permutation_iterator.h>
#include <utility>
namespace hydra {


template<typename Iterable>
typename std::enable_if<hydra::detail::is_iterable<Iterable>::value,
   Range< hydra_thrust::permutation_iterator<decltype(std::declval<Iterable&>().begin()),
		 hydra_thrust::transform_iterator< detail::RndUniform<size_t, hydra_thrust::random::default_random_engine>
 ,hydra_thrust::counting_iterator<size_t>,size_t > > >>::type
boost_strapped_range(Iterable&& iterable, size_t seed){

	using hydra_thrust::make_permutation_iterator;

	auto permutations = random_uniform_range(size_t(0), std::forward<Iterable>(iterable).size()-1, seed );

	return make_range(make_permutation_iterator( std::forward<Iterable>(iterable).begin(), permutations.begin()),
			make_permutation_iterator( std::forward<Iterable>(iterable).end(), permutations.end()));
}


}  // namespace hydra


#endif /* BOOSTSTRAPPEDRANGE_INL_ */
