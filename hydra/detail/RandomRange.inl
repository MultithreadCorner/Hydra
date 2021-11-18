/*----------------------------------------------------------------------------
 *
 *   Copyright (C) 2016 - 2021 Antonio Augusto Alves Junior
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
 * RandonRange.inl
 *
 *  Created on: 20/05/2018
 *      Author: Antonio Augusto Alves Junior
 */

#ifndef RANDONRANGE_INL_
#define RANDONRANGE_INL_



#include <hydra/detail/Config.h>
#include <hydra/detail/functors/RandomUtils.h>
#include <hydra/detail/external/hydra_thrust/iterator/constant_iterator.h>
#include <hydra/detail/external/hydra_thrust/iterator/transform_iterator.h>
#include <hydra/detail/functors/DistributionSampler.h>
#include <hydra/detail/PRNGTypedefs.h>

namespace hydra {


template<typename Engine=hydra::default_random_engine, typename Functor>
Range< hydra_thrust::transform_iterator< detail::Sampler<Functor,Engine >,
		 hydra_thrust::counting_iterator<std::size_t>,
		 typename detail::Sampler<Functor,Engine>::value_type > >
random_range( Functor const& functor,  std::size_t seed=0x8ec74d321e6b5a27,  std::size_t length=0,std::size_t rng_jump=0) {

	typedef hydra_thrust::counting_iterator<std::size_t> index_t;
	typedef detail::Sampler<Functor,Engine>      sampler_t;
	typedef typename detail::Sampler<Functor,Engine>::value_type  value_t;

	index_t first(0);
	index_t last( length==0 ? std::numeric_limits<std::size_t>::max() : length);

	auto sampler= detail::Sampler<Functor, Engine>(functor, seed, rng_jump);

	return make_range(
			 hydra_thrust::transform_iterator<sampler_t, index_t, value_t>(first, sampler),
		     hydra_thrust::transform_iterator<sampler_t, index_t, value_t>( last, sampler) );

}

}  // namespace hydra

#endif /* RANDONRANGE_INL_ */
