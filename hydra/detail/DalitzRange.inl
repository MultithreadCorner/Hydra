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
 * DalitzRange.inl
 *
 *  Created on: 30/12/2020
 *      Author: Antonio Augusto Alves Junior
 */

#ifndef DALITZRANGE_INL_
#define DALITZRANGE_INL_

#include <hydra/detail/Config.h>
#include <hydra/detail/external/hydra_thrust/tuple.h>
#include <hydra/detail/external/hydra_thrust/iterator/counting_iterator.h>
#include <hydra/detail/external/hydra_thrust/iterator/transform_iterator.h>
#include <hydra/detail/functors/DalitzSampler.h>

#include <array>

namespace hydra {

template <typename RNG = hydra_thrust::random::default_random_engine,
		      typename Functor = detail::dalitz::_unity_weight>
Range<
hydra_thrust::transform_iterator< detail::DalitzSampler<RNG, Functor>,
	hydra_thrust::counting_iterator<size_t>,
	hydra_thrust::tuple<double,double,double,double> >>
dalitz_range(double  mother_mass, const double (& masses)[3], size_t seed, size_t length=0,
		     Functor const& functor=detail::dalitz::_unity_weight{} )
{
	typedef hydra_thrust::tuple<double,double,double,double> event_type;


	typedef hydra_thrust::counting_iterator<size_t> index_iterator;

	typedef detail::DalitzSampler<RNG, Functor> decayer_type;

	auto first_index = index_iterator(0);
	auto  last_index = index_iterator( length==0 ? std::numeric_limits<size_t>::max(): length);

	auto decayer = decayer_type(mother_mass, masses, seed, functor);

    auto first_event = hydra_thrust::transform_iterator<decayer_type,index_iterator, event_type>(first_index, decayer);
    auto  last_event = hydra_thrust::transform_iterator<decayer_type,index_iterator, event_type>( last_index, decayer);

	return make_range( first_event, last_event );

}

template <typename RNG = hydra_thrust::random::default_random_engine,
	               typename Functor = detail::dalitz::_unity_weight>
Range<
hydra_thrust::transform_iterator< detail::DalitzSampler<RNG, Functor>,
	hydra_thrust::counting_iterator<size_t>,
	hydra_thrust::tuple<double,double,double,double> >>
dalitz_range(double  mother_mass, std::array<double, 3>const& masses, size_t seed, size_t length=0,
	     Functor const& functor=detail::dalitz::_unity_weight{}  )
{
	double masses_data[3]{ masses[0], masses[1], masses[2]};

	return dalitz_range( mother_mass, masses_data, seed, length, functor );

}

}  // namespace hydra

#endif /* DALITZRANGE_INL_ */
