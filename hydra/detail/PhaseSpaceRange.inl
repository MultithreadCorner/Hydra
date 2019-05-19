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
 * PhaseSpaceRange.inl
 *
 *  Created on: 12/07/2018
 *      Author: Antonio Augusto Alves Junior
 */

#ifndef PHASESPACERANGE_INL_
#define PHASESPACERANGE_INL_

#include <hydra/detail/Config.h>
#include <hydra/detail/external/thrust/tuple.h>
#include <hydra/detail/external/thrust/iterator/counting_iterator.h>
#include <hydra/detail/external/thrust/iterator/transform_iterator.h>
#include <hydra/detail/functors/GenerateDecay.h>
#include <hydra/Vector4R.h>
#include <array>

#include <array>

namespace hydra {

template <size_t N>
Range<
HYDRA_EXTERNAL_NS::thrust::transform_iterator<
	detail::GenerateDecay<N,HYDRA_EXTERNAL_NS::thrust::random::default_random_engine>,
	HYDRA_EXTERNAL_NS::thrust::counting_iterator<size_t>,
	typename hydra::detail::tuple_cat_type< HYDRA_EXTERNAL_NS::thrust::tuple<double>,
				 typename hydra::detail::tuple_type<N,Vector4R>::type>::type>>
phase_space_range(Vector4R const& mother, std::array<double, N> masses, size_t nentries )
{
	typedef typename hydra::detail::tuple_cat_type<
			 HYDRA_EXTERNAL_NS::thrust::tuple<double>,
			 typename hydra::detail::tuple_type<N,Vector4R>::type
			>::type	 event_t;


	typedef HYDRA_EXTERNAL_NS::thrust::counting_iterator<size_t> index_iterator;

	typedef detail::GenerateDecay<N,HYDRA_EXTERNAL_NS::thrust::random::default_random_engine> decayer_t;

	auto first_index = index_iterator(0);
	auto  last_index = index_iterator(nentries);

	auto decayer = decayer_t(mother, masses,456852);

    auto first_event = HYDRA_EXTERNAL_NS::thrust::transform_iterator<decayer_t,index_iterator, event_t>(first_index, decayer);
    auto  last_event = HYDRA_EXTERNAL_NS::thrust::transform_iterator<decayer_t,index_iterator, event_t>(last_index, decayer);

	return make_range( first_event, last_event );

}

}  // namespace hydra



#endif /* PHASESPACERANGE_INL_ */
