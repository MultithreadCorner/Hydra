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
#include <hydra/Vector4R.h>
#include <array>

#include <array>

namespace hydra {

template <size_t N>
Range<HYDRA_EXTERNAL_NS::thrust::counting_iterator<long int>>
phase_space_range(Vector4R const& mother, std::array<double, N> masses, size_t nentries ){


	typedef HYDRA_EXTERNAL_NS::thrust::tuple<double,
			Vector4R, Vector4R, Vector4R, Vector4R> event_t;

	typedef HYDRA_EXTERNAL_NS::thrust::counting_iterator<size_t> index_iterator;

	auto first_index = index_iterator(0);
	auto  last_index = index_iterator(nentries);

    auto first_event = HYDRA_EXTERNAL_NS::thrust::transform_iterator<square_root,
    		index_iterator, event_t>();

	return make_range( HYDRA_EXTERNAL_NS::thrust::counting_iterator<long int>(first),
			HYDRA_EXTERNAL_NS::thrust::counting_iterator<long int>(last) );

}

}  // namespace hydra



#endif /* PHASESPACERANGE_INL_ */
