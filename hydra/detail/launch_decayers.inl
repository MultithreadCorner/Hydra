/*----------------------------------------------------------------------------
 *
 *   Copyright (C) 2016 Antonio Augusto Alves Junior
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
 * launch_decayer
 *
 *  Created on: Jun 16, 2016
 *      Author: Antonio Augusto Alves Junior
 */

/**
 * \file
 * \ingroup phsp
 */


#ifndef LAUNCH_DECAYER_INC
#define LAUNCH_DECAYER_INC

#include <hydra/detail/Config.h>
#include <hydra/Types.h>
#include <hydra/Containers.h>
#include <hydra/Events.h>
#include <hydra/detail/functors/DecayMother.h>
#include <hydra/detail/functors/DecayMothers.h>
#include <hydra/detail/utility/Utility_Tuple.h>

#include <thrust/iterator/counting_iterator.h>
#include <thrust/sequence.h>
#include <thrust/tuple.h>
#include <thrust/transform.h>

namespace hydra {

	namespace detail {
/*
		template<size_t ...Index>
		__host__
		inline auto get_zip_iterator(std::array<Particles_d, sizeof ...(Index)>& particles,	index_sequence<Index...>)
				-> decltype( thrust::make_zip_iterator( thrust::make_tuple( particles[Index].begin()...)) )
		{
			return thrust::make_zip_iterator(
			thrust::make_tuple(particles[Index].begin()...));
		}

		template<size_t ...Index>
		__host__
		inline auto get_zip_iterator(Particles_d& mothers, std::array<Particles_d, sizeof ...(Index)>& particles,	index_sequence<Index...>)
		-> decltype( thrust::make_zip_iterator( thrust::make_tuple(mothers.begin() , particles[Index].begin()...)) )
		{
			return thrust::make_zip_iterator(
					thrust::make_tuple(mothers.begin() , particles[Index].begin()...));
		}
*/
		template<size_t N, unsigned int BACKEND, typename GRND>
		__host__ inline
		void launch_decayer(DecayMother<N, BACKEND, GRND> const& decayer, Events<N, BACKEND>& events)
		{

			thrust::counting_iterator<GLong_t> first(0);
			thrust::counting_iterator<GLong_t> last = first + events.GetNEvents();

			std::array< typename Events<N, BACKEND>::vector_particles_iterator,N> begins;
					for(int i =0; i < N; i++)
						begins[i]= events.DaughtersBegin(i);

			auto zip_begin = get_zip_iterator(begins);

			thrust::transform(first, last, zip_begin, events.WeightsBegin(), decayer);

			return;
		}

		template<size_t N, unsigned int BACKEND, typename GRND>
		__host__ inline
		void launch_decayer(DecayMothers<N, BACKEND,GRND> const& decayer,
				typename Events<N, BACKEND>::vector_particles_iterator mothers_begin,
				Events<N, BACKEND>& events)
		{

			thrust::counting_iterator<GLong_t> first(0);
			thrust::counting_iterator<GLong_t> last = first + events.GetNEvents();

			std::array< typename Events<N, BACKEND>::vector_particles_iterator,N> begins;
			for(int i =0; i < N; i++)
				begins[i]= events.DaughtersBegin(i);

			auto zip_begin = get_zip_iterator(mothers_begin, begins);

			thrust::transform(first, last, zip_begin, events.WeightsBegin(), decayer);

			return;
		}



	}

}



#endif /* LAUNCH_DECAYER_INC */
