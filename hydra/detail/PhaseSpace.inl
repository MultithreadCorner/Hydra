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
 * Generate.inl
 *
 *  Created on: 21/08/2016
 *      Author: Antonio Augusto Alves Junior
 */

#ifndef GENERATE_INL_
#define GENERATE_INL_

#include <thrust/distance.h>

namespace hydra {


/*
template <size_t N, typename GRND, unsigned int BACKEND>
void PhaseSpace<N,GRND>::Generate( Vector4R const& mother, Events<N, BACKEND>& events)
{

	typedef detail::BackendTraits<BACKEND> system_t;

	typedef typename system_t::template container<GReal_t> vector_real;

	vector_real masses(fMasses);

	DecayMother decayer(mother, masses, fNDaughters, fSeed);
	detail::launch_decayer(decayer, events );

	//setting maximum weight
	RealVector_d::iterator w = thrust::max_element(events.WeightsBegin(),
			fWeights.WeightsEnd());
	events.SetMaxWeight(*w);
}


template <size_t N, typename GRND, unsigned int BACKEND>
void PhaseSpace<N,GRND>::Generate(typename Events<N, BACKEND>::vector_particles_iterator mothers_begin,
		typename Events<N, BACKEND>::vector_particles_iterator mothers_end,	Events<N, BACKEND>& events)
		{

	typedef detail::BackendTraits<BACKEND> system_t;

	typedef typename system_t::template container<GReal_t> vector_real;

	vector_real masses(fMasses);

	size_t n_mothers thrust::distance(mothers_end,mothers_begin);


	if ( events.GetNEvents()  != n_mothers){
		cout << "NEvents != NMothers" << endl;
		exit(1);
	}

	DecayMothers decayer(masses, fNDaughters, fSeed);
	detail::launch_decayer(decayer,mothers_begin, events );

	RealVector_d::iterator w = thrust::max_element(events.WeightsBegin(),
			fWeights.WeightsEnd());
	events.SetMaxWeight(*w);

		}
*/
}//namespace hydra

#endif /* GENERATE_INL_ */
