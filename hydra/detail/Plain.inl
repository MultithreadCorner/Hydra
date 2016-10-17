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
 * Plain.inl
 *
 *  Created on: 21/08/2016
 *      Author: Antonio Augusto Alves Junior
 */

/**
 * \file
 * \ingroup numerical_integration
 */


#ifndef PLAIN_INL_
#define PLAIN_INL_

namespace hydra {

template< size_t N, typename GRND>
template<typename FUNCTOR>
GInt_t Plain<N,GRND>::Integrate(FUNCTOR const& fFunctor)
{

	// create iterators
	thrust::counting_iterator<size_t> first(0);
	thrust::counting_iterator<size_t> last = first + fNCalls;


	// compute summary statistics
	PlainState result = thrust::transform_reduce(first, last,
			detail::ProcessCallsPlainUnary<FUNCTOR,N,GRND>(const_cast<GReal_t*>(thrust::raw_pointer_cast(fXLow.data())),
					const_cast<GReal_t*>(thrust::raw_pointer_cast(fDeltaX.data())), fFunctor),
			PlainState(), detail::ProcessCallsPlainBinary() );

	fResult   = fVolume*result.fMean;
	fAbsError = fVolume*sqrt( result.fM2/(fNCalls*(fNCalls-1)) );


	return 0;

}

}

#endif /* PLAIN_INL_ */
