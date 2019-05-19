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
 * Plain.inl
 *
 *  Created on: 21/08/2016
 *      Author: Antonio Augusto Alves Junior
 */



//#ifndef PLAIN_INL_
//#define PLAIN_INL_

namespace hydra {

template< size_t N,hydra::detail::Backend BACKEND, typename GRND>
template<typename FUNCTOR>
inline std::pair<GReal_t, GReal_t>
Plain<N,hydra::detail::BackendPolicy<BACKEND>,GRND>::Integrate(FUNCTOR const& fFunctor)
{

	// create iterators
	HYDRA_EXTERNAL_NS::thrust::counting_iterator<size_t> first(0);
	HYDRA_EXTERNAL_NS::thrust::counting_iterator<size_t> last = first + fNCalls;


	// compute summary statistics
	PlainState result = HYDRA_EXTERNAL_NS::thrust::transform_reduce(system_t(), first, last,
			detail::ProcessCallsPlainUnary<FUNCTOR,N,GRND>(const_cast<GReal_t*>(HYDRA_EXTERNAL_NS::thrust::raw_pointer_cast(fXLow.data())),
					const_cast<GReal_t*>(HYDRA_EXTERNAL_NS::thrust::raw_pointer_cast(fDeltaX.data())), fSeed,fFunctor),
			PlainState(), detail::ProcessCallsPlainBinary() );

	fResult   = fVolume*result.fMean;
	fAbsError = fVolume*sqrt( result.fM2/((fNCalls-1)*(fNCalls-1)) );


	return std::make_pair(fResult, fAbsError);

}

}

//#endif /* PLAIN_INL_ */
