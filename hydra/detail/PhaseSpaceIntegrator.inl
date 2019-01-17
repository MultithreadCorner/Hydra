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
 * PhaseSpaceIntegrator.inl
 *
 *  Created on: 25/08/2017
 *      Author: Antonio Augusto Alves Junior
 */

#ifndef PHASESPACEINTEGRATOR_INL_
#define PHASESPACEINTEGRATOR_INL_

namespace hydra {

template <size_t N, hydra::detail::Backend BACKEND, typename GRND>
template<typename FUNCTOR>
std::pair<GReal_t, GReal_t>
PhaseSpaceIntegrator<N,hydra::detail::BackendPolicy<BACKEND>, GRND>::Integrate(  FUNCTOR  const& functor)
{
 return	fGenerator.AverageOn(hydra::detail::BackendPolicy<BACKEND>(),  fMother, functor, fNSamples );
}


}  // namespace hydra



#endif /* PHASESPACEINTEGRATOR_INL_ */
