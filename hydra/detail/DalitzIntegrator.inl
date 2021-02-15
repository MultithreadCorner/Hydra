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
 * DalitzIntegrator.inl
 *
 *  Created on: 15/01/2021
 *      Author: Antonio Augusto Alves Junior
 */

#ifndef DALITZINTEGRATOR_INL_
#define DALITZINTEGRATOR_INL_


namespace hydra {

template <hydra::detail::Backend BACKEND, typename GRND>
template<typename FUNCTOR>
std::pair<GReal_t, GReal_t>
DalitzIntegrator<hydra::detail::BackendPolicy<BACKEND>, GRND>::Integrate(  FUNCTOR  const& functor)
{
 return	fGenerator.AverageOn(hydra::detail::BackendPolicy<BACKEND>(), functor, fNSamples );
}


}  // namespace hydra


#endif /* DALITZINTEGRATOR_INL_ */
