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
 * Decays.h
 *
 *  Created on: 24/02/2020
 *      Author: Antonio Augusto Alves Junior
 */

#ifndef DECAYS_H_
#define DECAYS_H_

#include <hydra/detail/Config.h>
#include <hydra/detail/BackendPolicy.h>
#include <hydra/Vector3R.h>
#include <hydra/Vector4R.h>
#include <hydra/multivector.h>
#include <hydra/Tuple.h>

namespace hydra {

/**
* \ingroup phsp
*/
template<typename Particles,  typename Backend>
class Decays;

/**
 * \ingroup phsp
 * \brief This class provides storage for N-particle states. Data is stored using SoA layout.
 * \tparam Particles list of particles in the final state
 * \tparam Backend memory space to allocate storage for the particles.
 */
template<typename ...Particles,   hydra::detail::Backend Backend>
class Decays<hydra::tuple<Particles...>, hydra::detail::BackendPolicy<Backend>>
{
	typedef hydra::detail::BackendPolicy<Backend>  system_type;
	typedef hydra_thrust::tuple<Particles...>               tuple_type;
	typedef multivector<tuple_type, system_type>    storage_type;





};


}  // namespace hydra



#endif /* DECAYS_H_ */
