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
 * GaussKronrodQuadrature.inl
 *
 *  Created on: 01/02/2017
 *      Author: Antonio Augusto Alves Junior
 */

#ifndef GAUSSKRONRODQUADRATURE_INL_
#define GAUSSKRONRODQUADRATURE_INL_

#include <hydra/detail/Config.h>
#include <hydra/Types.h>
#include <tuple>

namespace hydra {

namespace experimental {

template<size_t NRULE, size_t NBIN>
template<typename FUNCTOR>
std::pair<GReal_t, GReal_t> GaussKronrodQuadrature<NRULE,NBIN>::Integrate(FUNCTOR const& functor)
{
/*
 * total number of function calls = NRULE*NBIN
 * global index = call
 * rule index   = call % NRULE
 * bin  index   = call / NRULE
 *
 * - Setup the call
 *
 * each thread needs: [rule_index, abiscisa-x, abiscisa-weight ]
 */


return std::pair<GReal_t, GReal_t>(0.0, 0.0);
}

}  // namespace experimental

}  // namespace hydra


#endif /* GAUSSKRONRODQUADRATURE_INL_ */
