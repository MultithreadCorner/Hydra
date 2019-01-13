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
 * GaussKronrodQuadrature.inl
 *
 *  Created on: 01/02/2017
 *      Author: Antonio Augusto Alves Junior
 */

#ifndef GAUSSKRONRODQUADRATURE_INL_
#define GAUSSKRONRODQUADRATURE_INL_

#include <hydra/detail/Config.h>
#include <hydra/detail/BackendPolicy.h>
#include <hydra/Types.h>
#include <cmath>
#include <tuple>
#include <limits>
#include <hydra/detail/external/thrust/transform_reduce.h>

namespace hydra {


template<size_t NRULE, size_t NBIN, hydra::detail::Backend  BACKEND>
template<typename FUNCTOR>
std::pair<GReal_t, GReal_t>
GaussKronrodQuadrature<NRULE, NBIN, hydra::detail::BackendPolicy<BACKEND>>::Integrate(FUNCTOR const& functor)
{
	GaussKronrodCall init{};
	init.fGaussCall =0;
	init.fGaussKronrodCall =0;

	GaussKronrodCall result = HYDRA_EXTERNAL_NS::thrust::transform_reduce(hydra::detail::BackendPolicy<BACKEND>{},
			fCallTable.begin(), fCallTable.end(),
			GaussKronrodUnary<FUNCTOR>(functor),   init, GaussKronrodBinary() );

	GReal_t error = std::max(std::numeric_limits<GReal_t>::epsilon(),
			std::pow(200.0*std::fabs(result.fGaussCall- result.fGaussKronrodCall ), 1.5));

	return std::pair<GReal_t, GReal_t>(result.fGaussKronrodCall, error);
}

}  // namespace hydra

#endif /* GAUSSKRONRODQUADRATURE_INL_ */
