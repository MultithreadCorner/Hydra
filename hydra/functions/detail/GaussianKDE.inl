/*----------------------------------------------------------------------------
 *
 *   Copyright (C) 2016 - 2018 Antonio Augusto Alves Junior
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
 * GaussianKDE.inl
 *
 *  Created on: 15/04/2018
 *      Author: Antonio Augusto Alves Junior
 */

#ifndef GAUSSIANKDE_INL_
#define GAUSSIANKDE_INL_

#include <hydra/detail/Config.h>
#include <hydra/detail/BackendPolicy.h>
#include <hydra/Types.h>
#include <hydra/Function.h>

#include <hydra/detail/external/thrust/transform_reduce.h>
#include <hydra/detail/external/thrust/extrema.h>

#include <math.h>
#include <algorithm>

namespace hydra {

template< size_t NBins, size_t ArgIndex>
template<typename Iterator>
void GaussianKDE<NBins, ArgIndex>::BuildKDE(Iterator begin, Iterator end, double h) {

	auto _KDE = [](double x){

		double init = 0;

		double sum  = HYDRA_EXTERNAL_NS::thrust::transform_reduce(begin, end,
				GaussianKDE<NBins, ArgIndex>::Kernel(x,fH), 0.0,
				HYDRA_EXTERNAL_NS::thrust::plus() );

		return  sum/(fH*HYDRA_EXTERNAL_NS::thrust::distance(begin, end) ) ;
	};



}

}  // namespace hydra



#endif /* GAUSSIANKDE_INL_ */
