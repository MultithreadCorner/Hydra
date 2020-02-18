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
 * GaussianKDE.inl
 *
 *  Created on: 15/04/2018
 *      Author: Antonio Augusto Alves Junior
 *
 *  Updated on: Feb 18 2020
 *      Author: Davide Brundu
 *         Log: Update call interface
 */

#ifndef GAUSSIANKDE_INL_
#define GAUSSIANKDE_INL_

#include <hydra/detail/Config.h>
#include <hydra/detail/BackendPolicy.h>
#include <hydra/Types.h>
#include <hydra/Function.h>
#include <hydra/detail/external/hydra_thrust/transform_reduce.h>
#include <hydra/detail/external/hydra_thrust/transform_reduce.h>
#include <hydra/detail/external/hydra_thrust/extrema.h>

#include <math.h>
#include <algorithm>

namespace hydra {

template< size_t NBins, typename ArgType>
template<typename Iterator>
__hydra_host__ __hydra_device__
inline CubicSpiline<NBins> GaussianKDE<NBins, ArgType>::BuildKDE(double min, double max, double h, Iterator begin, Iterator end) {

	auto _KDE = [=](double x){

		double init = 0;

		double sum  = hydra_thrust::transform_reduce(begin, end,
				GaussianKDE<NBins, ArgType>::Kernel(h,x), 0.0,
				hydra_thrust::plus<double>() );

		return  sum/(h*hydra_thrust::distance(begin, end) ) ;
	};

	double bin_width = (max-min)/(NBins);

	auto _X = [=](size_t index){

		return min + index*bin_width;
	};

	typedef hydra_thrust::counting_iterator<size_t> citerator;
	citerator counter(0);

	typedef hydra_thrust::transform_iterator<decltype(_X),   citerator> xiterator;
	xiterator Xiterator(counter,  _X);

	hydra_thrust::transform_iterator<decltype(_KDE), xiterator> diterator(Xiterator,  _KDE);

	return CubicSpiline<NBins>(Xiterator, diterator);


}

}  // namespace hydra



#endif /* GAUSSIANKDE_INL_ */
