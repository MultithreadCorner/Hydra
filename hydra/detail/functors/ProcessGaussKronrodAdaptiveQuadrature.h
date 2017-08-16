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
 * ProcessGaussKronrodAdaptiveQuadrature.h
 *
 *  Created on: 05/02/2017
 *      Author: Antonio Augusto Alves Junior
 */

#ifndef PROCESSGAUSSKRONRODADAPTIVEQUADRATURE_H_
#define PROCESSGAUSSKRONRODADAPTIVEQUADRATURE_H_

#include <hydra/detail/Config.h>
#include <hydra/Types.h>
#include <hydra/Collection.h>

namespace hydra {


template <typename FUNCTOR>
struct ProcessGaussKronrodAdaptiveQuadrature
{
	typedef thrust::tuple<GUInt_t, double, double> result_row_t;

	ProcessGaussKronrodAdaptiveQuadrature()=delete;

	ProcessGaussKronrodAdaptiveQuadrature(FUNCTOR functor):
		fFunctor(functor)
	{}

	__host__ __device__ inline
	ProcessGaussKronrodAdaptiveQuadrature(ProcessGaussKronrodAdaptiveQuadrature<FUNCTOR> const& other ):
	fFunctor(other.fFunctor)
	{}

	__host__ __device__ inline
	ProcessGaussKronrodAdaptiveQuadrature&
	operator=(ProcessGaussKronrodAdaptiveQuadrature<FUNCTOR> const& other )
	{
		if( this== &other) return *this;

		fFunctor=other.fFunctor;
		return *this;
	}

	template<typename T>
	__host__ __device__ inline
	result_row_t operator()(T row)
	{
		GUInt_t bin                      = thrust::get<0>(row);
		GReal_t abscissa_X_P             = thrust::get<1>(row);
		GReal_t abscissa_X_M             = thrust::get<2>(row);
		GReal_t abscissa_Weight          = thrust::get<3>(row);
		GReal_t rule_GaussKronrod_Weight = thrust::get<4>(row);
		GReal_t rule_Gauss_Weight        = thrust::get<5>(row);

	//	GaussKronrodCall result;

		GReal_t function_call    = abscissa_Weight*(fFunctor(abscissa_X_P)
				+ fFunctor(abscissa_X_M) ) ;
		GReal_t fGaussCall        = function_call*rule_Gauss_Weight;
		GReal_t fGaussKronrodCall = function_call*rule_GaussKronrod_Weight;

		//printf("%d %f %f\n", bin, fGaussCall, fGaussKronrodCall);
		return result_row_t(bin, fGaussCall, fGaussKronrodCall);
	}

	FUNCTOR fFunctor;
};


}  // namespace hydra


#endif /* PROCESSGAUSSKRONRODADAPTIVEQUADRATURE_H_ */
