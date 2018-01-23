/*----------------------------------------------------------------------------
 *
 *   Copyright (C) 2016 - 2017 Antonio Augusto Alves Junior
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
 * s.h
 *
 *  Created on: 29/12/2017
 *      Author: Antonio Augusto Alves Junior
 */

#ifndef ZEMACHFUNCTIONS_H_
#define ZEMACHFUNCTIONS_H_

#include <hydra/detail/Config.h>
#include <hydra/Types.h>
#include <hydra/Function.h>
#include <hydra/detail/utility/CheckValue.h>
#include <hydra/Tuple.h>
#include <tuple>
#include <limits>
#include <stdexcept>
#include <assert.h>
#include <utility>
#include <cmath>

namespace hydra {

template<unsigned int L>
double legendre_polynomial(const double x);

/**
 * @class ZemachFunction
 *
 * Zemach's angular probability distribution for 3-body decays of spinless particle into spinless final
 * states as a function of \f$\theta\f$, the helicity angle:
 *
 * 	- \f$ Z_L(\theta) \propto  (-1)^L P_L(cos(\theta))\f$
 *
 * 	\f$ The missing (2p.q)^L \f$ factors should be included included in the dynamical part. This formulation
 * 	makes Zemach's formalism fully compatible with Helicity amplitudes for 3-body decays of spinless particle into spinless final
 *  states inside Hydra.
 */
template<Wave L, unsigned int ArgIndex=0>
class ZemachFunction: public BaseFunctor<ZemachFunction<L,ArgIndex>, double, 0>{

public:
	__host__  __device__
	ZemachFunction()=default;



	__host__  __device__ inline
	ZemachFunction<L, ArgIndex>&
	operator=(ZemachFunction<L, ArgIndex>  const& other){
		if(this==&other) return  *this;
		BaseFunctor<ZemachFunction<L, ArgIndex>,
		double, 0>::operator=(other);
		return  *this;
	}

	template<typename T>
	__host__ __device__ inline
	double Evaluate(unsigned int n, T*x)  const	{

		const double theta = x[ArgIndex] ;
		return  legendre_polynomial<L>( theta );

	}

	template<typename T>
	__host__ __device__ inline
	double Evaluate(T x)  const {

		const double theta =  get<ArgIndex>(x);

		return  legendre_polynomial<L>( theta );
	}

};


template<>__host__ __device__ inline
double legendre_polynomial<0>(const double x){

	return 1.0;
}

template<>__host__ __device__ inline
double legendre_polynomial<1>(const double x){

	return -x;
}

template<>__host__ __device__ inline
double legendre_polynomial<2>(const double x){

	return 0.5*(3.0*pow<double, 2>(x) -1) ;
}

template<>__host__ __device__ inline
double legendre_polynomial<3>(const double x){

	return -0.5*(5.0*pow<double, 3>(x) - 3.0*x);
}

template<>__host__ __device__ inline
double legendre_polynomial<4>(const double x){

	return 0.125*(35.0*pow<double, 4>(x) - 30.0*pow<double, 2>(x) + 3);
}

template<>__host__ __device__ inline
double legendre_polynomial<5>(const double x){

	return -0.125*(63.0*pow<double, 5>(x) - 70.0*pow<double, 3>(x) + 15.0*x);
}


}  // namespace hydra



#endif /* ZEMACHFUNCTIONS_H_ */
