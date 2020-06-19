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
 * s.h
 *
 *  Created on: 29/12/2017
 *      Author: Antonio Augusto Alves Junior
 *
 *  Updated on: Feb 21 2020
 *      Author: Davide Brundu
 *         Log: Update call interface
 */

#ifndef ZEMACHFUNCTIONS_H_
#define ZEMACHFUNCTIONS_H_

#include <hydra/detail/Config.h>
#include <hydra/detail/BackendPolicy.h>
#include <hydra/Types.h>
#include <hydra/Function.h>
#include <hydra/detail/utility/CheckValue.h>
#include <hydra/functions/Math.h>
#include <hydra/Tuple.h>
#include <tuple>
#include <limits>
#include <stdexcept>
#include <assert.h>
#include <utility>
#include <cmath>

namespace hydra {


/**
 * \ingroup common_functions
 *
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
template<Wave L, typename ArgType, typename Signature=double(ArgType)>
class ZemachFunction: public BaseFunctor<ZemachFunction<L, ArgType>, Signature, 0>{

public:

	__hydra_host__  __hydra_device__
	ZemachFunction(){};

	__hydra_host__  __hydra_device__ inline
	ZemachFunction<L, ArgType>&
	operator=(ZemachFunction<L, ArgType>  const& other){
		if(this==&other) return  *this;
		BaseFunctor<ZemachFunction<L, ArgType>, Signature, 0>::operator=(other);
		return  *this;
	}


	__hydra_host__ __hydra_device__ inline
	double Evaluate(ArgType x)  const	{

		return  legendre(L, x);

	}



};



}  // namespace hydra



#endif /* ZEMACHFUNCTIONS_H_ */
