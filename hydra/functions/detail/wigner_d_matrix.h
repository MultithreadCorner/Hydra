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
 * wigner_d_matrix.h
 *
 *  Created on: 23/10/2018
 *      Author: Antonio Augusto Alves Junior
 */

#ifndef WIGNER_D_MATRIX_H_
#define WIGNER_D_MATRIX_H_





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


template<typename T>
__hydra_host__ __hydra_device__
inline T wigner_d_matrix(double j, double m, double n, const T theta){


	double mu = ::fabs(rint(m-n));
	double nu = ::fabs(rint(m+n));
	unsigned s	= ::rint(j-0.5*(mu+nu));
	int      xi = n>=m ? 1: ::pow(-1,n-m);

	double factor = ::sqrt(::tgamma(s+1.0)*::tgamma(s+mu+nu+1.0)/(::tgamma(s+mu+1.0)*::tgamma(s+nu+1.0)));
    // FIXME:
	// all previous definitions expensive are independent of theta and can be saved if
	// wigner_d_matrix is promoted to a functor
	return xi*factor*::pow(::sin(theta*0.5),mu)*::pow(::cos(theta*0.5),nu)*jacobi(mu,nu,s, ::cos(theta));

}

}  // namespace hydra




#endif /* WIGNER_D_MATRIX_H_ */
