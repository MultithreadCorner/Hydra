/*----------------------------------------------------------------------------
 *
 *   Copyright (C) 2016 - 2021 Antonio Augusto Alves Junior
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
 * jacobi.h
 *
 *  Created on: 23/10/2018
 *      Author: Antonio Augusto Alves Junior
 */

#ifndef JACOBI_H_
#define JACOBI_H_



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

	namespace detail {

			namespace jacobi {

			__hydra_host__ __hydra_device__
			inline double c_n(double a, double b, unsigned n){
				return n + a + b;
			}

		}

	}


__hydra_host__ __hydra_device__
inline double jacobi(double a, double b, unsigned n, const double x){

	using hydra::detail::jacobi::c_n;

	switch(n) {

	case 0:

		return 1.0;

	case 1:

		return (a-b)*0.5 + (1.0 + (a+b)*0.5)*x;

	default:

		double JL = 1.0;
		double JM = (a-b)*0.5 + (1.0 + (a+b)*0.5)*x;
		double JN = static_cast<double>(0.0);

		for(unsigned m=2; m<=n; m++){

			JN = static_cast<double>( c_n(a, b, 2*m-1) * ( c_n(a, b, 2*m-2) * c_n(a, b, 2*m) * x + a*a - b*b ) ) * JM -
					static_cast<double>(2 * (m-1+a) * (m-1+b) * c_n(a, b, 2*m)) * JL;

			JN /=static_cast<double>( 2 * m * c_n(a, b, 2*m-2) *  c_n(a, b, m));

			JL = JM;
			JM = JN;
		}

		return JN;
	}

}

}  // namespace hydra



#endif /* JACOBI_H_ */
