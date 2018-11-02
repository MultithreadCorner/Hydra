/*----------------------------------------------------------------------------
 *
 *   Copyright (C) 2016-2017 Antonio Augusto Alves Junior
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
 * Legendre.h
 *
 *  Created on: 05/04/2018
 *      Author: Antonio Augusto Alves Junior
 */

#ifndef LEGENDRE_H_
#define LEGENDRE_H_


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

/**
 * Implementation of Legendre polynomials \f$ P_n(n) \f$ using the recursive relation (Bonnet’s recursion formula)
 *
 * \f[ (n+1)P_{n+1}(x) = (2n+1)xP_{n}(x) - nP_{n-1}(x) \f]
 *
 * @param n
 * @param x
 * @return
 */
template<typename T>
inline T legendre(unsigned n, const T x){

	switch(n) {

	case 0:

		return 1.0;

	case 1:

		return x;

	case 2:

		return 0.5*(3.0*x*x -1.0);

	default:

		T LL = x;
		T LM = static_cast<T>(0.5)*(static_cast<T>(3.0)*x*x -static_cast<T>(1.0));
		T LN = static_cast<T>(0.0);

		for(unsigned m=3; m<=n; m++){

			LN = ((static_cast<T>(2)*m - static_cast<T>(1))/m ) * x * LM -
					((static_cast<T>(m)- static_cast<T>(1))/m) * LL;
			LL = LM;    LM = LN;
		}

		return LN;
	}

}


}  // namespace hydra

#endif /* LEGENDRE_H_ */
