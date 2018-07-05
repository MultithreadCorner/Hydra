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
 * Chebyshev_polynomials.h
 *
 *  Created on: 05/04/2018
 *      Author: Antonio Augusto Alves Junior
 */

#ifndef CHEBYSHEV_POLYNOMIALS_H_
#define CHEBYSHEV_POLYNOMIALS_H_

namespace hydra {

template<size_t N>
__hydra_host__ __hydra_device__
inline std::enable_if<N==0, double> Chebyshev_polynomial(const double ){

	return 1.0;
}

template<size_t N>
__hydra_host__ __hydra_device__
inline std::enable_if<N==1, double> Chebyshev_polynomial(const double x){

	return x;
}

template<size_t N>
__hydra_host__ __hydra_device__
inline std::enable_if< (N>2) , double> Chebyshev_polynomial(const double x){

	return 2*x*Chebyshev_polynomial<N-1>(x) - Chebyshev_polynomial<N-2>(x);
}

}  // namespace hydra


#endif /* CHEBYSHEV_POLYNOMIALS_H_ */
