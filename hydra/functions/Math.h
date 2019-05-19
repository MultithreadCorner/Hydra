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
 * Math.h
 *
 *  Created on: 08/04/2018
 *      Author: Antonio Augusto Alves Junior
 */

#ifndef MATH_H_
#define MATH_H_




#include <hydra/detail/Config.h>
#include <hydra/detail/BackendPolicy.h>
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

#include "hydra/functions/detail/rint.h"

namespace hydra {

/**
 * Implementation of Legendre polynomials \f$ P_n(n) \f$ using the recursive relation
 *
 * \f[ (n+1)P_{n+1}(x) = (2n+1)xP_{n}(x) - nP_{n-1}(x) \f]
 *
 * @param n order of the polynomial
 * @param x argument
 * @return
 */
template<typename T>
__hydra_host__ __hydra_device__
inline T legendre(unsigned n, const T x);

/**
 * Implementation of Laguerre polynomials \f$ P_n(n) \f$ using the recursive relation
 *
 * \f[ P_{n + 1}(x) = \frac{(2n + 1 - x)P_n(x) - n P_{n - 1}(x)}{n + 1} \f]
 *
 * @param n order of the polynomial
 * @param x argument
 * @return
 */
template<typename T>
__hydra_host__ __hydra_device__
inline T laguerre(unsigned n, const T x);

/**
 * Implementation of Chebychev polynomials of first kind \f$ P_n(n) \f$ using the recursive relation
 *
 * \f[P_{n+1}(x) = 2xP_n(x)-P_{n-1}(x)  \f]
 *
 * @param n order of the polynomial
 * @param x argument
 * @return
 */
template<typename T>
__hydra_host__ __hydra_device__
inline T chebychev_1st_kind(unsigned n, const T x);

/**
 * Implementation of Chebychev polynomials of second kind \f$ P_n(n) \f$ using the recursive relation
 *
 * \f[ P_{n+1}(x) = 2xP_n(x) - P_{n-1}(x) \f]
 *
 * @param n order of the polynomial
 * @param x argument
 * @return
 */
template<typename T>
__hydra_host__ __hydra_device__
inline T chebychev_2nd_kind(unsigned n, const T x);

/**
 * Implementation of Hermite polynomials \f$ P_n(n) \f$ using the recursive relation
 *
 * \f[ P_{n + 1}(x) = 2xP_n(x) - 2nP_{n - 1}(x) \f]
 *
 * @param n order of the polynomial
 * @param x argument
 * @return
 */
template<typename T>
__hydra_host__ __hydra_device__
inline T hermite(unsigned n, const T x);

/**
 * Recurrence relation|recurrence relation]] for the Jacobi polynomials of fixed ''α'',''β'' is:<ref name=sz/>

:<math>\begin{align}
&2n (n + \alpha + \beta) (2n + \alpha + \beta - 2) P_n^{(\alpha,\beta)}(z) \\
&\qquad= (2n+\alpha + \beta-1) \Big\{ (2n+\alpha + \beta)(2n+\alpha+\beta-2) z +  \alpha^2 - \beta^2 \Big\} P_{n-1}^{(\alpha,\beta)}(z) - 2 (n+\alpha - 1) (n + \beta-1) (2n+\alpha + \beta) P_{n-2}^{(\alpha, \beta)}(z),
\end{align}</math>

for ''n'' = 2, 3, ....
 */
template<typename T>
__hydra_host__ __hydra_device__
inline T jacobi(double a, double b, unsigned n, const T x);


template<typename T>
__hydra_host__ __hydra_device__
inline T wigner_d_matrix(unsigned j, unsigned m, unsigned n, const T theta);


}  // namespace hydra

#include "hydra/functions/detail/hermite.h"
#include "hydra/functions/detail/legendre.h"
#include "hydra/functions/detail/laguerre.h"
#include "hydra/functions/detail/chebychev.h"
#include "hydra/functions/detail/jacobi.h"
#include "hydra/functions/detail/wigner_d_matrix.h"

#endif /* MATH_H_ */
