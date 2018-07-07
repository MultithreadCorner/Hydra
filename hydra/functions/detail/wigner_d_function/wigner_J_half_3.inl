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
 * wigner_J_half_3.inl
 *
 *  Created on: Jul 5, 2018
 *      Author: Davide Brundu
 */

#ifndef WIGNER_J_HALF_3_INL_
#define WIGNER_J_HALF_3_INL_


#include<hydra/functions/detail/wigner_d_function/wigner_d_function_macro.inl>

namespace hydra {

// Generate wigner_d_function header template for J = 3/2
//
// J,M,N = {3/2, -3/2, -3/2}
WIGNER_D_FUNCTION(3, -3, -3, half, ::pow(::cos(0.5*theta),3))


// J,M,N = {3/2, -3/2, -1/2}
WIGNER_D_FUNCTION(3, -3, -1, half, math_constants::sqrt3*::pow(::cos(0.5*theta),2)*::sin(0.5*theta))


// J,M,N = {3/2, -3/2, 1/2}
WIGNER_D_FUNCTION(3, -3, 1, half, math_constants::sqrt3*::pow(::sin(0.5*theta),2)*::cos(0.5*theta))


// J,M,N = {3/2, -3/2, 3/2}
WIGNER_D_FUNCTION(3, -3, 3, half, ::pow(::sin(0.5*theta),3))


// J,M,N = {3/2, -1/2, -3/2}
WIGNER_D_FUNCTION(3, -1, -3, half, -math_constants::sqrt3*::sin(0.5*theta)*::pow(::cos(0.5*theta),2))


// J,M,N = {3/2, -1/2, -1/2}
WIGNER_D_FUNCTION(3, -1, -1, half, ::cos(0.5*theta)*(3.0*::pow(::cos(0.5*theta),2) - 2.0))


// J,M,N = {3/2, -1/2, 1/2}
WIGNER_D_FUNCTION(3, -1, 1, half, -::sin(0.5*theta)*(3.0*::pow(::sin(0.5*theta),2) - 2.0))


// J,M,N = {3/2, -1/2, 3/2}
WIGNER_D_FUNCTION(3, -1, 3, half, math_constants::sqrt3*::cos(0.5*theta)*::pow(::sin(0.5*theta),2))


// J,M,N = {3/2, 1/2, -3/2}
WIGNER_D_FUNCTION(3, 1, -3, half, math_constants::sqrt3*::cos(0.5*theta)*::pow(::sin(0.5*theta),2))


// J,M,N = {3/2, 1/2, -1/2}
WIGNER_D_FUNCTION(3, 1, -1, half, ::sin(0.5*theta)*(3.0*::pow(::sin(0.5*theta),2) - 2.0))


// J,M,N = {3/2, 1/2, 1/2}
WIGNER_D_FUNCTION(3, 1, 1, half, ::cos(0.5*theta)*(3.0*::pow(::cos(0.5*theta),2) - 2.0))


// J,M,N = {3/2, 1/2, 3/2}
WIGNER_D_FUNCTION(3, 1, 3, half, math_constants::sqrt3*::sin(0.5*theta)*::pow(::cos(0.5*theta),2))


// J,M,N = {3/2, 3/2, -3/2}
WIGNER_D_FUNCTION(3, 3, -3, half, -::pow(::sin(0.5*theta),3))


// J,M,N = {3/2, 3/2, -1/2}
WIGNER_D_FUNCTION(3, 3, -1, half, math_constants::sqrt3*::pow(::sin(0.5*theta),2)*::cos(0.5*theta))


// J,M,N = {3/2, 3/2, 1/2}
WIGNER_D_FUNCTION(3, 3, 1, half, -math_constants::sqrt3*::sin(0.5*theta)*::pow(::cos(0.5*theta),2))


// J,M,N = {3/2, 3/2, 3/2}
WIGNER_D_FUNCTION(3, 3, 3, half, ::pow(::cos(0.5*theta),3))



}  // namespace hydra


#endif /* WIGNER_J_HALF_3_INL_ */
