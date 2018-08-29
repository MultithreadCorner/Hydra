/*----------------------------------------------------------------------------
 *
 *   Copyright (C) 2016 - 2018 Antonio Augusto Alves Junior
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
 * wigner_J_unit_2.inl
 *
 *  Created on: Jul 08, 2018
 *      Author: Antonio Augusto Alves Junior
 */


#ifndef WIGNER_J_UNIT_2_INL_
#define WIGNER_J_UNIT_2_INL_


#include<hydra/functions/detail/wigner_d_function/wigner_d_function_macro.inl>

namespace hydra {

// J,M,N = {2, -2, -2}
WIGNER_D_FUNCTION(2, -2, -2, unit, 0.25*::pow((1.0 + ::cos(theta)), 2.0) )


// J,M,N = {2, -2, -1}
WIGNER_D_FUNCTION(2, -2, -1, unit, 0.5*::sin(theta)*(1.0+::cos(theta)))


// J,M,N = {2, -2, 0}
WIGNER_D_FUNCTION(2, -2, 0, unit, 0.5*math_constants::sqrt3_2*::pow(::sin(theta),2.0))


// J,M,N = {2, -2, 1}
WIGNER_D_FUNCTION(2, -2, 1, unit, 0.5*::sin(theta)*(1.0-::cos(theta)))


// J,M,N = {2, -2, 2}
WIGNER_D_FUNCTION(2, -2, 2, unit, 0.25*::pow((1.0 - ::cos(theta)), 2.0))


// J,M,N = {2, -1, -2}
WIGNER_D_FUNCTION(2, -1, -2, unit, -0.5*::sin(theta)*(1.0+::cos(theta)))


// J,M,N = {2, -1, -1}
WIGNER_D_FUNCTION(2, -1, -1, unit, ::pow(::cos(theta),2.0) + 0.5*::cos(theta)-0.5)


// J,M,N = {2, -1, 0}
WIGNER_D_FUNCTION(2, -1, 0, unit, math_constants::sqrt3_2*::sin(theta)*::cos(theta))


// J,M,N = {2, -1, 1}
WIGNER_D_FUNCTION(2, -1, 1, unit, -::pow(::cos(theta),2.0) + 0.5*::cos(theta)+0.5)


// J,M,N = {2, -1, 2}
WIGNER_D_FUNCTION(2, -1, 2, unit, 0.5*::sin(theta)*(1.0-::cos(theta)))


// J,M,N = {2, 0, -2}
WIGNER_D_FUNCTION(2, 0, -2, unit, 0.5*math_constants::sqrt3_2**::pow(::sin(theta),2))


// J,M,N = {2, 0, -1}
WIGNER_D_FUNCTION(2, 0, -1, unit, -math_constants::sqrt3_2*::sin(theta)*::cos(theta))


// J,M,N = {2, 0, 0}
WIGNER_D_FUNCTION(2, 0, 0, unit, 1.5*::pow(::cos(theta),2.0)-0.5)


// J,M,N = {2, 0, 1}
WIGNER_D_FUNCTION(2, 0, 1, unit, math_constants::sqrt3_2*::sin(theta)*::cos(theta))


// J,M,N = {2, 0, 2}
WIGNER_D_FUNCTION(2, 0, 2, unit, 0.5*math_constants::sqrt3_2**::pow(::sin(theta),2))


// J,M,N = {2, 1, -2}
WIGNER_D_FUNCTION(2, 1, -2, unit, -0.5*::sin(theta)*(1.0-::cos(theta)))


// J,M,N = {2, 1, -1}
WIGNER_D_FUNCTION(2, 1, -1, unit, -::pow(::cos(theta),2.0) + 0.5*::cos(theta) + 0.5)


// J,M,N = {2, 1, 0}
WIGNER_D_FUNCTION(2, 1, 0, unit, -math_constants::sqrt3_2*::sin(theta)*::cos(theta))


// J,M,N = {2, 1, 1}
WIGNER_D_FUNCTION(2, 1, 1, unit, ::pow(::cos(theta),2.0) + 0.5*::cos(theta) - 0.5)


// J,M,N = {2, 1, 2}
WIGNER_D_FUNCTION(2, 1, 2, unit, 0.5*::sin(theta)*(1+::cos(theta)))


// J,M,N = {2, 2, -2}
WIGNER_D_FUNCTION(2, 2, -2, unit, 0.25*::pow(1-::cos(theta),2))


// J,M,N = {2, 2, -1}
WIGNER_D_FUNCTION(2, 2, -1, unit, -0.5*::sin(theta)*(1-::cos(theta)))


// J,M,N = {2, 2, 0}
WIGNER_D_FUNCTION(2, 2, 0, unit, 0.5*math_constants::sqrt3_2*::pow(::sin(theta), 2.0))


// J,M,N = {2, 2, 1}
WIGNER_D_FUNCTION(2, 2, 1, unit, -0.5*::sin(theta)*(1+::cos(theta)))


// J,M,N = {2, 2, 2}
WIGNER_D_FUNCTION(2, 2, 2, unit, 0.25*::pow((1.0 + ::cos(theta)), 2.0))

} //namespace hydra


#endif /* WIGNER_J_UNIT_2_INL_ */
