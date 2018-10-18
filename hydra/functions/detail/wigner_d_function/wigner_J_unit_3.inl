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
 * wigner_J_unit_3.inl
 *
 *  Created on: Jul 08, 2018
 *      Author: Antonio Augusto Alves Junior
 */


#ifndef WIGNER_J_UNIT_3_INL_
#define WIGNER_J_UNIT_3_INL_


#include<hydra/functions/detail/wigner_d_function/wigner_d_function_macro.inl>

namespace hydra {

// J,M,N = {3, -3, -3}
WIGNER_D_FUNCTION_SWAPPED_MN(3, 3, 3, unit)


// J,M,N = {3, -3, -2}
WIGNER_D_FUNCTION_SWAPPED_MN(3, 2, 3, unit)


// J,M,N = {3, -3, -1}
WIGNER_D_FUNCTION_SWAPPED_NEGATIVE_MN(3, 1, 3, unit)


// J,M,N = {3, -3, 0}
WIGNER_D_FUNCTION_SWAPPED_NEGATIVE_MN(3, 0, 3, unit)


// J,M,N = {3, -3, 1}
WIGNER_D_FUNCTION_SWAPPED_NEGATIVE_MN(3, -1, 3, unit)


// J,M,N = {3, -3, 2}
WIGNER_D_FUNCTION_SWAPPED_NEGATIVE_MN(3, -2, 3, unit)


// J,M,N = {3, -3, 3}
WIGNER_D_FUNCTION_SWAPPED_NEGATIVE_MN(3, -3, 3, unit)


// J,M,N = {3, -2, -3}
WIGNER_D_FUNCTION_SWAPPED_NEGATIVE_MN(3, 2, 3, unit)


// J,M,N = {3, -2, -2}
WIGNER_D_FUNCTION_SWAPPED_NEGATIVE_MN(3, 2, 2, unit)


// J,M,N = {3, -2, -1}
WIGNER_D_FUNCTION_SWAPPED_NEGATIVE_MN(3, 1, 2, unit)


// J,M,N = {3, -2, 0}
WIGNER_D_FUNCTION_SWAPPED_NEGATIVE_MN(3, 0, 2, unit)


// J,M,N = {3, -2, 1} => {3, 2, -1}
WIGNER_D_FUNCTION_SWAPPED_NEGATIVE_MN(3, -1, 2, unit )


// J,M,N = {3, -2, 2} => {3, 2, -2}
WIGNER_D_FUNCTION_SWAPPED_MN(3, -2, 2, unit)


// J,M,N = {3, -2, 3} => {3, 3, -2}
WIGNER_D_FUNCTION_SWAPPED_MN(3, -2, 3, unit)


// J,M,N = {3, -1, -3} => {3, 3, 1}
WIGNER_D_FUNCTION_SWAPPED_NEGATIVE_MN(3, 1, 3, unit)


// J,M,N = {3, -1, -2} => {3, 2, 1}
WIGNER_D_FUNCTION_SWAPPED_NEGATIVE_MN(3, 1, 2, unit)


// J,M,N = {3, -1, -1} => {3, 1, 1}
WIGNER_D_FUNCTION_SWAPPED_NEGATIVE_MN(3, 1, 1, unit)


// J,M,N = {3, -1, 0} => {3, 1, 0}
WIGNER_D_FUNCTION_SWAPPED_NEGATIVE_MN(3, 1, 0, unit)


// J,M,N = {3, -1, 1} => {3, 1, -1}
WIGNER_D_FUNCTION_SWAPPED_MN(3, -1, 1, unit)


// J,M,N = {3, -1, 2} => {3, 2, -1}
WIGNER_D_FUNCTION_SWAPPED_MN(3, -1, 2, unit)


// J,M,N = {3, -1, 3} => {3, 3, -1}
WIGNER_D_FUNCTION_SWAPPED_MN(3, -1, 3, unit)


// J,M,N = {3, 0, -3} => {3, 3, 0}
WIGNER_D_FUNCTION_SWAPPED_NEGATIVE_MN(3, 0, 3, unit)


// J,M,N = {3, 0, -2} => {3, 2, 0}
WIGNER_D_FUNCTION_SWAPPED_NEGATIVE_MN(3, 0, 2, unit)


// J,M,N = {3, 0, -1} => {3, -1, 0}
WIGNER_D_FUNCTION_SWAPPED_NEGATIVE_MN(3, 0, 1, unit)


// J,M,N = {3, 0, 0}
WIGNER_D_FUNCTION(3, 0, 0, unit, -0.5*::cos(theta)*(3.0 - 5.0*::pow( ::cos(theta), 2.0) ))


// J,M,N = {3, 0, 1}
WIGNER_D_FUNCTION_SWAPPED_MN(3, 0, 1, unit)


// J,M,N = {3, 0, 2}
WIGNER_D_FUNCTION_SWAPPED_MN(3, 0, 2, unit)


// J,M,N = {3, 0, 3}
WIGNER_D_FUNCTION_SWAPPED_MN(3, 0, 3, unit)


// J,M,N = {3, 1, -3} => {3, 3,-1}
WIGNER_D_FUNCTION_SWAPPED_NEGATIVE_MN(3, -1, 3, unit)


// J,M,N = {3, 1, -2} => {3, 2,-1}
WIGNER_D_FUNCTION_SWAPPED_NEGATIVE_MN(3, -1, 2, unit)


// J,M,N = {3, 1, -1}
WIGNER_D_FUNCTION(3, 1, -1, unit, -0.125*(1.0 - ::cos(theta))( 1.0 - 10.0*::cos(theta) - 15.0*::pow(::cos(theta) , 2.0)) )


// J,M,N = {3, 1, 0}
WIGNER_D_FUNCTION(3, 1, 0, unit, 0.25*math_constants::sqrt3*::sin(theta)*(1.0 - 5.0*::pow(::cos(theta) , 2.0)))


// J,M,N = {3, 1, 1}
WIGNER_D_FUNCTION(3, 1, 1, unit, -0.125*(1.0 + ::cos(theta))( 1.0 + 10.0*::cos(theta) + 15.0*::pow(::cos(theta) , 2.0)) )


// J,M,N = {3, 1, 2} => {3, 2, 1}
WIGNER_D_FUNCTION_SWAPPED_MN(3, 1, 2, unit)


// J,M,N = {3, 1, 3} => {3, 3, 1}
WIGNER_D_FUNCTION_SWAPPED_MN(3, 1, 3, unit)


// J,M,N = {3, 2, -3} => {3, -3, 2}
WIGNER_D_FUNCTION_SWAPPED_MN(3, 2, -3, unit)


// J,M,N = {3, 2, -2}
WIGNER_D_FUNCTION(3, 2, -2, unit, -0.25*::pow( 1.0 - ::cos(theta), 2.0)*(2.0 + 3.0*::cos(theta)) )


// J,M,N = {3, 2, -1}
WIGNER_D_FUNCTION(3, 2, -1, unit, -0.125*math_constants::sqrt10*::sin(theta)*(1.0 + 2.0*::cos(theta) - 3.0*::pow( ::cos(theta), 2.0)) )


// J,M,N = {3, 2, 0}
WIGNER_D_FUNCTION(3, 2, 0, unit,  -0.25*math_constants::sqrt30*::cos(theta)*::pow(::sin(theta), 2.0))


// J,M,N = {3, 2, 1}
WIGNER_D_FUNCTION(3, 2, 1, unit, -0.125*math_constants::sqrt10*::sin(theta)*(1.0 - 2.0*::cos(theta) - 3.0*::pow( ::cos(theta), 2.0)) )


// J,M,N = {3, 2, 2}
WIGNER_D_FUNCTION(3, 2, 2, unit, -0.25*::pow( 1.0 + ::cos(theta), 2.0)*(2.0 - 3.0*::cos(theta)) )


// J,M,N = {3, 2, 3} => {3, 3, 2}
WIGNER_D_FUNCTION_SWAPPED_MN(3, 2, 3, unit)


// J,M,N = {3, 3, -3}
WIGNER_D_FUNCTION(3, 3, -3, unit,  0.125*::pow( 1.0 + ::cos(theta), 3.0))


// J,M,N = {3, 3, -2}
WIGNER_D_FUNCTION(3, 3, -2, unit,  -0.125*math_constants::sqrt6*::sin(theta)*::pow( 1.0 - ::cos(theta), 2.0) )


// J,M,N = {3, 3, -1}
WIGNER_D_FUNCTION(3, 3, -1, unit, -0.125*math_constants::sqrt15*::pow(::sin(theta),2.0)*(1.0-::cos(theta)) )


// J,M,N = {3, 3, 0}
WIGNER_D_FUNCTION(3, 3, 0, unit, 0.25*math_constants::sqrt5*::pow(::sin(theta),3.0))


// J,M,N = {3, 3, 1}
WIGNER_D_FUNCTION(3, 3, 1, unit, -0.125*math_constants::sqrt15*::pow(::sin(theta),2.0)*(1.0+::cos(theta)) )


// J,M,N = {3, 3, 2}
WIGNER_D_FUNCTION(3, 3, 2, unit, -0.125*math_constants::sqrt6*::sin(theta)*::pow( 1.0 + ::cos(theta), 2.0) )


// J,M,N = {3, 3, 3}
WIGNER_D_FUNCTION(3, 3, 3, unit, 0.125*::pow( 1.0 + ::cos(theta), 3.0))

} //namespace hydra


#endif /* WIGNER_J_UNIT_3_INL_ */