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
 * wigner_J_unit_1.inl
 *
 *  Created on: Jul 5, 2018
 *      Author: Antonio Augusto Alves Junior
 */

#ifndef WIGNER_J_UNIT_1_INL_
#define WIGNER_J_UNIT_1_INL_


#include<hydra/functions/detail/wigner_d_funcion.h>

namespace hydra {

//J=1, M=1, N=1
template<>
double wigner_d_function<_unit<1>,_unit<1>,_unit<1>>(const double theta){

	return 0.5*(1.0 + ::cos(theta) );
}
//J=1, M=1, N=0
template<>
double wigner_d_function<_unit<1>,_unit<1>,_unit<0>>(const double theta){

	return -math_constants::sqrt2*::sin(theta);
}
//J=1, M=1, N=-1
template<>
double wigner_d_function<_unit<1>,_unit<1>,_unit<-1>>(const double theta){

	return  0.5*(1.0 - ::cos(theta) );
}
//----------------------------

//J=1, M=0, N=1
template<>
double wigner_d_function<_unit<1>,_unit<0>,_unit<1>>(const double theta){

	return math_constants::sqrt2*::sin(theta);
}
//J=1, M=0, N=0
template<>
double wigner_d_function<_unit<1>,_unit<0>,_unit<0>>(const double theta){

	return ::cos(theta);
}
//J=1, M=0, N=-1
template<>
double wigner_d_function<_unit<1>,_unit<0>,_unit<-1>>(const double theta){

	return  -math_constants::sqrt2*::sin(theta);
}

//----------------------------

//J=1, M=-1, N=1
template<>
double wigner_d_function<_unit<1>,_unit<0>,_unit<1>>(const double theta){

	return 0.5*(1.0 - ::cos(theta) );
}
//J=1, M=-1, N=0
template<>
double wigner_d_function<_unit<1>,_unit<0>,_unit<0>>(const double theta){

	return math_constants::sqrt2*::sin(theta);
}
//J=1, M=-1, N=-1
template<>
double wigner_d_function<_unit<1>,_unit<0>,_unit<-1>>(const double theta){

	return  0.5*(1.0 + ::cos(theta) );
}





}  // namespace hydra


#endif /* WIGNER_J_UNIT_1_INL_ */
