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
 * wigner_J_half_1.inl
 *
 *  Created on: Jul 5, 2018
 *      Author: Antonio Augusto Alves Junior
 */

#ifndef WIGNER_J_HALF_1_INL_
#define WIGNER_J_HALF_1_INL_


#include<hydra/functions/detail/wigner_d_funcion.h>

namespace hydra {

//J=1/2, M=1/2, N=1/2

template<>
double wigner_d_function<_half<1>,_half<1>,_half<1>>(const double theta){

	return  ::cos(0.5*theta) ;
}
//J=1/2, M=1/2, N=-1/2
template<>
double wigner_d_function<_half<1>,_half<1>,_half<-1>>(const double theta){

	return -::sin(0.5*theta);
}
//J=1/2, M=-1/2, N=1/2
template<>
double wigner_d_function<_half<1>,_half<1>,_half<-1>>(const double theta){

	return  -::sin(0.5*theta);
}


//J=1/2, M=-1/2, N=-1/2
template<>
double wigner_d_function<_half<1>,_half<0>,_half<1>>(const double theta){

	return ::cos(0.5*theta) ;
}


}  // namespace hydra


#endif /* WIGNER_J_HALF_1_INL_ */
