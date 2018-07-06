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
 * wigner_J_half_5.inl
 *
 *  Created on: Jul 6, 2018
 *      Author: Davide Brundu
 */

#ifndef WIGNER_J_HALF_5_INL_
#define WIGNER_J_HALF_5_INL_


#include<hydra/functions/detail/wigner_d_funcion.h>

namespace hydra {

//---------------//

//J=5/2, M=5/2, N=5/2
template<>
double wigner_d_function<_half<5>,_half<5>,_half<5>>(const double theta){

	return  ::pow(::cos(0.5*theta),5) ;
}

//J=5/2, M=5/2, N=3/2
template<>
double wigner_d_function<_half<5>,_half<5>,_half<3>>(const double theta){

	return -math_constants::sqrt5*::sin(0.5*theta)*::pow(::cos(0.5*theta),4);
}

//J=5/2, M=5/2, N=1/2
template<>
double wigner_d_function<_half<5>,_half<5>,_half<1>>(const double theta){

	return  math_constants::sqrt10*::pow(::sin(0.5*theta),2)*::pow(::cos(0.5*theta),3);
}


//J=5/2, M=5/2, N=-1/2
template<>
double wigner_d_function<_half<5>,_half<5>,_half<-1>>(const double theta){

	return  -math_constants::sqrt10*::pow(::sin(0.5*theta),3)*::pow(::cos(0.5*theta),2);
}

//J=5/2, M=5/2, N=-3/2
template<>
double wigner_d_function<_half<5>,_half<5>,_half<-3>>(const double theta){

	return  math_constants::sqrt5*::cos(0.5*theta)*::pow(::sin(0.5*theta),4);
}

//J=5/2, M=5/2, N=-5/2
template<>
double wigner_d_function<_half<5>,_half<5>,_half<-5>>(const double theta){

	return  -::pow(::sin(0.5*theta),5) ;
}

//---------------//

//J=5/2, M=3/2, N=5/2
template<>
double wigner_d_function<_half<5>,_half<3>,_half<5>>(const double theta){

	return math_constants::sqrt5*::sin(0.5*theta)*::pow(::cos(0.5*theta),4); ;
}

//J=5/2, M=3/2, N=3/2
template<>
double wigner_d_function<_half<5>,_half<3>,_half<3>>(const double theta){

	return ::pow(::cos(0.5*theta),3)*(1.0 - 5.0*::pow(sin(0.5*theta),2));
}

//J=5/2, M=3/2, N=1/2
template<>
double wigner_d_function<_half<5>,_half<3>,_half<1>>(const double theta){

	return  -math_constants::sqrt2*::sin(0.5*theta)*::pow(::cos(0.5*theta),2)*(2.0 - 5.0*::pow(::sin(0.5*theta),2));
}


//J=5/2, M=3/2, N=-1/2
template<>
double wigner_d_function<_half<5>,_half<3>,_half<-1>>(const double theta){

	return  -math_constants::sqrt2*::cos(0.5*theta)*::pow(::sin(0.5*theta),2)*(2.0 - 5.0*::pow(::cos(0.5*theta),2));
}

//J=5/2, M=3/2, N=-3/2
template<>
double wigner_d_function<_half<5>,_half<3>,_half<-3>>(const double theta){

	return ::pow(::sin(0.5*theta),3)*(1.0 - 5.0*::pow(cos(0.5*theta),2));
}

//J=5/2, M=3/2, N=-5/2
template<>
double wigner_d_function<_half<5>,_half<3>,_half<-5>>(const double theta){

	return  math_constants::sqrt5*::cos(0.5*theta)*::pow(::sin(0.5*theta),4) ;
}

//---------------//

//J=5/2, M=1/2, N=5/2
template<>
double wigner_d_function<_half<5>,_half<1>,_half<5>>(const double theta){

	return  math_constants::sqrt10*::pow(::sin(0.5*theta),2)*::pow(::cos(0.5*theta),3); 
}

//J=5/2, M=1/2, N=3/2
template<>
double wigner_d_function<_half<5>,_half<1>,_half<3>>(const double theta){

	return  math_constants::sqrt2*::sin(0.5*theta)*::pow(::cos(0.5*theta),2)*(2.0 - 5.0*::pow(::sin(0.5*theta),2));
}

//J=5/2, M=1/2, N=1/2
template<>
double wigner_d_function<_half<5>,_half<1>,_half<1>>(const double theta){

	return  ::cos(0.5*theta)*(3.0 - 12.0*::pow(::cos(0.5*theta),2) + 10.0*::pow(::cos(0.5*theta),4)) ;
}


//J=5/2, M=1/2, N=-1/2
template<>
double wigner_d_function<_half<5>,_half<1>,_half<-1>>(const double theta){

	return  -::sin(0.5*theta)*(3.0 - 12.0*::pow(::sin(0.5*theta),2) + 10.0*::pow(::sin(0.5*theta),4)) ;
}

//J=5/2, M=1/2, N=-3/2
template<>
double wigner_d_function<_half<5>,_half<1>,_half<-3>>(const double theta){

	return -math_constants::sqrt2*::cos(0.5*theta)*::pow(::sin(0.5*theta),2)*(2.0 - 5.0*::pow(::cos(0.5*theta),2));
}

//J=5/2, M=1/2, N=-5/2
template<>
double wigner_d_function<_half<5>,_half<1>,_half<-5>>(const double theta){

	return  -math_constants::sqrt10*::pow(::sin(0.5*theta),3)*::pow(::cos(0.5*theta),2);
}

//---------------//

//J=5/2, M=-1/2, N=5/2
template<>
double wigner_d_function<_half<5>,_half<-1>,_half<5>>(const double theta){

	return  math_constants::sqrt10*::pow(::sin(0.5*theta),3)*::pow(::cos(0.5*theta),2); 
}

//J=5/2, M=-1/2, N=3/2
template<>
double wigner_d_function<_half<5>,_half<-1>,_half<3>>(const double theta){

	return  -math_constants::sqrt2*::cos(0.5*theta)*::pow(::sin(0.5*theta),2)*(2.0 - 5.0*::pow(::cos(0.5*theta),2));
}

//J=5/2, M=-1/2, N=1/2
template<>
double wigner_d_function<_half<5>,_half<-1>,_half<1>>(const double theta){

	return  ::sin(0.5*theta)*(3.0 - 12.0*::pow(::sin(0.5*theta),2) + 10.0*::pow(::sin(0.5*theta),4)) ;
}


//J=5/2, M=-1/2, N=-1/2
template<>
double wigner_d_function<_half<5>,_half<-1>,_half<-1>>(const double theta){

	return  ::cos(0.5*theta)*(3.0 - 12.0*::pow(::cos(0.5*theta),2) + 10.0*::pow(::cos(0.5*theta),4)) ;
}

//J=5/2, M=-1/2, N=-3/2
template<>
double wigner_d_function<_half<5>,_half<-1>,_half<-3>>(const double theta){

	return  -math_constants::sqrt2*::sin(0.5*theta)*::pow(::cos(0.5*theta),2)*(2.0 - 5.0*::pow(::sin(0.5*theta),2));
}

//J=5/2, M=-1/2, N=-5/2
template<>
double wigner_d_function<_half<5>,_half<-1>,_half<-5>>(const double theta){

	return  math_constants::sqrt10*::pow(::sin(0.5*theta),2)*::pow(::cos(0.5*theta),3);
}

//---------------//

//J=5/2, M=-3/2, N=5/2
template<>
double wigner_d_function<_half<5>,_half<-3>,_half<5>>(const double theta){

	return  math_constants::sqrt5*::cos(0.5*theta)*::pow(::sin(0.5*theta),4);
}

//J=5/2, M=-3/2, N=3/2
template<>
double wigner_d_function<_half<5>,_half<-3>,_half<3>>(const double theta){

	return  -::pow(::sin(0.5*theta),3)*(1.0 - 5.0*::pow(cos(0.5*theta),2));
}

//J=5/2, M=-3/2, N=1/2
template<>
double wigner_d_function<_half<5>,_half<-3>,_half<1>>(const double theta){

	return  -math_constants::sqrt2*::cos(0.5*theta)*::pow(::sin(0.5*theta),2)*(2.0 - 5.0*::pow(::cos(0.5*theta),2));
}


//J=5/2, M=-3/2, N=-1/2
template<>
double wigner_d_function<_half<5>,_half<-3>,_half<-1>>(const double theta){

	return  math_constants::sqrt2*::sin(0.5*theta)*::pow(::cos(0.5*theta),2)*(2.0 - 5.0*::pow(::sin(0.5*theta),2));
}

//J=5/2, M=-3/2, N=-3/2
template<>
double wigner_d_function<_half<5>,_half<-3>,_half<-3>>(const double theta){

	return ::pow(::cos(0.5*theta),3)*(1.0 - 5.0*::pow(sin(0.5*theta),2));
}

//J=5/2, M=-3/2, N=-5/2
template<>
double wigner_d_function<_half<5>,_half<-3>,_half<-5>>(const double theta){

	return  -math_constants::sqrt5*::sin(0.5*theta)*::pow(::cos(0.5*theta),4);
}

//---------------//


//J=5/2, M=-5/2, N=5/2
template<>
double wigner_d_function<_half<5>,_half<-5>,_half<5>>(const double theta){

	return  ::pow(::sin(0.5*theta),5) ;
}

//J=5/2, M=-5/2, N=3/2
template<>
double wigner_d_function<_half<5>,_half<-5>,_half<3>>(const double theta){

	return  math_constants::sqrt5*::cos(0.5*theta)*::pow(::sin(0.5*theta),4) ;
}

//J=5/2, M=-5/2, N=1/2
template<>
double wigner_d_function<_half<5>,_half<-5>,_half<1>>(const double theta){

	return  math_constants::sqrt10*::pow(::sin(0.5*theta),3)*::pow(::cos(0.5*theta),2);
}


//J=5/2, M=-5/2, N=-1/2
template<>
double wigner_d_function<_half<5>,_half<-5>,_half<-1>>(const double theta){

	return  math_constants::sqrt10*::pow(::sin(0.5*theta),2)*::pow(::cos(0.5*theta),3);
}

//J=5/2, M=-5/2, N=-3/2
template<>
double wigner_d_function<_half<5>,_half<-5>,_half<-3>>(const double theta){

	return math_constants::sqrt5*::sin(0.5*theta)*::pow(::cos(0.5*theta),4);
}

//J=5/2, M=-5/2, N=-5/2
template<>
double wigner_d_function<_half<5>,_half<-5>,_half<-5>>(const double theta){

	return  ::pow(::cos(0.5*theta),5) ;
}

}  // namespace hydra


#endif /* WIGNER_J_HALF_5_INL_ */
