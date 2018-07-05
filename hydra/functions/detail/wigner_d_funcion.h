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
 * wigner_d_funcion.h
 *
 *  Created on: Jul 4, 2018
 *      Author: Antonio Augusto Alves Junior
 */

#ifndef WIGNER_D_FUNCION_H_
#define WIGNER_D_FUNCION_H_

#include <hydra/detail/Config.h>
#include <hydra/Types.h>
#include <assert.h>
#include <utility>
#include <cmath>

namespace hydra {


template<int I>
struct _half{};

template<int I>
struct _unit{};

template<typename J, typename M, typename N>
double wigner_d_function(const double theta);

//trivial specialization
template<>
double wigner_d_function<_unit<0>,_unit<0>,_unit<0>>(const double theta){return 1.0;}

//integer J
#include <hydra/functions/detail/wigner_d_function/wigner_J_unit_1.inl>
//#include <hydra/functions/detail/wigner_d_function/wigner_J_unit_2.inl>
//#include <hydra/functions/detail/wigner_d_function/wigner_J_unit_3.inl>
//#include <hydra/functions/detail/wigner_d_function/wigner_J_unit_4.inl>
//#include <hydra/functions/detail/wigner_d_function/wigner_J_unit_5.inl>

//half-integer J
#include <hydra/functions/detail/wigner_d_function/wigner_J_half_1.inl>
//#include <hydra/functions/detail/wigner_d_function/wigner_J_half_3.inl>
//#include <hydra/functions/detail/wigner_d_function/wigner_J_half_5.inl>
///#include <hydra/functions/detail/wigner_d_function/wigner_J_half_7.inl>
//#include <hydra/functions/detail/wigner_d_function/wigner_J_half_9.inl>

}  // namespace hydra

#endif /* WIGNER_D_FUNCION_H_ */
