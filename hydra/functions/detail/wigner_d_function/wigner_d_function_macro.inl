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
 * wigner_d_function_macro.inl
 *
 *  Created on: 06/07/2018
 *      Author: Antonio Augusto Alves Junior
 */

#ifndef WIGNER_D_FUNCTION_MACRO_INL_
#define WIGNER_D_FUNCTION_MACRO_INL_


#define WIGNER_D_FUNCTION(J, M, N, type, formula)\
template<> inline double wigner_d_function<_##type<J>,_##type<M>,_##type<N>>(const double theta){\
\
	return formula;\
}\

#define WIGNER_D_FUNCTION_SWAPPED_MN(J, M, N, type)\
template<> inline double wigner_d_function<_##type<J>,_##type<M>,_##type<N>>(const double theta){\
\
	return ::pow(-1, M - N)*wigner_d_function<_##type<J>,_##type<N>,_##type<M>>(theta);\
}\

#define WIGNER_D_FUNCTION_NEGATIVE_MN(J, M, N, type)\
template<> inline double wigner_d_function<_##type<J>,_##type<M>,_##type<N>>(const double theta){\
\
	return ::pow(-1, M - N)*wigner_d_function<_##type<J>,_##type<M>,_##type<N>>(theta);\
}\

#define WIGNER_D_FUNCTION_SWAPPED_NEGATIVE_MN(J, M, N, type)\
template<> inline double wigner_d_function<_##type<J>,_##type<M>,_##type<N>>(const double theta){\
\
	return wigner_d_function<_##type<J>,_##type<N>,_##type<M>>(theta);\
}\


#endif /* WIGNER_D_FUNCTION_MACRO_INL_ */
