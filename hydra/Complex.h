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
 * Complex.h
 *
 *  Created on: 18/10/2017
 *      Author: Antonio Augusto Alves Junior
 */

#ifndef COMPLEX_H_
#define COMPLEX_H_

#include <hydra/detail/Config.h>
#include <hydra/Types.h>
#include <hydra/detail/external/thrust/complex.h>
#include <type_traits>
#include <cmath>
#include <complex>
#include <sstream>
#include <hydra/detail/external/thrust/detail/type_traits.h>

namespace hydra {

template<typename T>
using complex =  HYDRA_EXTERNAL_NS::thrust::complex<T>;


template<typename  T> __hydra_host__ __hydra_device__
T 	abs(const complex<T> & z){

	return  HYDRA_EXTERNAL_NS::thrust::abs( z );
}

template<typename T > __hydra_host__ __hydra_device__
T 	arg (const complex<T> &z){

	return HYDRA_EXTERNAL_NS::thrust::arg ( z );
}

template<typename T > __hydra_host__ __hydra_device__
T 	norm (const complex<T> &z){

	return HYDRA_EXTERNAL_NS::thrust::norm(z);
}


template<typename T > __hydra_host__ __hydra_device__
complex<T> 	conj (const complex<T> &z){

	return HYDRA_EXTERNAL_NS::thrust::conj(z);
}

template<typename T > __hydra_host__ __hydra_device__
complex<T> 	polar (const T &m, const T &theta=0){

	return HYDRA_EXTERNAL_NS::thrust::polar(m, theta);
}

template<typename T > __hydra_host__ __hydra_device__
complex<T> 	proj (const T &z){

	return HYDRA_EXTERNAL_NS::thrust::proj(z);
}

// Multiplication
using HYDRA_EXTERNAL_NS::thrust::operator*;
/*
template<typename T > __hydra_host__ __hydra_device__
complex<T> 	operator* (const complex<T> &lhs, const complex<T> &rhs){

	return HYDRA_EXTERNAL_NS::thrust::operator*( lhs, rhs);
}

template<typename T > __hydra_host__ __hydra_device__
complex<T> 	operator* (const complex<T> &lhs, const T &rhs){

	return HYDRA_EXTERNAL_NS::thrust::operator*( lhs, rhs);
}

template<typename T > __hydra_host__ __hydra_device__
complex<T> 	operator* (const T &lhs, const complex<T> &rhs){

	return HYDRA_EXTERNAL_NS::thrust::operator*( lhs, rhs);
}
*/

// Division
using HYDRA_EXTERNAL_NS::thrust::operator/;

/*
template<typename T > __hydra_host__ __hydra_device__
complex<T> 	operator/ (const complex<T> &lhs, const complex<T> &rhs){

	return HYDRA_EXTERNAL_NS::thrust::operator/( lhs, rhs);
}

template<typename T > __hydra_host__ __hydra_device__
complex<T> 	operator/ (const complex<T> &lhs, const T &rhs){

	return HYDRA_EXTERNAL_NS::thrust::operator/( lhs, rhs);
}

template<typename T > __hydra_host__ __hydra_device__
complex<T> 	operator/ (const T &lhs, const complex<T> &rhs){

	return HYDRA_EXTERNAL_NS::thrust::operator/( lhs, rhs);
}

*/

// Addition
using HYDRA_EXTERNAL_NS::thrust::operator+;
/*
template<typename T > __hydra_host__ __hydra_device__
complex<T> 	operator+ (const complex<T> &lhs, const complex<T> &rhs){

	return HYDRA_EXTERNAL_NS::thrust::operator+( lhs, rhs);
}

template<typename T > __hydra_host__ __hydra_device__
complex<T> 	operator+ (const complex<T> &lhs, const T &rhs){

	return HYDRA_EXTERNAL_NS::thrust::operator+( lhs, rhs);
}

template<typename T > __hydra_host__ __hydra_device__
complex<T> 	operator+ (const T &lhs, const complex<T> &rhs){

	return HYDRA_EXTERNAL_NS::thrust::operator+( lhs, rhs);
}
*/

//Minus
using HYDRA_EXTERNAL_NS::thrust::operator-;
/*
template<typename T > __hydra_host__ __hydra_device__
complex<T> 	operator- (const complex<T> &lhs, const complex<T> &rhs){

	return HYDRA_EXTERNAL_NS::thrust::operator-( lhs, rhs);
}

template<typename T > __hydra_host__ __hydra_device__
complex<T> 	operator- (const complex<T> &lhs, const T &rhs){

	return HYDRA_EXTERNAL_NS::thrust::operator-( lhs, rhs);
}

template<typename T> __hydra_host__ __hydra_device__
complex<T> operator-(const T &lhs, const complex<T> &rhs){

	return HYDRA_EXTERNAL_NS::thrust::operator-(lhs, rhs);
}
*/
//Unary-operators
/*
template<typename T > __hydra_host__ __hydra_device__
complex<T> 	operator+ (const complex<T> &rhs){

	return HYDRA_EXTERNAL_NS::thrust::operator+( rhs);
}

template<typename T > __hydra_host__ __hydra_device__
complex<T> 	operator- (const complex<T> &rhs){

	return HYDRA_EXTERNAL_NS::thrust::operator-( rhs);
}
*/
//transcendental functions
template<typename T > __hydra_host__ __hydra_device__
complex<T> 	exp(const complex<T> &z){

	return HYDRA_EXTERNAL_NS::thrust::exp(z);
}

template<typename T > __hydra_host__ __hydra_device__
complex<T> 	log(const complex<T> &z){

	return HYDRA_EXTERNAL_NS::thrust::log(z);
}

template<typename T > __hydra_host__ __hydra_device__
complex<T> 	log10(const complex<T> &z){

	return HYDRA_EXTERNAL_NS::thrust::log10(z);
}

template<typename T > __hydra_host__ __hydra_device__
complex<T> 	pow(const complex<T> &x, const complex<T> &y){

	return HYDRA_EXTERNAL_NS::thrust::pow(x,y);
}

template<typename T > __hydra_host__ __hydra_device__
complex<T>  pow(const complex<T> &x, const T &y){

	return HYDRA_EXTERNAL_NS::thrust::pow(x,y);
}

template<typename T > __hydra_host__ __hydra_device__
complex<T> pow(const T &x, const complex<T> &y){

	return HYDRA_EXTERNAL_NS::thrust::pow(x,y);
}


template<typename T , typename U > __hydra_host__ __hydra_device__
complex< typename HYDRA_EXTERNAL_NS::thrust::detail::promoted_numerical_type< T, U >::type >
pow(const complex<T> &x, const complex< U > &y){

	return HYDRA_EXTERNAL_NS::thrust::pow(x,y);
}

template<typename T , typename U > __hydra_host__ __hydra_device__
complex< typename HYDRA_EXTERNAL_NS::thrust::detail::promoted_numerical_type< T, U >::type >
pow(const complex<T> &x, const U &y){

	return HYDRA_EXTERNAL_NS::thrust::pow(x,y);
}

template<typename T , typename U > __hydra_host__ __hydra_device__
complex< typename HYDRA_EXTERNAL_NS::thrust::detail::promoted_numerical_type< T, U >::type >
pow(const T &x, const complex< U > &y){

	return HYDRA_EXTERNAL_NS::thrust::pow(x,y);
}

template<typename T > __hydra_host__ __hydra_device__
complex<T> 	sqrt(const complex<T> &z){

	return HYDRA_EXTERNAL_NS::thrust::sqrt(z);
}

template<typename T > __hydra_host__ __hydra_device__
complex<T> 	cos(const complex<T> &z){

	return HYDRA_EXTERNAL_NS::thrust::cos(z);
}

template<typename T > __hydra_host__ __hydra_device__
complex<T> 	sin(const complex<T> &z){

	return HYDRA_EXTERNAL_NS::thrust::sin(z);
}

template<typename T > __hydra_host__ __hydra_device__
complex<T> 	tan(const complex<T> &z){

	return HYDRA_EXTERNAL_NS::thrust::tan(z);
}

template<typename T > __hydra_host__ __hydra_device__
complex<T> cosh(const complex<T> &z){

	return HYDRA_EXTERNAL_NS::thrust::cosh(z);
}

template<typename T > __hydra_host__ __hydra_device__
complex<T> sinh(const complex<T> &z){

	return HYDRA_EXTERNAL_NS::thrust::sinh(z);
}

template<typename T > __hydra_host__ __hydra_device__
complex<T> tanh(const complex<T> &z){

	return HYDRA_EXTERNAL_NS::thrust::tanh(z);
}

template<typename T > __hydra_host__ __hydra_device__
complex<T> acos (const complex<T> &z){

	return HYDRA_EXTERNAL_NS::thrust::acos(z);
}

template<typename T > __hydra_host__ __hydra_device__
complex<T> asin(const complex<T> &z){

	return HYDRA_EXTERNAL_NS::thrust::asin(z);
}

template<typename T > __hydra_host__ __hydra_device__
complex<T> atan(const complex<T> &z){

	return HYDRA_EXTERNAL_NS::thrust::atan(z);
}

template<typename T >__hydra_host__ __hydra_device__
complex<T> acosh(const complex<T> &z){

	return HYDRA_EXTERNAL_NS::thrust::acosh(z);
}

template<typename T > __hydra_host__ __hydra_device__
complex<T> asinh(const complex<T> &z){

	return HYDRA_EXTERNAL_NS::thrust::asinh(z);
}

template<typename T > __hydra_host__ __hydra_device__
complex<T> atanh(const complex<T> &z){

	return HYDRA_EXTERNAL_NS::thrust::atanh(z);
}

//streamers
template<typename ValueType , class charT , class traits >
std::basic_ostream< charT,traits > &
operator<< (std::basic_ostream< charT, traits > &os, const complex< ValueType > &z){

	return HYDRA_EXTERNAL_NS::thrust::operator<<(os, z);
}

template<typename ValueType , typename charT , class traits >
std::basic_istream< charT,traits > &
operator>> (std::basic_istream< charT, traits > &is, complex< ValueType > &z){

	return HYDRA_EXTERNAL_NS::thrust::operator>>(is, z);
}

//logigal operators
template<typename T >__hydra_host__ __hydra_device__
bool operator==(const complex<T> &lhs, const complex<T> &rhs){

	return HYDRA_EXTERNAL_NS::thrust::operator==(lhs,rhs);
}

template<typename T >__hydra_host__ __hydra_device__
bool operator== (const T &lhs, const complex<T> &rhs){

	return HYDRA_EXTERNAL_NS::thrust::operator==(lhs,rhs);
}


template<typename T >__hydra_host__ __hydra_device__
bool operator== (const complex<T> &lhs, const T &rhs){

	return HYDRA_EXTERNAL_NS::thrust::operator==(lhs,rhs);
}


template<typename T >__hydra_host__ __hydra_device__
bool operator!= (const complex<T> &lhs, const complex<T> &rhs){

	return HYDRA_EXTERNAL_NS::thrust::operator!=(lhs,rhs);
}

template<typename T >__hydra_host__ __hydra_device__
bool operator!= (const T &lhs, const complex<T> &rhs){

	return HYDRA_EXTERNAL_NS::thrust::operator!=(lhs,rhs);
}

template<typename T >__hydra_host__ __hydra_device__
bool operator!= (const complex<T> &lhs, const T &rhs){

	return HYDRA_EXTERNAL_NS::thrust::operator!=(lhs,rhs);
}


}  // namespace hydra

#endif /* COMPLEX_H_ */


