/*----------------------------------------------------------------------------
 *
 *   Copyright (C) 2016 - 2025 Antonio Augusto Alves Junior
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
#include <hydra/detail/external/hydra_thrust/complex.h>
#include <type_traits>
#include <cmath>
#include <complex>
#include <sstream>
#include <hydra/detail/external/hydra_thrust/detail/type_traits.h>

namespace hydra {

template<typename T>
using complex =  hydra::thrust::complex<T>;


template<typename  T> __hydra_host__ __hydra_device__
T 	abs(const complex<T> & z){

	return  hydra::thrust::abs( z );
}

template<typename T > __hydra_host__ __hydra_device__
T 	arg (const complex<T> &z){

	return hydra::thrust::arg ( z );
}

template<typename T > __hydra_host__ __hydra_device__
T 	norm (const complex<T> &z){

	return hydra::thrust::norm(z);
}


template<typename T > __hydra_host__ __hydra_device__
complex<T> 	conj (const complex<T> &z){

	return hydra::thrust::conj(z);
}

template<typename T > __hydra_host__ __hydra_device__
complex<T> 	polar (const T &m, const T &theta=0){

	return hydra::thrust::polar(m, theta);
}

template<typename T > __hydra_host__ __hydra_device__
complex<T> 	proj (const T &z){

	return hydra::thrust::proj(z);
}

// Multiplication
using hydra::thrust::operator*;
/*
template<typename T > __hydra_host__ __hydra_device__
complex<T> 	operator* (const complex<T> &lhs, const complex<T> &rhs){

	return hydra::thrust::operator*( lhs, rhs);
}

template<typename T > __hydra_host__ __hydra_device__
complex<T> 	operator* (const complex<T> &lhs, const T &rhs){

	return hydra::thrust::operator*( lhs, rhs);
}

template<typename T > __hydra_host__ __hydra_device__
complex<T> 	operator* (const T &lhs, const complex<T> &rhs){

	return hydra::thrust::operator*( lhs, rhs);
}
*/

// Division
using hydra::thrust::operator/;

/*
template<typename T > __hydra_host__ __hydra_device__
complex<T> 	operator/ (const complex<T> &lhs, const complex<T> &rhs){

	return hydra::thrust::operator/( lhs, rhs);
}

template<typename T > __hydra_host__ __hydra_device__
complex<T> 	operator/ (const complex<T> &lhs, const T &rhs){

	return hydra::thrust::operator/( lhs, rhs);
}

template<typename T > __hydra_host__ __hydra_device__
complex<T> 	operator/ (const T &lhs, const complex<T> &rhs){

	return hydra::thrust::operator/( lhs, rhs);
}

*/

// Addition
using hydra::thrust::operator+;
/*
template<typename T > __hydra_host__ __hydra_device__
complex<T> 	operator+ (const complex<T> &lhs, const complex<T> &rhs){

	return hydra::thrust::operator+( lhs, rhs);
}

template<typename T > __hydra_host__ __hydra_device__
complex<T> 	operator+ (const complex<T> &lhs, const T &rhs){

	return hydra::thrust::operator+( lhs, rhs);
}

template<typename T > __hydra_host__ __hydra_device__
complex<T> 	operator+ (const T &lhs, const complex<T> &rhs){

	return hydra::thrust::operator+( lhs, rhs);
}
*/

//Minus
using hydra::thrust::operator-;
/*
template<typename T > __hydra_host__ __hydra_device__
complex<T> 	operator- (const complex<T> &lhs, const complex<T> &rhs){

	return hydra::thrust::operator-( lhs, rhs);
}

template<typename T > __hydra_host__ __hydra_device__
complex<T> 	operator- (const complex<T> &lhs, const T &rhs){

	return hydra::thrust::operator-( lhs, rhs);
}

template<typename T> __hydra_host__ __hydra_device__
complex<T> operator-(const T &lhs, const complex<T> &rhs){

	return hydra::thrust::operator-(lhs, rhs);
}
*/
//Unary-operators
/*
template<typename T > __hydra_host__ __hydra_device__
complex<T> 	operator+ (const complex<T> &rhs){

	return hydra::thrust::operator+( rhs);
}

template<typename T > __hydra_host__ __hydra_device__
complex<T> 	operator- (const complex<T> &rhs){

	return hydra::thrust::operator-( rhs);
}
*/
//transcendental functions
template<typename T > __hydra_host__ __hydra_device__
complex<T> 	exp(const complex<T> &z){

	return hydra::thrust::exp(z);
}

template<typename T > __hydra_host__ __hydra_device__
complex<T> 	log(const complex<T> &z){

	return hydra::thrust::log(z);
}

template<typename T > __hydra_host__ __hydra_device__
complex<T> 	log10(const complex<T> &z){

	return hydra::thrust::log10(z);
}

template<typename T > __hydra_host__ __hydra_device__
complex<T> 	pow(const complex<T> &x, const complex<T> &y){

	return hydra::thrust::pow(x,y);
}

template<typename T > __hydra_host__ __hydra_device__
complex<T>  pow(const complex<T> &x, const T &y){

	return hydra::thrust::pow(x,y);
}

template<typename T > __hydra_host__ __hydra_device__
complex<T> pow(const T &x, const complex<T> &y){

	return hydra::thrust::pow(x,y);
}


template<typename T , typename U > __hydra_host__ __hydra_device__
complex< typename hydra::thrust::detail::promoted_numerical_type< T, U >::type >
pow(const complex<T> &x, const complex< U > &y){

	return hydra::thrust::pow(x,y);
}

template<typename T , typename U > __hydra_host__ __hydra_device__
complex< typename hydra::thrust::detail::promoted_numerical_type< T, U >::type >
pow(const complex<T> &x, const U &y){

	return hydra::thrust::pow(x,y);
}

template<typename T , typename U > __hydra_host__ __hydra_device__
complex< typename hydra::thrust::detail::promoted_numerical_type< T, U >::type >
pow(const T &x, const complex< U > &y){

	return hydra::thrust::pow(x,y);
}

template<typename T > __hydra_host__ __hydra_device__
complex<T> 	sqrt(const complex<T> &z){

	return hydra::thrust::sqrt(z);
}

template<typename T > __hydra_host__ __hydra_device__
complex<T> 	cos(const complex<T> &z){

	return hydra::thrust::cos(z);
}

template<typename T > __hydra_host__ __hydra_device__
complex<T> 	sin(const complex<T> &z){

	return hydra::thrust::sin(z);
}

template<typename T > __hydra_host__ __hydra_device__
complex<T> 	tan(const complex<T> &z){

	return hydra::thrust::tan(z);
}

template<typename T > __hydra_host__ __hydra_device__
complex<T> cosh(const complex<T> &z){

	return hydra::thrust::cosh(z);
}

template<typename T > __hydra_host__ __hydra_device__
complex<T> sinh(const complex<T> &z){

	return hydra::thrust::sinh(z);
}

template<typename T > __hydra_host__ __hydra_device__
complex<T> tanh(const complex<T> &z){

	return hydra::thrust::tanh(z);
}

template<typename T > __hydra_host__ __hydra_device__
complex<T> acos (const complex<T> &z){

	return hydra::thrust::acos(z);
}

template<typename T > __hydra_host__ __hydra_device__
complex<T> asin(const complex<T> &z){

	return hydra::thrust::asin(z);
}

template<typename T > __hydra_host__ __hydra_device__
complex<T> atan(const complex<T> &z){

	return hydra::thrust::atan(z);
}

template<typename T >__hydra_host__ __hydra_device__
complex<T> acosh(const complex<T> &z){

	return hydra::thrust::acosh(z);
}

template<typename T > __hydra_host__ __hydra_device__
complex<T> asinh(const complex<T> &z){

	return hydra::thrust::asinh(z);
}

template<typename T > __hydra_host__ __hydra_device__
complex<T> atanh(const complex<T> &z){

	return hydra::thrust::atanh(z);
}

//streamers
template<typename ValueType , class charT , class traits >
std::basic_ostream< charT,traits > &
operator<< (std::basic_ostream< charT, traits > &os, const complex< ValueType > &z){

	return hydra::thrust::operator<<(os, z);
}

template<typename ValueType , typename charT , class traits >
std::basic_istream< charT,traits > &
operator>> (std::basic_istream< charT, traits > &is, complex< ValueType > &z){

	return hydra::thrust::operator>>(is, z);
}

//logigal operators
template<typename T >__hydra_host__ __hydra_device__
bool operator==(const complex<T> &lhs, const complex<T> &rhs){

	return hydra::thrust::operator==(lhs,rhs);
}

template<typename T >__hydra_host__ __hydra_device__
bool operator== (const T &lhs, const complex<T> &rhs){

	return hydra::thrust::operator==(lhs,rhs);
}


template<typename T >__hydra_host__ __hydra_device__
bool operator== (const complex<T> &lhs, const T &rhs){

	return hydra::thrust::operator==(lhs,rhs);
}


template<typename T >__hydra_host__ __hydra_device__
bool operator!= (const complex<T> &lhs, const complex<T> &rhs){

	return hydra::thrust::operator!=(lhs,rhs);
}

template<typename T >__hydra_host__ __hydra_device__
bool operator!= (const T &lhs, const complex<T> &rhs){

	return hydra::thrust::operator!=(lhs,rhs);
}

template<typename T >__hydra_host__ __hydra_device__
bool operator!= (const complex<T> &lhs, const T &rhs){

	return hydra::thrust::operator!=(lhs,rhs);
}


}  // namespace hydra

#endif /* COMPLEX_H_ */


