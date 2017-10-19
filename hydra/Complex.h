/*----------------------------------------------------------------------------
 *
 *   Copyright (C) 2016-2017 Antonio Augusto Alves Junior
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

namespace hydra {

template<typename T,
			typename = typename HYDRA_EXTERNAL_NS::thrust::detail::enable_if<std::is_arithmetic<T>::value, void>::type>
using Complex =  HYDRA_EXTERNAL_NS::thrust::complex<T>;

template<typename T >
__host__ __device__ T  abs(hydra::complex<T>&& z){

	HYDRA_EXTERNAL_NS::thrust::abs(z);
}

}  // namespace hydra

#endif /* COMPLEX_H_ */

/*

 template<typename T >
 using abs = __host__ __device__ T 	thrust::abs (const complex< T > &z)

template<typename T >
__host__ __device__ T 	thrust::arg (const complex< T > &z)

template<typename T >
__host__ __device__ T 	thrust::norm (const complex< T > &z)

template<typename T >
__host__ __device__ complex< T > 	thrust::conj (const complex< T > &z)

template<typename T >
__host__ __device__ complex< T > 	thrust::polar (const T &m, const T &theta=0)

template<typename T >
__host__ __device__ complex< T > 	thrust::proj (const T &z)

template<typename T >
__host__ __device__ complex< T > 	thrust::operator* (const complex< T > &lhs, const complex< T > &rhs)

template<typename T >
__host__ __device__ complex< T > 	thrust::operator* (const complex< T > &lhs, const T &rhs)

template<typename T >
__host__ __device__ complex< T > 	thrust::operator* (const T &lhs, const complex< T > &rhs)

template<typename T >
__host__ __device__ complex< T > 	thrust::operator/ (const complex< T > &lhs, const complex< T > &rhs)

template<typename T >
__host__ __device__ complex< T > 	thrust::operator/ (const complex< T > &lhs, const T &rhs)

template<typename T >
__host__ __device__ complex< T > 	thrust::operator/ (const T &lhs, const complex< T > &rhs)

template<typename T >
__host__ __device__ complex< T > 	thrust::operator+ (const complex< T > &lhs, const complex< T > &rhs)

template<typename T >
__host__ __device__ complex< T > 	thrust::operator+ (const complex< T > &lhs, const T &rhs)

template<typename T >
__host__ __device__ complex< T > 	thrust::operator+ (const T &lhs, const complex< T > &rhs)

template<typename T >
__host__ __device__ complex< T > 	thrust::operator- (const complex< T > &lhs, const complex< T > &rhs)

template<typename T >
__host__ __device__ complex< T > 	thrust::operator- (const complex< T > &lhs, const T &rhs)

template<typename T >
__host__ __device__ complex< T > 	thrust::operator- (const T &lhs, const complex< T > &rhs)

template<typename T >
__host__ __device__ complex< T > 	thrust::operator+ (const complex< T > &rhs)

template<typename T >
__host__ __device__ complex< T > 	thrust::operator- (const complex< T > &rhs)

template<typename T >
__host__ __device__ complex< T > 	thrust::exp (const complex< T > &z)

template<typename T >
__host__ __device__ complex< T > 	thrust::log (const complex< T > &z)

template<typename T >
__host__ __device__ complex< T > 	thrust::log10 (const complex< T > &z)

template<typename T >
__host__ __device__ complex< T > 	thrust::pow (const complex< T > &x, const complex< T > &y)

template<typename T >
__host__ __device__ complex< T > 	thrust::pow (const complex< T > &x, const T &y)

template<typename T >
__host__ __device__ complex< T > 	thrust::pow (const T &x, const complex< T > &y)

template<typename T , typename U >
__host__ __device__ complex
< typename
detail::promoted_numerical_type
< T, U >::type > 	thrust::pow (const complex< T > &x, const complex< U > &y)

template<typename T , typename U >
__host__ __device__ complex
< typename
detail::promoted_numerical_type
< T, U >::type > 	thrust::pow (const complex< T > &x, const U &y)

template<typename T , typename U >
__host__ __device__ complex
< typename
detail::promoted_numerical_type
< T, U >::type > 	thrust::pow (const T &x, const complex< U > &y)

template<typename T >
__host__ __device__ complex< T > 	thrust::sqrt (const complex< T > &z)

template<typename T >
__host__ __device__ complex< T > 	thrust::cos (const complex< T > &z)

template<typename T >
__host__ __device__ complex< T > 	thrust::sin (const complex< T > &z)

template<typename T >
__host__ __device__ complex< T > 	thrust::tan (const complex< T > &z)

template<typename T >
__host__ __device__ complex< T > 	thrust::cosh (const complex< T > &z)

template<typename T >
__host__ __device__ complex< T > 	thrust::sinh (const complex< T > &z)

template<typename T >
__host__ __device__ complex< T > 	thrust::tanh (const complex< T > &z)

template<typename T >
__host__ __device__ complex< T > 	thrust::acos (const complex< T > &z)

template<typename T >
__host__ __device__ complex< T > 	thrust::asin (const complex< T > &z)

template<typename T >
__host__ __device__ complex< T > 	thrust::atan (const complex< T > &z)

template<typename T >
__host__ __device__ complex< T > 	thrust::acosh (const complex< T > &z)

template<typename T >
__host__ __device__ complex< T > 	thrust::asinh (const complex< T > &z)

template<typename T >
__host__ __device__ complex< T > 	thrust::atanh (const complex< T > &z)

template<typename ValueType , class charT , class traits >
std::basic_ostream< charT,traits > &
thrust::operator<< (std::basic_ostream< charT, traits > &os, const complex< ValueType > &z)

template<typename ValueType , typename charT , class traits >
std::basic_istream< charT,traits > &
thrust::operator>> (std::basic_istream< charT, traits > &is, complex< ValueType > &z)

template<typename T >
__host__ __device__ bool 	thrust::operator== (const complex< T > &lhs, const complex< T > &rhs)

template<typename T >
__host__ __device__ bool 	thrust::operator== (const T &lhs, const complex< T > &rhs)

template<typename T >
__host__ __device__ bool 	thrust::operator== (const complex< T > &lhs, const T &rhs)

template<typename T >
__host__ __device__ bool 	thrust::operator!= (const complex< T > &lhs, const complex< T > &rhs)

template<typename T >
__host__ __device__ bool 	thrust::operator!= (const T &lhs, const complex< T > &rhs)

template<typename T >
__host__ __device__ bool 	thrust::operator!= (const complex< T > &lhs, const T &rhs)


 */
