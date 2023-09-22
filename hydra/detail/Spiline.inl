/*----------------------------------------------------------------------------
 *
 *   Copyright (C) 2016 - 2023 Antonio Augusto Alves Junior
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
 * Spiline.inl
 *
 *  Created on: 18/12/2018
 *      Author: Antonio Augusto Alves Junior
 */

#ifndef SPILINE_INL_
#define SPILINE_INL_



#include <hydra/detail/Config.h>
#include <hydra/detail/BackendPolicy.h>
#include <hydra/Types.h>
#include <hydra/Function.h>
#include <hydra/detail/utility/CheckValue.h>
#include <hydra/detail/external/hydra_thrust/copy.h>
#include <hydra/detail/external/hydra_thrust/iterator/zip_iterator.h>
#include <hydra/detail/external/hydra_thrust/execution_policy.h>
#include <hydra/detail/external/hydra_thrust/binary_search.h>
#include <hydra/detail/external/hydra_thrust/extrema.h>
#include <hydra/detail/external/hydra_thrust/iterator/iterator_traits.h>
#include <hydra/detail/external/hydra_thrust/functional.h>
#include <hydra/detail/external/hydra_thrust/advance.h>
#include <math.h>
#include <algorithm>
#include <type_traits>

namespace hydra {

namespace detail {

namespace spiline {

//thrust::lower_bound have problems in cuda backend
template<typename Iterator, typename T>
__hydra_host__ __hydra_device__
inline Iterator lower_bound(Iterator first, Iterator last, const T& value)
{
	Iterator it=first;
	typename hydra::thrust::iterator_traits<Iterator>::difference_type count, step;
	count = hydra::thrust::distance(first, last);

	while (count > 0) {
		it = first;
		step = count / 2;
		hydra::thrust::advance(it, step);
		if (*it < value) {
			first = ++it;
			count -= step + 1;
		}
		else
			count = step;
	}
	return first;
}

template<typename T>
inline typename std::enable_if< std::is_floating_point<T>::value || std::is_convertible<T, double>::value, T>::type
cubic_spiline(size_t i, size_t N,  T const (&X)[4] ,   T const (&Y)[4], T value ){

	using hydra::thrust::min;

	const T y_i = Y[1], y_ip = Y[2], y_ipp = Y[3], y_im =  Y[0];

	const T x_i = X[1]       , x_ip = X[2],    x_ipp = X[3], x_im = X[0] ;

	//calculates s
	const T  h_i  = x_ip -x_i;
	const T  h_ip = x_ipp -x_ip;
	const T  h_im = x_i  -x_im;

	const T  s_i  = (y_ip - y_i)/h_i;
	const T  s_ip = (y_ipp - y_ip)/h_ip;
	const T  s_im = (y_i - y_im)/h_im;

	const T p_i  = i==0 ? ( s_i*(1 + h_i/(h_i + h_ip)) - s_ip*h_i/(h_i + h_ip) ):
			i==N-2 ? ( s_i*(1 + h_i/(h_i + h_im)) - s_im*h_i/(h_i + h_im) )
					: (s_im*h_i + s_i*h_im)/(h_i+ h_im);

	const T p_ip = (s_i*h_ip + s_ip*h_i)/(h_ip+ h_i);


	// calculates c

	const T c_i =  i==0  ? (::copysign(1.0, p_i ) + ::copysign(1.0, s_i ))
			*min( ::fabs(s_i) , 0.5*::fabs(p_i) ):
			i==N-2 ? (::copysign(1.0, p_i ) + ::copysign(1.0, s_i ))
					*min( ::fabs(s_i) , 0.5*::fabs(p_i) ):
					(::copysign(1.0, s_im ) + ::copysign(1.0, s_i ))
					*min(min(::fabs(s_im), ::fabs(s_i)), 0.5*::fabs(p_i) );

	const T c_ip =  (::copysign(1.0, s_i ) + ::copysign(1.0, s_ip ))
		   															*min(min(::fabs(s_ip), ::fabs(s_i)), 0.5*::fabs(p_ip) );

	//calculates b
	const T b_i =  (-2*c_i - c_ip + 3*s_i)/h_i;

	//calculates a
	const T a_i = (c_i + c_ip - 2*s_i)/(h_i*h_i);

	//--------------------
	const T _X = (value-X[1]);

	return _X*( _X*(a_i*_X + b_i) + c_i) + y_i;
}

}  // namespace spiline

}  // namespace detail


template<typename Iterator1, typename Iterator2,typename Type>
__hydra_host__ __hydra_device__
inline typename std::enable_if< std::is_floating_point<typename hydra::thrust::iterator_traits<Iterator1>::value_type >::value &&
                       std::is_floating_point<typename hydra::thrust::iterator_traits<Iterator2>::value_type >::value , Type>::type
spiline(Iterator1 first, Iterator1 last,  Iterator2 measurements, Type value) {

		using hydra::thrust::min;

		auto iter = detail::spiline::lower_bound(first, last, value);
		size_t dist_i = hydra::thrust::distance(first, iter);
		size_t i = dist_i > 0 ? dist_i - 1: 0;

		size_t N = hydra::thrust::distance(first, last);

		double X[4] = {first[ (i>0)?i-1:i ], first[i], first[i+1], first[i+2]};
		double Y[4] = {measurements[ (i>0)?i-1:i ], measurements[i],  measurements[i+1], measurements[i+2]};

		//--------------------
/*
		const double y_i = measurements[i], y_ip = measurements[i+1], y_ipp = measurements[i+2], y_im =  measurements[i-1];

		const 	double x_i = first[i]       , x_ip = first[i+1],    x_ipp = first[i+2], x_im = first[i-1] ;

		//calculates s
		const double  h_i  = x_ip -x_i;
		const double  h_ip = x_ipp -x_ip;
		const double  h_im = x_i  -x_im;

		const double  s_i  = (y_ip - y_i)/h_i;
		const double  s_ip = (y_ipp - y_ip)/h_ip;
		const double  s_im = (y_i - y_im)/h_im;

		const double p_i  = i==0 ? ( s_i*(1 + h_i/(h_i + h_ip)) - s_ip*h_i/(h_i + h_ip) ):
					i==N-2 ? ( s_i*(1 + h_i/(h_i + h_im)) - s_im*h_i/(h_i + h_im) )
				: (s_im*h_i + s_i*h_im)/(h_i+ h_im);

		const double p_ip = (s_i*h_ip + s_ip*h_i)/(h_ip+ h_i);


		// calculates c

		const double c_i =  i==0  ? (::copysign(1.0, p_i ) + ::copysign(1.0, s_i ))
				*min( ::fabs(s_i) , 0.5*::fabs(p_i) ):
				i==N-2 ? (::copysign(1.0, p_i ) + ::copysign(1.0, s_i ))
						*min( ::fabs(s_i) , 0.5*::fabs(p_i) ):
					(::copysign(1.0, s_im ) + ::copysign(1.0, s_i ))
				        *min(min(::fabs(s_im), ::fabs(s_i)), 0.5*::fabs(p_i) );

		const double c_ip =  (::copysign(1.0, s_i ) + ::copysign(1.0, s_ip ))
									*min(min(::fabs(s_ip), ::fabs(s_i)), 0.5*::fabs(p_ip) );

		//calculates b
		const double b_i =  (-2*c_i - c_ip + 3*s_i)/h_i;

		//calculates a
		const double a_i = (c_i + c_ip - 2*s_i)/(h_i*h_i);

		//--------------------
		const double X = (value-*(first+i));

		return X*( X*(a_i*X + b_i) + c_i) + y_i;
		*/
		return detail::spiline::cubic_spiline(i, N, X, Y, double(value));
	}

template<typename Iterable1, typename Iterable2,typename Type>
__hydra_host__ __hydra_device__
inline typename std::enable_if< hydra::detail::is_iterable<Iterable1>::value &&
                       hydra::detail::is_iterable<Iterable2>::value &&
                       std::is_floating_point<typename Iterable1::value_type >::value &&
                       std::is_floating_point<typename Iterable2::value_type >::value,
                       Type >::type
spiline(Iterable1&& abscissae,  Iterable2&& ordinate, Type value){

	return spiline( std::forward<Iterable1>(abscissae).begin(),
			std::forward<Iterable1>(abscissae).end(),
			std::forward<Iterable2>(ordinate).begin() , value);

}



} // namespace hydra


#endif /* SPILINE_INL_ */
