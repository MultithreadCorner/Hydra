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
#include <hydra/detail/external/thrust/copy.h>
#include <hydra/detail/external/thrust/iterator/zip_iterator.h>
#include <hydra/detail/external/thrust/execution_policy.h>
#include <hydra/detail/external/thrust/binary_search.h>
#include <hydra/detail/external/thrust/extrema.h>
#include <hydra/detail/external/thrust/iterator/iterator_traits.h>
#include <hydra/detail/external/thrust/functional.h>
#include <hydra/detail/external/thrust/advance.h>
#include <math.h>
#include <algorithm>
#include <type_traits>

namespace hydra {

	namespace detail {

		namespace spiline {

		//thrust::lower_bound have problems in cuda backend
		template<typename Iterator, typename T>
		__hydra_host__ __hydra_device__
		Iterator lower_bound(Iterator first, Iterator last, const T& value)
		{
		    Iterator it=first;
		    typename HYDRA_EXTERNAL_NS::thrust::iterator_traits<Iterator>::difference_type count, step;
		    count = HYDRA_EXTERNAL_NS::thrust::distance(first, last);

		    while (count > 0) {
		        it = first;
		        step = count / 2;
		        HYDRA_EXTERNAL_NS::thrust::advance(it, step);
		        if (*it < value) {
		            first = ++it;
		            count -= step + 1;
		        }
		        else
		            count = step;
		    }
		    return first;
		}



		}  // namespace spiline

	}  // namespace detail


template<typename Iterator1, typename Iterator2,typename Type>
__hydra_host__ __hydra_device__
inline typename std::enable_if< std::is_floating_point<typename HYDRA_EXTERNAL_NS::thrust::iterator_traits<Iterator1>::value_type >::value &&
                       std::is_floating_point<typename HYDRA_EXTERNAL_NS::thrust::iterator_traits<Iterator2>::value_type >::value , Type>::type
spiline(Iterator1 first, Iterator1 last,  Iterator2 measurements, Type value) {

		using HYDRA_EXTERNAL_NS::thrust::min;

		auto iter = detail::spiline::lower_bound(first, last, value);

		size_t i = HYDRA_EXTERNAL_NS::thrust::distance(first, iter);

		size_t N = HYDRA_EXTERNAL_NS::thrust::distance(first, last);

		//--------------------

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
				*min( ::fabs(s_i) , 0.5*fabs(p_i) ):
				i==N-2 ? (::copysign(1.0, p_i ) + ::copysign(1.0, s_i ))
						*min( ::fabs(s_i) , 0.5*::fabs(p_i) ):
					(::copysign(1.0, s_im ) + ::copysign(1.0, s_i ))
				        *min(min(::fabs(s_im), ::fabs(s_i)), 0.5*::fabs(p_i) );

		const double c_ip =  (::copysign(1.0, s_i ) + ::copysign(1.0, s_ip ))
									*min(min(::fabs(s_ip), ::fabs(s_i)), 0.5*::fabs(p_ip) );

		//calculates b
		const double b_i =  (-2*c_i - c_ip - 3*s_i)/h_i;

		//calculates a
		const double a_i = (c_i + c_ip - 2*s_i)/(h_i*h_i);

		//--------------------
		const double X = (value-*(first+i));

		return X*( X*(a_i*X + b_i) + c_i) + y_i;
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
