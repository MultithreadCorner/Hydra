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
 * Spiline.h
 *
 *  Created on: 16/12/2018
 *      Author: Antonio Augusto Alves Junior
 */

#ifndef SPILINE_H_
#define SPILINE_H_


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

#include <math.h>
#include <algorithm>
#include <type_traits>

namespace hydra {

template<typename Iterator1, typename Iterator2,
    typename Type=typename HYDRA_EXTERNAL_NS::thrust::iterator_traits<Iterator2>::value_type>
__hydra_host__ __hydra_device__
inline std::enable_if< std::is_floating_point<typename HYDRA_EXTERNAL_NS::thrust::iterator_traits<Iterator1>::value_type >::value &&
                       std::is_floating_point<typename HYDRA_EXTERNAL_NS::thrust::iterator_traits<Iterator2>::value_type >::value >
spiline( Type value, Iterator1 first, Iterator1 last,  Iterator2 first2) {

		using HYDRA_EXTERNAL_NS::thrust::min;

		const size_t i = HYDRA_EXTERNAL_NS::thrust::distance(first,
							HYDRA_EXTERNAL_NS::thrust::lower_bound(
									HYDRA_EXTERNAL_NS::thrust::seq, first, last, value));
		//--------------------

		const double y_i = fD[i], y_ip = fD[i+1],y_ipp = fD[i+2], y_im = fD[i-1] ;

		const double x_i = fX[i], x_ip = fX[i+1],x_ipp = fX[i+2], x_im = fX[i-1] ;

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
		const double X = (x-fX[i]);

		return X*( X*(a_i*X + b_i) + c_i) + y_i;
	}



} // namespace hydra

#endif /* SPILINE_H_ */
