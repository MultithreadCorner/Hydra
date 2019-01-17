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
 * CubicSpiline.h
 *
 *  Created on: 23/12/2017
 *      Author: Antonio Augusto Alves Junior
 */

#ifndef CUBICSPILINE_H_
#define CUBICSPILINE_H_


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
#include <math.h>
#include <algorithm>

namespace hydra {

/**
 * \class CubicSpiline
 *
 * A simple method for a one—dimensional
interpolation on a given set of data points (xi, yi). In each interval
(xi,x,-+1) the interpolation function is assumed to be a third-
order polynomial passing through the data points. The slope at
each grid point is determined in such a way as to guarantee
a monotonic behavior of the interpolating function. The result
is a smooth curve with continuous ﬁrst-order derivatives that
passes through any given set of data points without spurious
oscillations. Local extrema can occur only at grid points where
they are given by the data, but not in between two adjacent
grid points. The method gives exact results if the data points
correspond to a second-order polynomial.

Reference: M. Steffen, Astron. Astrophys. 239, 443—450 (1990).
*/



template<size_t N, unsigned int ArgIndex=0>
class CubicSpiline: public BaseFunctor<CubicSpiline<N, ArgIndex>, GReal_t , 0>
{

public:


	CubicSpiline() = default;

	template<typename Iterator1, typename Iterator2>
	CubicSpiline( Iterator1 xbegin, Iterator2 ybegin ):
	BaseFunctor<CubicSpiline<N, ArgIndex>, GReal_t , 0>()
	{
		//populates fH and fX
		HYDRA_EXTERNAL_NS::thrust::copy( ybegin, ybegin+N,  fD );
		HYDRA_EXTERNAL_NS::thrust::copy( xbegin, xbegin+N,  fX);

	}

	__hydra_host__ __hydra_device__
	CubicSpiline(CubicSpiline<N, ArgIndex> const& other ):
	BaseFunctor<CubicSpiline<N, ArgIndex>, double, 0>(other)
	{
#pragma unroll
		for(size_t i =0; i< N; i++){

			fD[i] = other.GetD()[i];
			fX[i] = other.GetX()[i];
		}
	}

	__hydra_host__ __hydra_device__ inline
	CubicSpiline<N>& operator=(CubicSpiline<N, ArgIndex> const& other )
	{
		if(this == &other) return *this;

		BaseFunctor<CubicSpiline<N, ArgIndex>, double, 0>::operator=(other);

#pragma unroll
		for(size_t i =0; i< N; i++){

			fD[i] = other.GetD()[i];
			fX[i] = other.GetX()[i];
		}
		return *this;
	}

	__hydra_host__ __hydra_device__
	inline const GReal_t* GetD() const {
		return fD;
	}

	__hydra_host__ __hydra_device__
	inline void SetD(unsigned int i, GReal_t value)  {
		fD[i]=value;
	}

	__hydra_host__ __hydra_device__
	inline const GReal_t* GetX() const {
		return fX;
	}

	__hydra_host__ __hydra_device__
		inline void SetX(unsigned int i, GReal_t value)  {
			fX[i]=value;
		}

	template<typename T>
	__hydra_host__ __hydra_device__
	inline double Evaluate(unsigned int n, T*x)  const {

		GReal_t X  = x[ArgIndex];

		GReal_t r = X<=fX[0]?fD[0]: X>=fX[N-1] ? fD[N-1] :spiline( X);

		return  CHECK_VALUE( r, "r=%f",r) ;
	}

	template<typename T>
	__hydra_host__ __hydra_device__
	inline double Evaluate(T x)  const {

		GReal_t X  = hydra::get<ArgIndex>(x); //mass

		GReal_t r = X<=fX[0]?fD[0]: X>=fX[N-1] ? fD[N-1] :spiline(X);

		return  CHECK_VALUE( r, "r=%f",r) ;
	}

private:

	__hydra_host__ __hydra_device__
	inline double spiline( const double x) const
	{
		using HYDRA_EXTERNAL_NS::thrust::min;

		const size_t i = HYDRA_EXTERNAL_NS::thrust::distance(fX,
							HYDRA_EXTERNAL_NS::thrust::lower_bound(HYDRA_EXTERNAL_NS::thrust::seq, fX, fX +N, x));
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

		const double c_i =  i==0  ? (copysign(1.0, p_i ) + copysign(1.0, s_i ))
				*min( fabs(s_i) , 0.5*fabs(p_i) ):
				i==N-2 ? (copysign(1.0, p_i ) + copysign(1.0, s_i ))
						*min( fabs(s_i) , 0.5*fabs(p_i) ):
					(copysign(1.0, s_im ) + copysign(1.0, s_i ))
				        *min(min(fabs(s_im), fabs(s_i)), 0.5*fabs(p_i) );

		const double c_ip =  (copysign(1.0, s_i ) + copysign(1.0, s_ip ))
									*min(min(fabs(s_ip), fabs(s_i)), 0.5*fabs(p_ip) );

		//calculates b
		const double b_i =  (-2*c_i - c_ip - 3*s_i)/h_i;

		//calculates a
		const double a_i = (c_i + c_ip - 2*s_i)/(h_i*h_i);

		//--------------------
		const double X = (x-fX[i]);

		return X*( X*(a_i*X + b_i) + c_i) + y_i;
	}

	GReal_t fX[N];
	GReal_t fD[N];

};

}  // namespace hydra



#endif /* CUBICSPILINE_H_ */
