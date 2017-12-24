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

#include <hydra/detail/external/thrust/copy.h>
#include <hydra/detail/external/thrust/iterator/zip_iterator.h>
#include <hydra/detail/external/thrust/execution_policy.h>
#include <hydra/detail/external/thrust/binary_search.h>

#include <cmath>
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


	CubicSpiline()=delete;

	template<typename Iterator1, typename Iterator2>
	CubicSpiline( Iterator1 xbegin, Iterator2 ybegin )
	{
		//populates fH and fX
		HYDRA_EXTERNAL_NS::thrust::copy(ybegin, ybegin+N, fD  );
		HYDRA_EXTERNAL_NS::thrust::copy( xbegin, xbegin+N,  fX);

		//calculates s_i
		GReal_t S[N]{};

		for(size_t i = 0; i< N-2; i++){
			S[i]=(fD[i+1]-fD[i])/(fX[i+1] -fX[i]);
		}

		//populates fC (y'_i)
		for(size_t i = 1; i< N-2; i++){

			GReal_t h_i = fX[i+1] -fX[i] ;
			GReal_t h_j = fX[i]   -fX[i-1] ;

			GReal_t 	p_i  = (S[i-1]*h_i + S[i]*h_j)/(h_i + h_j);

			fC[i] =  (std::copysign(1.0, S[i-1] ) + std::copysign(1.0, S[i] ))
							*std::min({fabs(S[i-1]), fabs(S[i]), 0.5*fabs(p_i)} );
		}

		GReal_t h_0 = fX[1] - fX[0] ;
		GReal_t h_1 = fX[2] - fX[1] ;

		GReal_t p_0  = S[0]*(1 + h_0/(h_0 + h_1)) - S[1]*h_0/(h_0 + h_1);

		fC[0] = (std::copysign(1.0, p_0 ) + std::copysign(1.0, S[0] ))
							*std::min( {fabs(S[0]) , 0.5*fabs(p_0)} ) ;

		GReal_t h_n = fX[N-1] - fX[N-2] ;
		GReal_t h_m = fX[N-2] - fX[N-3] ;

		GReal_t p_n  = S[N-2]*(1 + h_n/(h_n + h_m)) - S[N-3]*h_n/(h_n + h_m);

		fC[N-1] = (std::copysign(1.0, p_n ) + std::copysign(1.0, S[N-2] ))
											*std::min( {fabs(S[N-2]) , 0.5*fabs(p_n)} ) ;

		//populates fA and fB
		for(size_t i = 0; i< N-1; i++){

			fA[i] = (fC[i] + fC[i+1] - 2.0*S[i])/((fX[i+1] -fX[i])*(fX[i+1] -fX[i]));
			fB[i] = (3.0*S[i] - 2.0*fC[i]-fC[i+1])/(fX[i+1] -fX[i]);
		}
	}

	__host__ __device__
	CubicSpiline(CubicSpiline<N, ArgIndex> const& other ):
	BaseFunctor<CubicSpiline<N, ArgIndex>, double, 0>(other)
	{
#pragma unroll
		for(size_t i =0; i< N; i++){

			fA[i] = other.GetA()[i];
			fB[i] = other.GetB()[i];
			fC[i] = other.GetC()[i];
			fD[i] = other.GetD()[i];
			fX[i] = other.GetX()[i];
		}
	}

	__host__ __device__ inline
	CubicSpiline<N>* operator=(CubicSpiline<N, ArgIndex> const& other )
	{
		if(this == &other) return *this;

		BaseFunctor<CubicSpiline<N, ArgIndex>, double, 0>::operator=(other);

#pragma unroll
		for(size_t i =0; i< N; i++){

			fA[i] = other.GetA()[i];
			fB[i] = other.GetB()[i];
			fC[i] = other.GetC()[i];
			fD[i] = other.GetD()[i];
			fX[i] = other.GetX()[i];
		}
		return *this;
	}


	__host__ __device__ inline
	const GReal_t* GetA() const {
		return fA;
	}
	__host__ __device__ inline
	const GReal_t* GetB() const {
		return fB;
	}
	__host__ __device__ inline
	const GReal_t* GetC() const {
		return fC;
	}
	__host__ __device__ inline
	const GReal_t* GetD() const {
		return fD;
	}
	__host__ __device__ inline
	const GReal_t* GetX() const {
		return fX;
	}

	template<typename T>
	__host__ __device__ inline
	double Evaluate(unsigned int n, T*x)  const {

		GReal_t X  = x[ArgIndex];

		size_t interval = HYDRA_EXTERNAL_NS::thrust::distance(fX,
				HYDRA_EXTERNAL_NS::thrust::lower_bound(HYDRA_EXTERNAL_NS::thrust::seq, fX, fX +N, X));

		GReal_t r = X<=fX[0]?fD[0]: X>=fX[N-1] ? fD[N-1] :spiline(interval, X);

		return  CHECK_VALUE( r, "r=%f",r) ;
	}

	template<typename T>
	__host__ __device__ inline
	double Evaluate(T x)  const {

		GReal_t X  = hydra::get<ArgIndex>(x); //mass
		size_t interval = HYDRA_EXTERNAL_NS::thrust::distance(fX,
					HYDRA_EXTERNAL_NS::thrust::lower_bound(HYDRA_EXTERNAL_NS::thrust::seq, fX, fX +N, X));

		GReal_t r = X<=fX[0]?fD[0]: X>=fX[N-1] ? fD[N-1] :spiline(interval, X);

			return  CHECK_VALUE( r, "r=%f",r) ;
	}

private:

	GReal_t spiline(size_t i, double x) const
	{
		GReal_t X = (x-fX[i]);

		return fA[i]*X*X*X + fB[i]*X*X + fC[i]*X + fD[i];
	}

	GReal_t fX[N];
	GReal_t fA[N];
	GReal_t fB[N];
	GReal_t fC[N]; //y'_i
	GReal_t fD[N]; //y_i

};

}  // namespace hydra



#endif /* CUBICSPILINE_H_ */
