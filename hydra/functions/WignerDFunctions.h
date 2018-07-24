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
 * WignerD.h
 *
 *  Created on: 25/12/2017
 *      Author: Antonio Augusto Alves Junior
 */

#ifndef WIGNERD_H_
#define WIGNERD_H_


#include <hydra/detail/Config.h>
#include <hydra/detail/BackendPolicy.h>
#include <hydra/Types.h>
#include <hydra/Function.h>
#include <hydra/Pdf.h>
#include <hydra/detail/Integrator.h>
#include <hydra/detail/utility/CheckValue.h>
#include <hydra/Tuple.h>
#include <hydra/functions/Utils.h>
#include <tuple>
#include <limits>
#include <stdexcept>
#include <assert.h>
#include <utility>
#include <ratio>
#include <hydra/detail/external/thrust/extrema.h>

namespace hydra {

/**
 * \class WignerD
 *
 *  Calculates the beta-term
 *                   \f[ d^j_{mn}(beta) \f]
 *  in the matrix element of the finite rotation operator
 *  (Wigner's D-function), according to formula 4.3.1(3) in
 *  D.A. Varshalovich, A.N. Moskalev, and V.K. Khersonskii,
 *  Quantum Theory of Angular Momentum, World Scientific,
 *  Singapore 1988. CERNLIB DDJMNB function.
 */

template<unsigned int J, int M, int N, unsigned int Denominator=1,  unsigned int ArgIndex=0>
class WignerD: public BaseFunctor<WignerD<J, M,Denominator,ArgIndex>, double, 0>
{
	constexpr static int JPM  = detail::nearest_int<J+M,Denominator>::value;
	constexpr static int JPN  = detail::nearest_int<J+N,Denominator>::value;
	constexpr static int JMM  = detail::nearest_int<J-M,Denominator>::value;
	constexpr static int JMN  = detail::nearest_int<J-N,Denominator>::value;
	constexpr static int MPN  = detail::nearest_int<M+N,Denominator>::value;

	static_assert((JPM <0 || JPN < 0 || JMM < 0 || JMN < 0 || J < 0 || J > 25 ) ,
	                  "[Hydra::WignerD] : Wrong parameters combination");
public:

	WignerD()=default;

	__hydra_dual__
	WignerD( WignerD<J, M,Denominator, ArgIndex> const& other):
		BaseFunctor<WignerD<J, M,Denominator,ArgIndex>, double, 0>(other)
		{}

	__hydra_dual__
	WignerD<J, M,Denominator,ArgIndex>& operator=( WignerD<J, M,Denominator,ArgIndex> const& other){

		if(this == &other) return *this;
		BaseFunctor<WignerD<J, M,Denominator,ArgIndex>, double, 0>::operator=(other);

		return *this;
	}

	template<typename T>
	__hydra_dual__ inline
	double Evaluate(unsigned int, T*x)  const	{

		double beta = x[ArgIndex] ;
		double r = wignerd(beta);

		return  CHECK_VALUE(r, "r=%f", r);

	}

	template<typename T>
	__hydra_dual__ inline
	double Evaluate(T x)  const {

		double beta =  get<ArgIndex>(x);
		double r = wignerd(beta);

		return  CHECK_VALUE(r, "r=%f", r);

	}

private:

	__hydra_dual__ inline
	double wignerd( double beta ) const {

		double r = (beta < 0 || beta > 2.0*PI) ? printf("HYDRA WARNING: WignerD: Illegal argument  beta=%g\n", beta):
		    (beta == 0)  ? (JPM == JPN ) :
		    (beta == PI) ? (JPM == JMN ) - 2*(::abs(JPM)%2 == 1):
	    	(beta == 2.0*PI) ? (JPM == JPN) - 2*(::abs(MPN)%2 == 1) : wdf(beta);

		return r;

	}

	__hydra_dual__
	inline double wdf( double  b) const {

		using HYDRA_EXTERNAL_NS::thrust::max;
		using HYDRA_EXTERNAL_NS::thrust::min;

		double r = 0;
		double s  = ::log(::sin(b/2.0));
		double c  = ::log(::fabs(::cos(b/2.0)));
		double rt = 0.5*(fcl[JPM]+fcl[JMM]+fcl[JPN]+fcl[JMN]);
		int k0    = max(0,MPN);
		int kq    = (b > PI) ? k0 + JPM + MPN: k0 + JPM;

		double q  = 1 - 2*(kq%2 == 1);
		kq = k0+k0;
		double cx = kq-MPN;
		double sx = JPM+JPN-kq;

		for( int k=k0; k<= min(JPM,JPN); k++)
		{
			r  += q*::exp(rt-fcl[k]-fcl[JPM-k]-fcl[JPM-k]-fcl[k-MPN]+ cx*c+sx*s);
			cx += 2;
			sx -= 2;
			q   = -q;
		}

		return CHECK_VALUE(r, "r=%f", r);
	}

	static constexpr double fcl[51] = { 0.0 ,0.0,
            6.93147180559945309e-1 ,1.79175946922805500e00,
            3.17805383034794562e00 ,4.78749174278204599e00,
            6.57925121201010100e00 ,8.52516136106541430e00,
            1.06046029027452502e01 ,1.28018274800814696e01,
            1.51044125730755153e01 ,1.75023078458738858e01,
            1.99872144956618861e01 ,2.25521638531234229e01,
            2.51912211827386815e01 ,2.78992713838408916e01,
            3.06718601060806728e01 ,3.35050734501368889e01,
            3.63954452080330536e01 ,3.93398841871994940e01,
            4.23356164607534850e01 ,4.53801388984769080e01,
            4.84711813518352239e01 ,5.16066755677643736e01,
            5.47847293981123192e01 ,5.80036052229805199e01,
            6.12617017610020020e01 ,6.45575386270063311e01,
            6.78897431371815350e01 ,7.12570389671680090e01,
            7.46582363488301644e01 ,7.80922235533153106e01,
            8.15579594561150372e01 ,8.50544670175815174e01,
            8.85808275421976788e01 ,9.21361756036870925e01,
            9.57196945421432025e01 ,9.93306124547874269e01,
            1.02968198614513813e02 ,1.06631760260643459e02,
            1.10320639714757395e02 ,1.14034211781461703e02,
            1.17771881399745072e02 ,1.21533081515438634e02,
            1.25317271149356895e02 ,1.29123933639127215e02,
            1.32952575035616310e02 ,1.36802722637326368e02,
            1.40673923648234259e02 ,1.44565743946344886e02,
            1.48477766951773032e02 };


};


}  // namespace hydra




#endif /* WIGNERD_H_ */
